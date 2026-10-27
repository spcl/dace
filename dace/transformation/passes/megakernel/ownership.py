# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Thread ownership of megakernel stage iterations: a later stage needs no grid barrier after an earlier one where
every element both touch is touched by the same thread in both, whatever range each stage covers.

A stage is *owned* by ``k`` of its map parameters (its key) when one of its writes (its anchor) indexes one dimension
of its container with each of them as ``param + offset``. Its iteration runs in thread
``(p1 + M1 * p2 + M1 * M2 * p3 + ...) % S`` of ``S``, with chain-wide multipliers ``M``; with ``k = 1``, a thread runs
whole columns, its other parameters a loop inside it. An access is *keyed* when each key parameter indexes exactly one
dimension of the container, as ``p_j + offset_j`` with an offset no thread changes; its *signature* is those dimensions
and offsets. Two owned stages touch an element of a container in the same thread when both access it keyed with the
same signature: element ``c`` then belongs to iteration ``c - offset`` in both, whatever the other dimensions index.
Accesses with several signatures, through an index array or over a range of a key dimension are *off key*.
Each stage picks its key greedily, in program order: the parameters whose signatures agree with those of the stages
before it, then those indexing unit-stride dimensions (coalesced accesses).
"""

import copy
import itertools
from dataclasses import dataclass, field

from dace import SDFG, SDFGState, data, dtypes, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation.passes.megakernel.grid_stride import is_small

#: A container a stage accesses, at which elements (``None``: unknown), and whether it writes them
Access = tuple[str, subsets.Subset | None, bool]
#: The dimension and offset each key parameter indexes in a keyed access
Signature = tuple[tuple[int, str], ...]


@dataclass(slots=True)
class Owner:
    """An owned stage: its key parameters, fastest first, its anchor's extents along their dimensions, the signatures
    of the containers it accesses keyed, and the containers it accesses off key."""

    params: tuple[str, ...]
    extents: tuple[symbolic.SymbolicType, ...]
    keyed: dict[str, Signature] = field(default_factory=dict)
    off: frozenset[str] = frozenset()


@dataclass(slots=True)
class OwnershipPlan:
    """Which stages are owned, with ``k`` key parameters and the chain-wide key ``multipliers``."""

    k: int
    multipliers: list[symbolic.SymbolicType] = field(default_factory=list)
    owners: dict[SDFGState, Owner] = field(default_factory=dict)


def unit_stride(sdfg: SDFG, name: str, dim: int) -> bool:
    desc = sdfg.arrays[name]
    return isinstance(desc, data.Array) and desc.strides[dim] == 1


def leaf_edges(state: SDFGState, scope_nodes) -> list:
    """The edges of ``state`` with a memlet that touch one of ``scope_nodes`` other than a scope entry or exit or a
    nested SDFG: their memlets say which elements one iteration accesses."""
    leaves = [n for n in scope_nodes if not isinstance(n, (nodes.EntryNode, nodes.ExitNode, nodes.NestedSDFG))]
    edges = {e for n in leaves for e in state.all_edges(n)}
    return [
        e
        for e in edges
        if e.data.data is not None
        and not isinstance(e.src, nodes.NestedSDFG)
        and not isinstance(e.dst, nodes.NestedSDFG)
    ]


def edge_accesses(edge) -> list[Access]:
    if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):  # A copy: off key
        return [(edge.src.data, None, False), (edge.dst.data, None, True)]
    return [(edge.data.data, edge.data.subset, isinstance(edge.dst, (nodes.ExitNode, nodes.AccessNode)))]


def bound_in(sdfg: SDFG) -> set[str]:
    """The symbols ``sdfg`` and the SDFGs in it bind: map parameters, loop variables and interstate assignments."""
    names = set()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            names |= set(node.map.params)
    for nested in sdfg.all_sdfgs_recursive():
        names |= {b.loop_variable for b in nested.all_control_flow_blocks() if isinstance(b, LoopRegion)}
        names = names.union(*(e.data.assignments.keys() for e in nested.all_interstate_edges()))
    return names - {""}


def whole_connectors(state: SDFGState, node: nodes.NestedSDFG, outer: dict) -> set[str]:
    """The connectors of ``node`` that pass a whole container of ``state``, of the same rank inside."""
    return {
        conn
        for conn, memlet in outer.items()
        if memlet.data is not None
        and memlet.subset == subsets.Range.from_array(state.sdfg.arrays[memlet.data])
        and len(node.sdfg.arrays[conn].shape) == len(state.sdfg.arrays[memlet.data].shape)
    }


def nested_accesses(state: SDFGState, node: nodes.NestedSDFG) -> tuple[list[Access], set[str]]:
    """The accesses inside the nested SDFG ``node``, in the containers and symbols of ``state``, and the symbols the
    nested SDFG binds. A container passed in part keeps its connector's memlet."""
    outer = {e.dst_conn: e.data for e in state.in_edges(node) if e.dst_conn}
    outer |= {e.src_conn: e.data for e in state.out_edges(node) if e.src_conn}
    whole = whole_connectors(state, node, outer)
    accesses = [(m.data, m.subset, conn in node.out_connectors) for conn, m in outer.items() if conn not in whole]
    mapping = {symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v) for k, v in node.symbol_mapping.items()}
    bound = bound_in(node.sdfg)
    for inner in node.sdfg.states():
        inner_accesses, inner_bound = scope_accesses(inner, inner.nodes())
        bound |= inner_bound
        for name, subset, write in inner_accesses:
            if name in whole:
                if subset is not None:
                    subset = copy.deepcopy(subset)
                    subset.replace(mapping)
                accesses.append((outer[name].data, subset, write))
    return accesses, bound


def scope_accesses(state: SDFGState, scope_nodes) -> tuple[list[Access], set[str]]:
    """The accesses of ``scope_nodes`` in ``state``, nested SDFGs included, and the symbols bound inside them."""
    accesses = [a for e in leaf_edges(state, scope_nodes) for a in edge_accesses(e)]
    bound = set()
    for node in scope_nodes:
        if isinstance(node, nodes.MapEntry):
            bound |= set(node.map.params)
        elif isinstance(node, nodes.NestedSDFG):
            inner, inner_bound = nested_accesses(state, node)
            accesses += inner
            bound |= inner_bound
    return accesses, bound


def names_in(expr) -> set[str]:
    return {str(sym) for sym in symbolic.pystr_to_symbolic(expr).free_symbols}


def offset_of(index, param: str, frozen: set[str], params: list[str]) -> str | None:
    """The offset of ``index`` from ``param`` when ``index`` is ``param + offset`` with an offset free of parameters
    and ``frozen`` symbols, as a string (symbols of one name but other assumptions compare equal)."""
    index = symbolic.pystr_to_symbolic(index)
    symbol = next(sym for sym in index.free_symbols if str(sym) == param)
    offset = symbolic.simplify(index - symbol)
    if names_in(offset) & (set(params) | frozen):
        return None
    return str(offset)


def signature(subset: subsets.Subset | None, key: tuple[str, ...], params: list[str], frozen: set[str]):
    """The signature of an access to ``subset`` under ``key``, or ``None`` off key."""
    if not isinstance(subset, subsets.Range):
        return None
    result = []
    for param in key:
        dims = [d for d, (b, e, s) in enumerate(subset) if param in names_in(b) | names_in(e)]
        if len(dims) != 1:
            return None
        begin, end, _ = subset[dims[0]]
        offset = offset_of(begin, param, frozen, params) if symbolic.simplify(end - begin) == 0 else None
        if offset is None:
            return None
        result.append((dims[0], offset))
    return tuple(result)


@dataclass(slots=True)
class Stage:
    """What choosing a stage's key needs: its accesses to shared containers, parameters, usable key parameters and
    the symbols bound in the megakernel or inside the stage."""

    accesses: list[Access]
    params: list[str]
    usable: list[str]
    frozen: set[str]


def stage_of(state: SDFGState, entry: nodes.MapEntry, shared: set[str], small: set[str], frozen: set[str]) -> Stage:
    accesses, bound = scope_accesses(state, state.scope_subgraph(entry, False, False).nodes())
    ranges = zip(entry.map.params, entry.map.range, entry.map.range.size())
    usable = [p for p, (b, e, s), size in ranges if s == 1 and not is_small(size, small)]
    return Stage([a for a in accesses if a[0] in shared], list(entry.map.params), usable, frozen | bound)


def owner_by(sdfg: SDFG, stage: Stage, key: tuple[str, ...]) -> Owner | None:
    """``stage`` owned by ``key``, or ``None`` when no write anchors it."""
    signatures: dict[str, set] = {}
    extents = None
    for name, subset, write in stage.accesses:
        sig = signature(subset, key, stage.params, stage.frozen)
        signatures.setdefault(name, set()).add(sig)
        if write and sig is not None and extents is None:
            extents = tuple(sdfg.arrays[name].shape[d] for d, o in sig)
    if extents is None:
        return None
    off = frozenset(name for name, found in signatures.items() if None in found or len(found) > 1)
    keyed = {name: next(iter(found)) for name, found in signatures.items() if name not in off}
    return Owner(key, extents, keyed, off)


def score(sdfg: SDFG, owner: Owner, seen: dict[str, Signature]) -> tuple[int, int]:
    """How many containers ``owner`` accesses with the signature earlier stages chose, and how many key dimensions
    of its keyed accesses have unit stride."""
    agree = sum(seen.get(name) == sig for name, sig in owner.keyed.items())
    coalesced = sum(unit_stride(sdfg, name, sig[0][0]) for name, sig in owner.keyed.items())
    return agree, coalesced


def own_stride(
    sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, owner: Owner, plan: OwnershipPlan, tid: str, nthreads: str
) -> nodes.MapEntry:
    """Lowers the owned stage ``entry`` so that thread ``tid`` of ``nthreads`` runs exactly the iterations it owns:
    sequential loops over the slower key parameters, then the fastest one from the first value the thread owns in
    steps of ``nthreads``, then the remaining parameters.

    :return: The loop over the fastest key parameter.
    """
    params = entry.map.params
    slow, fastest = list(reversed(owner.params[1:])), owner.params[0]
    order = [params.index(p) for p in slow + [fastest]]
    if len(order) < len(params):
        entry, rest = xfh.extract_map_dims(sdfg, entry, order)
        rest.map.schedule = dtypes.ScheduleType.Sequential
    else:
        xfh.permute_map(entry, order)
    if slow:
        entry, inner = xfh.extract_map_dims(sdfg, entry, list(range(len(slow))))
        entry.map.schedule = dtypes.ScheduleType.Sequential
        entry = inner
    entry.map.schedule = dtypes.ScheduleType.Sequential
    thread, count = symbolic.pystr_to_symbolic(tid), symbolic.pystr_to_symbolic(nthreads)
    linear_rest, scale = 0, 1
    for multiplier, param in zip(plan.multipliers, owner.params[1:]):
        scale = scale * multiplier
        linear_rest = linear_rest + scale * symbolic.pystr_to_symbolic(param)
    begin, end, _ = entry.map.range[0]
    first = begin + (thread - begin - linear_rest) % count
    entry.map.range = subsets.Range([(first, end, count)])
    return entry


def plan_ownership(
    stages: list[tuple[SDFGState, nodes.MapEntry]], k: int, shared: set[str], small: set[str], frozen: set[str]
) -> OwnershipPlan:
    """Owns, in order, every stage some ``k`` of its parameters anchor, by the key that scores best (:func:`score`);
    the first owned stage's extents give the multipliers."""
    plan = OwnershipPlan(k)
    seen: dict[str, Signature] = {}
    for state, entry in stages:
        stage = stage_of(state, entry, shared, small, frozen)
        owners = [owner_by(state.sdfg, stage, key) for key in itertools.permutations(stage.usable, k)]
        owners = [o for o in owners if o is not None]
        if not owners:
            continue
        owner = max(owners, key=lambda o: score(state.sdfg, o, seen))
        plan.owners[state] = owner
        for name, sig in owner.keyed.items():
            seen.setdefault(name, sig)
    if plan.owners:
        plan.multipliers = list(next(iter(plan.owners.values())).extents[:-1])
    return plan
