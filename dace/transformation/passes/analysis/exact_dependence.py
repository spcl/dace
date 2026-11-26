# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The iteration order of the conflicts between two blocks of one loop body, decided exactly by ISL.

Each access is stated with the loops and maps around it inside its block as extra dimensions, and with its subset's
range per dimension as one shared element dimension: two instances conflict when they touch one element. For the
loop's instance counts ``k1`` (the first block) and ``k2`` (the second), one emptiness query per sign of ``k2 - k1``
says whether a conflict of that sign exists. Anything ISL cannot state (a non-affine subscript or bound, a non-unit
stride inside a block, a symbol the loop assigns) is no answer, never a permission.
"""

from typing import Any, NamedTuple

import islpy as isl
import sympy

from dace import subsets, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import narrowing, nodes
from dace.sdfg.analysis import polyhedral_isl
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis


class Access(NamedTuple):
    """One access of a container in a block: its subset, whether it writes, and the ``(variable, first, last)`` of
    each loop and map around it inside the block."""

    subset: subsets.Range
    write: bool
    domains: list[tuple[str, Any, Any]]


def loop_domain(loop: LoopRegion) -> tuple[str, Any, Any] | None:
    """``(variable, first, last)`` of a unit-stride ``loop`` (either direction), else ``None``."""
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if not loop.loop_variable or start is None or end is None or stride is None:
        return None
    stride = symbolic.simplify(stride)
    if stride == 1:
        return loop.loop_variable, start, end
    if stride == -1:
        return loop.loop_variable, end, start
    return None


def enclosing_domains(state: SDFGState, node: nodes.Node, block: ControlFlowBlock) -> list[tuple[str, Any, Any]] | None:
    """The domains of the loops between ``state`` and ``block`` and of the maps around ``node``; ``None`` when one is
    not a unit-stride range."""
    found: list[tuple[str, Any, Any]] = []
    region = None if state is block else state.parent_graph
    # Climb from the state to the block, the block included: each loop on the way bounds a variable the access uses
    while region is not None:
        if isinstance(region, LoopRegion):
            domain = loop_domain(region)
            if domain is None:
                return None
            found.append(domain)
        if region is block:
            break
        region = region.parent_graph
    entry = state.entry_node(node)
    # Climb the map scopes around the node: each map parameter is bounded by its range
    while entry is not None:
        # Over the map's parameters and ranges: each must step by one
        for param, (first, last, step) in zip(entry.map.params, entry.map.range):
            if symbolic.simplify(step) != 1:
                return None
            found.append((param, first, last))
        entry = state.entry_node(entry)
    return found


def block_accesses(block: ControlFlowBlock, name: str) -> list[Access] | None:
    """Every value access of ``name`` in ``block``; ``None`` when one has no unit-stride range subset."""
    blocks = [block] if isinstance(block, SDFGState) else list(block.all_control_flow_blocks())
    states = [b for b in blocks if isinstance(b, SDFGState)]
    # A nested SDFG's accesses are out of sight here: no answer rather than a missed conflict
    if any(isinstance(n, nodes.NestedSDFG) for state in states for n in state.nodes()):
        return None
    found: list[Access] = []
    # Over the block's states and their access nodes of ``name``: each value edge is one access
    for state in states:
        for node in state.data_nodes():
            if node.data != name:
                continue
            domains = enclosing_domains(state, node, block)
            if domains is None:
                return None
            edges = [(e.data.get_dst_subset(e, state), True) for e in state.in_edges(node) if not e.data.is_empty()]
            edges += [(e.data.get_src_subset(e, state), False) for e in state.out_edges(node) if not e.data.is_empty()]
            # Over the node's value edges: the subset each stores or loads
            for subset, write in edges:
                if not isinstance(subset, subsets.Range) or any(
                    symbolic.simplify(step) != 1 for low, high, step in subset.ranges
                ):
                    return None
                found.append(Access(subset, write, domains))
    return found


def conflict_signs(
    loop: LoopRegion, first: Access, second: Access, assigned: OrderedSet[str]
) -> OrderedSet[int] | None:
    """The signs of ``k2 - k1`` over the instance pairs at which ``first`` (loop instance ``k1``) and ``second``
    (instance ``k2``) touch one element; ``None`` when ISL cannot state them."""
    domain = loop_domain(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if domain is None or stride is None or first.subset.dims() != second.subset.dims():
        return None
    var = loop.loop_variable
    k = {1: sympy.Symbol(f"{var}_k1", integer=True), 2: sympy.Symbol(f"{var}_k2", integer=True)}
    start = narrowing.as_expr(loop_analysis.get_init_assignment(loop))
    end = narrowing.as_expr(loop_analysis.get_loop_end(loop))
    step = int(symbolic.simplify(stride))
    dims: list[str] = [k[1].name, k[2].name]
    cons: list[sympy.Expr] = []
    elements = [sympy.Symbol(f"{var}_x{d}", integer=True) for d in range(first.subset.dims())]
    dims += [e.name for e in elements]
    # Over the two accesses: each one's loop instance, inner variables and subset ranges, renamed per side
    for side, access in ((1, first), (2, second)):
        mapping: dict[str, sympy.Expr] = {var: start + step * k[side]}
        cons += [k[side], end - mapping[var] if step > 0 else mapping[var] - end]
        # Over the loops and maps around the access: a variable of its own on this side, within its bounds
        for name, low, high in access.domains:
            inner = sympy.Symbol(f"{name}_s{side}", integer=True)
            dims.append(inner.name)
            low_e = polyhedral_isl.subs_by_name(narrowing.as_expr(low), mapping)
            high_e = polyhedral_isl.subs_by_name(narrowing.as_expr(high), mapping)
            mapping[name] = inner
            cons += [inner - low_e, high_e - inner]
        # Over the subset's dimensions: the shared element lies within this access's range
        for element, (low, high, unit) in zip(elements, access.subset.ranges):
            cons += [
                element - polyhedral_isl.subs_by_name(narrowing.as_expr(low), mapping),
                polyhedral_isl.subs_by_name(narrowing.as_expr(high), mapping) - element,
            ]
    names = OrderedSet(s.name for c in cons for s in narrowing.free_symbols(c))
    params = [n for n in names if n not in dims]
    # A symbol the loop assigns changes between instances: the relation would not hold it fixed
    if any(p in assigned or p == var for p in params):
        return None
    signs: OrderedSet[int] = OrderedSet()
    # Over the three signs of ``k2 - k1``: one exact emptiness query each
    for sign, extra in ((-1, [k[1] - k[2] - 1]), (0, [k[2] - k[1], k[1] - k[2]]), (1, [k[2] - k[1] - 1])):
        try:
            empty = polyhedral_isl.is_domain_empty(dims, params, cons + extra)
        except (ValueError, TypeError, isl.Error):
            return None
        if not empty:
            signs.add(sign)
    return signs


def runs_whole_before(loop: LoopRegion, earlier: ControlFlowBlock, later: ControlFlowBlock, name: str) -> bool:
    """Whether every instance of ``earlier`` may run before every instance of ``later`` for ``name``: no conflict on
    it, one a write, has ``later``'s instance in an earlier iteration than ``earlier``'s."""
    first, second = block_accesses(earlier, name), block_accesses(later, name)
    if first is None or second is None:
        return False
    assigned = OrderedSet(n for e in loop.all_interstate_edges() for n in e.data.assignments)
    # Over the pairs of accesses, one of them a write: each must keep ``later`` in the same or a later iteration
    for a in first:
        for b in second:
            if not (a.write or b.write):
                continue
            signs = conflict_signs(loop, a, b, assigned)
            if signs is None or -1 in signs:
                return False
    return True
