# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop a guard ``count != 0`` that a predicate of the same guard already implies.

The pattern is Fortran's early exit over a count of flags (ICON ``velocity_tendencies``)::

    count = 0
    for i: flag[i] = ...; if flag[i]: count += 1      # map A, WCR-Sum into count
    if count == 0: skip
    for i: if flag[i]: S(i)                         # map B, guarded by (count != 0) and flag[i]

Map A adds a nonnegative amount per iteration and at least 1 where ``flag[i]`` holds, so ``flag[i]`` implies
``count != 0``: the ``count`` conjunct of map B's guard never changes the outcome and is dropped. ``count`` is then
dead, and maps A and B can fuse. The guard is the CPU's early exit, so this is an optimization choice
(placement-dependent), not a canonicalization: callers apply it per match (:func:`implied_guards`,
:func:`drop_implied_guard`) or everywhere (:class:`DropImpliedGuards`).
"""

import ast
import dataclasses
import itertools
import re

import sympy

from dace import SDFG, SDFGState, dtypes, graphlib, symbolic
from dace.frontend import operations
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.interstate.condition_fusion import flatten_and


@dataclasses.dataclass
class ImpliedGuard:
    """One droppable conjunct: ``conjunct`` of branch ``branch`` of ``block``, inside map ``guarded`` of ``state``."""

    state: SDFGState
    guarded: nodes.MapEntry
    block: ConditionalBlock
    branch: int
    conjunct: sympy.Basic


@dataclasses.dataclass
class Predicate:
    """A symbol assigned once from one element of a container."""

    symbol: str
    data: str
    index: tuple[sympy.Basic, ...]


COUNT = sympy.Symbol("guard_count", integer=True, positive=True)


def nonzero_when_positive(conjunct: sympy.Basic, name: str) -> bool:
    """Whether ``conjunct`` holds for every positive integer value of ``name`` (``count != 0``, ``count > 0``...)."""
    if {str(s) for s in conjunct.free_symbols} != {name}:
        return False
    held = conjunct.subs(dict.fromkeys(conjunct.free_symbols, COUNT))
    return held is sympy.true or held == True  # noqa: E712 -- sympy booleans compare to Python True


def map_body(state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG | None:
    """The one nested SDFG a one-parameter map runs (besides access nodes), or ``None``."""
    if len(entry.map.params) != 1:
        return None
    inner = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    nested = [n for n in inner if isinstance(n, nodes.NestedSDFG)]
    if len(nested) != 1 or any(not isinstance(n, (nodes.NestedSDFG, nodes.AccessNode)) for n in inner):
        return None
    return nested[0]


def outer_index(nsdfg: nodes.NestedSDFG, index: tuple[sympy.Basic, ...]) -> tuple[sympy.Basic, ...]:
    mapping = {symbolic.pystr_to_symbolic(k): v for k, v in nsdfg.symbol_mapping.items()}
    return tuple(symbolic.pystr_to_symbolic(str(i)).subs(mapping) for i in index)


def predicate(sdfg: SDFG, symbol: str) -> Predicate | None:
    """The single interstate assignment ``symbol = data[index]`` in ``sdfg`` (all regions, not nested SDFGs)."""
    found = [
        edge.data.assignments[symbol]
        for region in sdfg.all_control_flow_regions(recursive=False)
        for edge in region.edges()
        if symbol in edge.data.assignments
    ]
    if len(found) != 1:
        return None
    try:
        expr = ast.parse(found[0], mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Subscript) or not isinstance(expr.value, ast.Name):
        return None
    parts = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else [expr.slice]
    return Predicate(symbol, expr.value.id, tuple(symbolic.pystr_to_symbolic(ast.unparse(p)) for p in parts))


def assignment_edge(sdfg: SDFG, symbol: str):
    for region in sdfg.all_control_flow_regions(recursive=False):
        for edge in region.edges():
            if symbol in edge.data.assignments:
                return region, edge
    return None, None


def assigned_symbols(sdfg: SDFG) -> list[str]:
    return sorted(
        {k for r in sdfg.all_control_flow_regions(recursive=False) for e in r.edges() for k in e.data.assignments}
    )


def writes(sdfg: SDFG, data: str) -> list[tuple[SDFGState, nodes.AccessNode]]:
    return [(s, n) for s in sdfg.states() for n in s.data_nodes() if n.data == data and s.in_degree(n) > 0]


def implying_symbols(block: ConditionalBlock, branch: int, conjunct: sympy.Basic) -> list[str]:
    """Bare symbols that gate every effect of the branch: the other conjuncts, or the conjuncts of the one
    single-branch conditional the branch holds besides empty states."""
    condition, body = block.branches[branch]
    conjuncts = flatten_and(symbolic.pystr_to_symbolic(condition.as_string))
    same = [str(c) for c in conjuncts if c is not conjunct and isinstance(c, sympy.Symbol)]
    if same:
        return same
    blocks = list(body.nodes())
    inner = [b for b in blocks if isinstance(b, ConditionalBlock)]
    rest = [b for b in blocks if not isinstance(b, ConditionalBlock)]
    if len(inner) != 1 or len(inner[0].branches) != 1 or inner[0].branches[0][0] is None:
        return []
    if not all(isinstance(b, SDFGState) and b.number_of_nodes() == 0 for b in rest):
        return []
    deeper = flatten_and(symbolic.pystr_to_symbolic(inner[0].branches[0][0].as_string))
    return [str(c) for c in deeper if isinstance(c, sympy.Symbol)]


def tasklet_values(tasklet: nodes.Tasklet, inputs: dict[str, sympy.Basic]) -> dict[str, sympy.Basic] | None:
    if tasklet.language != dtypes.Language.Python:
        return None
    try:
        body = ast.parse(tasklet.code.as_string).body
    except SyntaxError:
        return None
    out: dict[str, sympy.Basic] = {}
    for stmt in body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return None
        expr = symbolic.pystr_to_symbolic(ast.unparse(stmt.value))
        out[stmt.targets[0].id] = expr.subs({sympy.Symbol(k): v for k, v in inputs.items()})
    return out


def run_state(state: SDFGState, env: dict[str, sympy.Basic], fresh: itertools.count) -> None:
    """Symbolically execute the scalar dataflow of ``state``; anything else becomes an unknown."""
    produced: dict[tuple[nodes.Node, str], sympy.Basic] = {}
    for node in sdutil.dfs_topological_sort(state):
        if isinstance(node, nodes.AccessNode):
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                if isinstance(edge.src, nodes.AccessNode):
                    value = env.get(edge.src.data)
                else:
                    value = produced.get((edge.src, edge.src_conn))
                scalar = edge.data.volume == 1 and state.sdfg.arrays[node.data].total_size == 1
                env[node.data] = value if scalar and value is not None else sympy.Symbol(f"unknown{next(fresh)}")
        elif isinstance(node, nodes.Tasklet):
            inputs = {
                e.dst_conn: env.get(e.src.data, sympy.Symbol(f"unknown{next(fresh)}"))
                for e in state.in_edges(node)
                if isinstance(e.src, nodes.AccessNode) and e.dst_conn
            }
            values = tasklet_values(node, inputs) or {}
            for conn in node.out_connectors:
                produced[(node, conn)] = values.get(conn, sympy.Symbol(f"unknown{next(fresh)}"))
        elif not isinstance(node, nodes.AccessNode):
            for conn in node.out_connectors:
                produced[(node, conn)] = sympy.Symbol(f"unknown{next(fresh)}")


def run_region(region: ControlFlowRegion, env: dict[str, sympy.Basic], fresh: itertools.count) -> bool:
    """Execute an acyclic region whose branches resolve under ``env``; ``False`` when it cannot."""
    if any(not e.data.is_unconditional() for e in region.edges()):
        return False
    for block in graphlib.topological_sort(region.nx):
        if isinstance(block, SDFGState):
            run_state(block, env, fresh)
        elif isinstance(block, ConditionalBlock):
            for condition, body in block.branches:
                if condition is None:
                    taken = sympy.true
                else:
                    taken = symbolic.pystr_to_symbolic(condition.as_string).subs(
                        {sympy.Symbol(k): v for k, v in env.items()}
                    )
                if taken not in (sympy.true, sympy.false):
                    return False
                if taken is sympy.true:
                    if not run_region(body, env, fresh):
                        return False
                    break
        else:
            return False
    return True


def contribution(nsdfg: SDFG, output: str, flag_symbol: str, held: bool) -> sympy.Basic | None:
    env = {flag_symbol: sympy.true if held else sympy.false}
    if not run_region(nsdfg, env, itertools.count()):
        return None
    return env.get(output)


def at_least(value: sympy.Basic | None, bound: int) -> bool:
    if value is None or value.free_symbols:
        return False
    return bool(value.is_integer) and bool(value >= bound)


def counting_map(state: SDFGState, count_node: nodes.AccessNode):
    """The map that WCR-sums into ``count_node``, its nested SDFG, and the nested output name; or ``None``."""
    in_edges = state.in_edges(count_node)
    if len(in_edges) != 1 or not isinstance(in_edges[0].src, nodes.MapExit):
        return None
    edge = in_edges[0]
    if edge.data.wcr is None or operations.detect_reduction_type(edge.data.wcr) != dtypes.ReductionType.Sum:
        return None
    root = state.memlet_path(edge)[0]
    if isinstance(root.src, nodes.AccessNode):
        inner = state.in_edges(root.src)
        if len(inner) != 1:
            return None
        root = inner[0]
    if not isinstance(root.src, nodes.NestedSDFG):
        return None
    entry = state.entry_node(edge.src)
    if map_body(state, entry) is not root.src:
        return None
    return entry, root.src, root.src_conn


def count_initialized_nonnegative(state: SDFGState, count_node: nodes.AccessNode, entry: nodes.MapEntry) -> bool:
    """Every other write of the count is a nonnegative constant, in this state, before the counting map."""
    sdfg = state.sdfg
    for other_state, node in writes(sdfg, count_node.data):
        if node is count_node:
            continue
        if other_state is not state or not graphlib.has_path(state.nx, node, entry):
            return False
        for edge in state.in_edges(node):
            if not isinstance(edge.src, nodes.Tasklet) or any(not e.data.is_empty() for e in state.in_edges(edge.src)):
                return False
            values = tasklet_values(edge.src, {})
            if values is None or not at_least(values.get(edge.src_conn), 0):
                return False
    return True


def flag_reaches_unchanged(
    state: SDFGState,
    counter: tuple[nodes.MapEntry, nodes.NestedSDFG],
    guarded: tuple[nodes.MapEntry, nodes.NestedSDFG],
    seen: Predicate,
    used: Predicate,
) -> bool:
    """The flag element the guard reads is the one the count saw: same element, written only before the count
    reads it, and by no one between the two maps."""
    centry, cnest = counter
    gentry, gnest = guarded
    if seen.data not in cnest.in_connectors and seen.data not in cnest.out_connectors:
        return False
    if writes(gnest.sdfg, used.data):
        return False
    cparam, gparam = (symbolic.pystr_to_symbolic(e.map.params[0]) for e in (centry, gentry))
    seen_at = outer_index(cnest, seen.index)
    used_at = tuple(i.subs(gparam, cparam) for i in outer_index(gnest, used.index))
    if seen_at != used_at or cparam not in seen_at:
        return False
    for name, nest in ((seen.data, cnest), (used.data, gnest)):
        outer = [e for e in state.in_edges(nest) if e.dst_conn == name]
        if nest.sdfg.arrays[name].shape != state.sdfg.arrays[state.memlet_path(outer[0])[0].src.data].shape:
            return False
    region, edge = assignment_edge(cnest.sdfg, seen.symbol)
    if region is not cnest.sdfg:
        return False
    order = list(graphlib.topological_sort(region.nx))
    for wstate, node in writes(cnest.sdfg, seen.data):
        if wstate.parent_graph is not region or order.index(wstate) > order.index(edge.src):
            return False
        for wedge in wstate.in_edges(node):
            index = tuple(outer_index(cnest, tuple(r[0] for r in wedge.data.subset.ndrange())))
            if wedge.data.subset.num_elements() != 1 or index != seen_at:
                return False
    gedges = [e for e in state.in_edges(gentry) if e.dst_conn and e.data.data == used.data]
    for gedge in gedges:
        source = state.memlet_path(gedge)[0].src
        if not isinstance(source, nodes.AccessNode):
            return False
        if any(e.src is not state.exit_node(centry) for e in state.in_edges(source)):
            return False
    return True


def implied_guards(sdfg: SDFG) -> list[ImpliedGuard]:
    """Every count conjunct some flag conjunct of the same guard implies, over ``sdfg`` and its nested SDFGs."""
    found: list[ImpliedGuard] = []
    for state in sdfg.states():
        for gentry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            gnest = map_body(state, gentry)
            if gnest is None:
                continue
            for block in gnest.sdfg.all_control_flow_blocks(recursive=False):
                if not isinstance(block, ConditionalBlock):
                    continue
                for branch, (condition, body) in enumerate(block.branches):
                    if condition is None:
                        continue
                    for conjunct in flatten_and(symbolic.pystr_to_symbolic(condition.as_string)):
                        guard = check(state, gentry, gnest, block, branch, conjunct)
                        if guard is not None:
                            found.append(guard)
    return found


def check(state, gentry, gnest, block, branch, conjunct) -> ImpliedGuard | None:
    names = [str(s) for s in conjunct.free_symbols]
    if len(names) != 1 or names[0] not in gnest.in_connectors or not nonzero_when_positive(conjunct, names[0]):
        return None
    count_edges = [e for e in state.in_edges(gnest) if e.dst_conn == names[0]]
    if len(count_edges) != 1:
        return None
    count_node = state.memlet_path(count_edges[0])[0].src
    if not isinstance(count_node, nodes.AccessNode) or not state.sdfg.arrays[count_node.data].transient:
        return None
    if state.sdfg.arrays[count_node.data].total_size != 1:
        return None
    counted = counting_map(state, count_node)
    if counted is None:
        return None
    centry, cnest, output = counted
    if centry.map.range != gentry.map.range or state.entry_node(centry) is not state.entry_node(gentry):
        return None
    if not count_initialized_nonnegative(state, count_node, centry):
        return None
    for symbol in implying_symbols(block, branch, conjunct):
        used = predicate(gnest.sdfg, symbol)
        if used is None:
            continue
        for seen_symbol in assigned_symbols(cnest.sdfg):
            seen = predicate(cnest.sdfg, seen_symbol)
            if seen is None or seen.data != used.data:
                continue
            if not flag_reaches_unchanged(state, (centry, cnest), (gentry, gnest), seen, used):
                continue
            if at_least(contribution(cnest.sdfg, output, seen.symbol, True), 1) and at_least(
                contribution(cnest.sdfg, output, seen.symbol, False), 0
            ):
                return ImpliedGuard(state, gentry, block, branch, conjunct)
    return None


def drop_implied_guard(guard: ImpliedGuard) -> None:
    """Remove the implied conjunct; a guard left empty becomes ``True`` (:class:`LiftTrivialIf` removes it)."""
    condition, body = guard.block.branches[guard.branch]
    conjuncts = flatten_and(symbolic.pystr_to_symbolic(condition.as_string))
    rest = [c for c in conjuncts if c != guard.conjunct]
    text = symbolic.symstr(sympy.And(*rest)) if rest else "True"
    guard.block.branches[guard.branch] = (CodeBlock(text), body)
    nest = map_body(guard.state, guard.guarded)
    name = str(next(iter(guard.conjunct.free_symbols)))
    if not reads(nest.sdfg, name):
        for edge in [e for e in guard.state.in_edges(nest) if e.dst_conn == name]:
            guard.state.remove_memlet_path(edge, remove_orphans=True)
        nest.remove_in_connector(name)
        nest.sdfg.remove_data(name, validate=False)


def reads(sdfg: SDFG, name: str) -> bool:
    """Whether ``sdfg`` still reads ``name``: an access node, a condition or an interstate edge."""
    if any(n.data == name for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode)):
        return True
    texts = [e.data.condition.as_string for r in sdfg.all_control_flow_regions(recursive=True) for e in r.edges()]
    texts += [
        v for r in sdfg.all_control_flow_regions(recursive=True) for e in r.edges() for v in e.data.assignments.values()
    ]
    for block in sdfg.all_control_flow_blocks(recursive=True):
        if isinstance(block, ConditionalBlock):
            texts += [c.as_string for c, body in block.branches if c is not None]
    return any(re.search(rf"\b{re.escape(name)}\b", t) for t in texts)


class DropImpliedGuards(ppl.Pass):
    """Drop every count guard a flag in the same guard implies (:func:`implied_guards`)."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.CFG | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        dropped = 0
        for nested in sdfg.all_sdfgs_recursive():
            for guard in implied_guards(nested):
                drop_implied_guard(guard)
                dropped += 1
        return dropped or None
