# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Merge guards that test the same predicate, so the bodies they keep apart meet in one branch."""

import ast
from collections import deque
from typing import Any

from dace import SDFG, InterstateEdge, symbolic
from dace.properties import CodeBlock
from dace.sdfg.state import (
    ConditionalBlock,
    ControlFlowRegion,
    LoopRegion,
    SymbolResolver,
    is_written,
    rehome_claimed_block,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.loop_fission import _linear_blocks
from dace.transformation.passes.loop_invariant_code_motion import subset_in_bounds


def single_guard(block: ConditionalBlock) -> tuple[CodeBlock, ControlFlowRegion] | None:
    """The one ``(condition, body)`` of a block with a single, non-``else`` branch, else ``None``."""
    if len(block.branches) != 1 or block.branches[0][0] is None:
        return None
    return block.branches[0]


def changed_by(block: ConditionalBlock) -> set[str]:
    """The names ``block`` may change: the symbols it assigns, the loop variables its loops leave set, and the
    containers it writes."""
    changed: set[str] = set()
    for edge in block.all_interstate_edges(recursive=True):
        changed |= edge.data.assignments.keys()
    for region in block.all_control_flow_regions(recursive=True):
        if isinstance(region, LoopRegion) and region.loop_variable:
            changed.add(region.loop_variable)
    for state in block.states():
        changed |= {node.data for node in state.data_nodes() if is_written(state, node)}
    return changed


#: ``(guard, other guard) -> 'equal' | 'opposite' | None``: the relation proved between two guard texts, per pass run.
Verdicts = dict[tuple[str, str], str | None]


def guard_relation(guard: str, other: str, verdicts: Verdicts) -> str | None:
    """Whether ``other`` is provably ``'equal'`` or ``'opposite'`` to ``guard``, memoized in ``verdicts``."""
    key = (guard, other)
    if key not in verdicts:
        if symbolic.guards_equal(guard, other):
            verdicts[key] = "equal"
        elif symbolic.guards_opposite(guard, other):
            verdicts[key] = "opposite"
        else:
            verdicts[key] = None
    return verdicts[key]


def merge_consecutive(first: ConditionalBlock, second: ConditionalBlock, verdicts: Verdicts | None = None) -> bool:
    """Merge ``second`` into ``first`` when their guards are equal or opposite. ``False`` if they stay apart."""
    region = first.parent_graph
    if region.out_degree(first) != 1 or region.in_degree(second) != 1:
        return False
    link = region.out_edges(first)[0]
    # Merged, an assignment on the link would run only on the guarded path.
    if link.dst is not second or not link.data.is_unconditional() or link.data.assignments:
        return False
    guard, body = single_guard(first) or (None, None)
    other_guard, other_body = single_guard(second) or (None, None)
    if guard is None or other_guard is None or len(body.sink_nodes()) != 1:
        return False
    if other_guard.get_free_symbols() & changed_by(first):
        return False

    relation = guard_relation(guard.as_string, other_guard.as_string, {} if verdicts is None else verdicts)
    if relation == "equal":
        sink = body.sink_nodes()[0]
        start = other_body.start_block
        for block in other_body.nodes():
            body.add_node(block, ensure_unique_name=True)
        body.add_edge(sink, start, InterstateEdge())
        for edge in other_body.edges():
            body.add_edge(edge.src, edge.dst, edge.data)
    elif relation == "opposite":
        second.remove_branch(other_body)
        first.add_branch(None, other_body)
    else:
        return False

    for edge in region.out_edges(second):
        region.add_edge(first, edge.dst, edge.data)
    region.remove_node(second)
    rehome_claimed_block(first, region.sdfg)
    return True


#: Expression nodes a hoisted load may hold: nothing a skipped iteration could trip on (a division, a call).
SAFE_LOAD_NODES = (
    ast.Expression,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Tuple,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.USub,
)


def hoist_loads(outer: ConditionalBlock, inner: ConditionalBlock) -> bool:
    """Move the symbol loads between ``outer``'s test and ``inner`` above ``outer``, leaving ``inner`` alone in the
    branch (dbcsr's ``if a_bid >= 0: {b_bid = b_index[ib, 2]; if b_bid >= 0: ...}``)."""
    branch = inner.parent_graph
    order = _linear_blocks(branch, allow_assignments=True)
    if order is None or order[-1] is not inner or any(b.number_of_nodes() for b in order[:-1]):
        return False
    sdfg = outer.sdfg
    loads = [branch.edges_between(a, b)[0].data for a, b in zip(order, order[1:])]
    names = {lhs for load in loads for lhs in load.assignments}
    if names & {str(s) for s in single_guard(outer)[0].get_free_symbols()}:
        return False
    if any(
        not isinstance(n, SAFE_LOAD_NODES)
        for load in loads
        for rhs in load.assignments.values()
        for n in ast.walk(ast.parse(rhs, mode="eval"))
    ):
        return False
    reads = [m for load in loads for m in load.get_read_memlets(sdfg.arrays)]
    if {m.data for m in reads} & changed_by(outer):
        return False
    inside = set(outer.all_control_flow_blocks(recursive=True))
    if any(
        edge.src not in inside and names & (edge.data.assignments.keys() | edge.data.free_symbols)
        for edge in sdfg.all_interstate_edges(recursive=True)
    ) or any(names & state.free_symbols for state in sdfg.states() if state not in inside):
        return False
    facts = SymbolResolver().facts_at(outer.parent_graph)
    if not all(subset_in_bounds(m.subset, sdfg.arrays[m.data], facts) for m in reads):
        return False
    region = outer.parent_graph
    for load in loads:
        region.add_state_before(
            outer, f"{outer.label}_load", is_start_block=region.start_block is outer, assignments=load.assignments
        )
    for block in order[:-1]:
        branch.remove_node(block)
    return True


def flatten_nested(inner: ConditionalBlock) -> bool:
    """Fold ``inner`` into the guard around it when it is alone in that guard's only branch, or only loads that
    :func:`hoist_loads` moves above that guard precede it."""
    branch = inner.parent_graph
    outer = branch.parent_graph if branch is not None else None
    if not isinstance(outer, ConditionalBlock):
        return False
    outer_guard = single_guard(outer)
    inner_guard = single_guard(inner)
    if outer_guard is None or inner_guard is None:
        return False
    if branch.number_of_nodes() != 1 and not hoist_loads(outer, inner):
        return False
    body = inner_guard[1]
    inner.remove_branch(body)
    outer.remove_branch(branch)
    outer.add_branch(CodeBlock(f"({outer_guard[0].as_string}) and ({inner_guard[0].as_string})"), body)
    rehome_claimed_block(outer, outer.sdfg)
    return True


def flatten_else(inner: ConditionalBlock) -> bool:
    """Lift ``inner``'s branches into the guard around it when ``inner`` is alone in that guard's ``else``."""
    branch = inner.parent_graph
    outer = branch.parent_graph if branch is not None else None
    if not isinstance(outer, ConditionalBlock) or branch.number_of_nodes() != 1:
        return False
    if outer.branches[-1][1] is not branch or outer.branches[-1][0] is not None:
        return False
    outer.remove_branch(branch)
    for condition, body in list(inner.branches):
        inner.remove_branch(body)
        outer.add_branch(condition, body)
    rehome_claimed_block(outer, outer.sdfg)
    return True


def attached(region: ControlFlowRegion) -> bool:
    """Whether ``region`` is still part of its SDFG: a merge empties and detaches the branches it dissolves."""
    while not isinstance(region, SDFG):
        parent = region.parent_graph
        if parent is None:
            return False
        if isinstance(parent, ConditionalBlock):
            if all(branch is not region for _, branch in parent.branches):
                return False
        elif region not in parent.nodes():
            return False
        region = parent
    return True


def merge_region(region: ControlFlowRegion, pending: deque[ControlFlowRegion], verdicts: Verdicts) -> int:
    """Apply every merge rooted at a guard of ``region``, consecutive guards before nested ones, and queue onto
    ``pending`` the regions a merge changed elsewhere: the body that received a second body's blocks, the region of
    the guard a nested one folded into."""
    merged = 0
    progress = True
    # The guard a nested one folds into, read before the fold detaches ``region`` from it.
    outer = region.parent_graph
    while progress and attached(region):
        progress = False
        for block in region.nodes():
            if not isinstance(block, ConditionalBlock):
                continue
            successors = region.successors(block)
            if (
                len(successors) == 1
                and isinstance(successors[0], ConditionalBlock)
                and merge_consecutive(block, successors[0], verdicts)
            ):
                pending.extend(branch for _, branch in block.branches)
                progress = True
            elif flatten_nested(block) or flatten_else(block):
                pending.append(outer.parent_graph)
                pending.extend(branch for _, branch in outer.branches)
                return merged + 1
            else:
                continue
            merged += 1
            break
    return merged


@transformation.explicit_cf_compatible
class MergeGuards(ppl.Pass):
    """Merge consecutive guards with equal or opposite predicates and fold a guard nested alone in another's only
    branch or ``else``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.CFG | ppl.Modifies.States))

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Merge every qualifying guard of ``sdfg`` and its nested SDFGs, to a fixpoint."""
        merged = 0
        # Every region once, nested SDFGs included
        pending = deque(sdfg.all_control_flow_regions(recursive=True))
        verdicts: Verdicts = {}
        while pending:
            merged += merge_region(pending.popleft(), pending, verdicts)
        return merged or None


__all__ = ["MergeGuards"]
