# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Merge guards that test the same predicate, so the bodies they keep apart meet in one branch.

Three shapes merge, and nothing else does:

* consecutive guards with the same predicate (``if c: A`` then ``if c: B`` becomes ``if c: A; B``) or with
  opposite predicates (``if c: A`` then ``if not c: B`` becomes ``if c: A else: B``), when nothing between the two
  tests changes what the second reads;
* a guard alone in the only branch of another (``if a: if b: X`` becomes ``if (a) and (b): X``);
* a guard alone in the ``else`` of another (``if a: X else: if b: Y else: Z`` becomes ``if a: X elif b: Y else: Z``).

Every body moves; none is copied. Guards that test different predicates stay apart: merging them takes one copy
of a body per combination of branches, and the copies keep their transients' names, so a scratch transient of a
map body turns into one shared by every copy and code generation hoists it out of the parallel loop.
"""

from typing import Any

from dace import SDFG, InterstateEdge, symbolic
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, rehome_claimed_block
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def single_guard(block: ConditionalBlock) -> tuple[CodeBlock, ControlFlowRegion] | None:
    """The one ``(condition, body)`` of a block with a single, non-``else`` branch, else ``None``."""
    if len(block.branches) != 1 or block.branches[0][0] is None:
        return None
    return block.branches[0]


def changed_by(block: ConditionalBlock) -> set[str]:
    """The names ``block`` may change: the symbols it assigns and the containers it writes."""
    changed: set[str] = set()
    for edge in block.all_interstate_edges(recursive=True):
        changed |= edge.data.assignments.keys()
    for state in block.states():
        changed |= {node.data for node in state.data_nodes() if state.in_degree(node) > 0}
    return changed


def merge_consecutive(first: ConditionalBlock, second: ConditionalBlock) -> bool:
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

    if symbolic.guards_equal(guard.as_string, other_guard.as_string):
        sink = body.sink_nodes()[0]
        start = other_body.start_block
        for block in other_body.nodes():
            body.add_node(block, ensure_unique_name=True)
        body.add_edge(sink, start, InterstateEdge())
        for edge in other_body.edges():
            body.add_edge(edge.src, edge.dst, edge.data)
    elif symbolic.guards_opposite(guard.as_string, other_guard.as_string):
        second.remove_branch(other_body)
        first.add_branch(None, other_body)
    else:
        return False

    for edge in region.out_edges(second):
        region.add_edge(first, edge.dst, edge.data)
    region.remove_node(second)
    rehome_claimed_block(first, region.sdfg)
    return True


def flatten_nested(inner: ConditionalBlock) -> bool:
    """Fold ``inner`` into the guard around it when it is alone in that guard's only branch."""
    branch = inner.parent_graph
    outer = branch.parent_graph if branch is not None else None
    if not isinstance(outer, ConditionalBlock) or branch.number_of_nodes() != 1:
        return False
    outer_guard = single_guard(outer)
    inner_guard = single_guard(inner)
    if outer_guard is None or inner_guard is None:
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


def merge_one(sdfg: SDFG) -> bool:
    """Apply the first merge found in ``sdfg``'s own control flow, consecutive guards before nested ones."""
    for region in sdfg.all_control_flow_regions(recursive=True):
        for block in region.nodes():
            if not isinstance(block, ConditionalBlock):
                continue
            successors = region.successors(block)
            if (
                len(successors) == 1
                and isinstance(successors[0], ConditionalBlock)
                and merge_consecutive(block, successors[0])
            ):
                return True
            if flatten_nested(block) or flatten_else(block):
                return True
    return False


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
        """Merge every qualifying guard of ``sdfg`` and its nested SDFGs, to a fixpoint.

        :returns: How many merges were applied, or ``None`` if none.
        """
        merged = 0
        for sd in sdfg.all_sdfgs_recursive():
            while merge_one(sd):
                merged += 1
        return merged or None


__all__ = ["MergeGuards"]
