# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SiftAssignmentsUp``: moves interstate assignments up through the blocks before them, and out of loops and
conditional branches, so that the blocks between them can fuse.

An assignment ``x = rhs`` on the only edge out of a block moves to the only edge into it when that edge is
unconditional, the block does not read ``x``, and ``rhs`` reads nothing the block writes or assigns. On the edge out of
the first block of a loop or a conditional branch, it moves to the edge before the loop or the conditional when ``rhs``
reads nothing the loop or conditional changes (loop variables included), ``x`` has no other definition, and ``rhs`` is
safe to evaluate speculatively: it indexes arrays only at fixed elements (no symbols but the arrays' own descriptor
symbols). Then ``x`` holds the same value in every iteration, and where the loop runs no iteration or the branch is not
taken, it holds a value nothing else could have given it. In single-assignment form (``SymbolSSA``), this gathers the
bounds a sequence of loops or maps reads at the head of the sequence.
"""

from collections import Counter

from dace import SDFG, SDFGState, properties, symbolic
from dace.sdfg import InterstateEdge
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Names a block reads (symbols and data) and changes (data it writes, symbols it assigns)
Effects = tuple[set[str], set[str]]


def effects(block: ControlFlowBlock) -> Effects:
    if isinstance(block, SDFGState):
        reads, writes = block.read_and_write_sets()
        return block.free_symbols | reads, set(writes)
    assert isinstance(block, AbstractControlFlowRegion)
    reads = set(block.used_symbols(all_symbols=True, with_contents=True))
    changes = {b.loop_variable for b in [block, *block.all_control_flow_blocks()] if isinstance(b, LoopRegion)}
    for inner in block.all_control_flow_blocks():
        if isinstance(inner, SDFGState):
            inner_reads, inner_writes = inner.read_and_write_sets()
            reads |= inner_reads
            changes |= inner_writes
    for edge in block.all_interstate_edges():
        reads |= edge.data.read_symbols()
        changes |= edge.data.assignments.keys()
    return reads, changes


def rhs_names(rhs: str) -> set[str]:
    return {str(s) for s in symbolic.free_symbols_and_functions(rhs)} | set(symbolic.arrays(rhs))


def speculation_safe(rhs: str, sdfg: SDFG) -> bool:
    """Whether evaluating ``rhs`` where it would not run reads no out-of-bounds element: its array indices name no
    symbols but those of the arrays' descriptors."""
    arrays = set(symbolic.arrays(rhs)) & sdfg.arrays.keys()
    if not arrays:
        return True
    descriptor_symbols = {str(s) for name in arrays for s in sdfg.arrays[name].used_symbols(True)}
    index_symbols = {str(s) for s in symbolic.free_symbols_and_functions(rhs)} - arrays
    return index_symbols <= descriptor_symbols


def movable_past(
    before: InterstateEdge, after: InterstateEdge, passed: Effects, definitions: Counter | None, sdfg: SDFG | None
) -> list[str]:
    """The assignments of ``after`` that can move to ``before`` past code with the ``passed`` effects; with
    ``definitions`` and ``sdfg`` (moving out of a loop or branch), only symbols defined once whose value is safe to
    evaluate speculatively."""
    if not before.is_unconditional():
        return []
    reads, changes = passed
    # The code passed and the edge before must not see the old value; the value must not depend on what they change
    sees_name = reads | before.read_symbols() | before.assignments.keys()
    changes = changes | before.assignments.keys()
    return [
        name
        for name, rhs in after.assignments.items()
        if name not in sees_name and not rhs_names(rhs) & changes and (definitions is None or definitions[name] == 1)
    ]


def move(before: InterstateEdge, after: InterstateEdge, names: list[str]) -> int:
    for name in names:
        before.assignments[name] = after.assignments.pop(name)
    return len(names)


def sift_through(region: ControlFlowRegion, block: ControlFlowBlock) -> int:
    """Moves assignments from the edge after ``block`` to the edge before it."""
    if region.in_degree(block) != 1 or region.out_degree(block) != 1:
        return 0
    before, after = region.in_edges(block)[0].data, region.out_edges(block)[0].data
    return move(before, after, movable_past(before, after, effects(block), None, None))


def sift_out_of(region: ControlFlowRegion, definitions: Counter) -> int:
    """Moves the invariant assignments on the edge after the first block of ``region``, a loop or a conditional
    branch, to the edge before the loop or the conditional."""
    outer = region.parent_graph if isinstance(region.parent_graph, ConditionalBlock) else region
    parent, first = outer.parent_graph, region.start_block
    if parent.in_degree(outer) != 1 or region.in_degree(first) != 0 or region.out_degree(first) != 1:
        return 0
    before, after = parent.in_edges(outer)[0].data, region.out_edges(first)[0].data
    outer_changes = effects(outer)[1]
    first_reads = effects(first)[0] | set(outer.used_symbols(all_symbols=True, with_contents=False))
    passed = (first_reads, outer_changes)
    return move(before, after, movable_past(before, after, passed, definitions, region.sdfg))


def leaves(region: ControlFlowRegion) -> bool:
    """Whether assignments may sift out of ``region``: a loop body or a conditional branch."""
    return isinstance(region, LoopRegion) or isinstance(region.parent_graph, ConditionalBlock)


def definition_counts(sdfg: SDFG) -> Counter:
    counts = Counter(name for edge in sdfg.all_interstate_edges() for name in edge.data.assignments)
    counts.update(b.loop_variable for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion))
    return counts


def sift_sdfg(sdfg: SDFG) -> int:
    """Sifts the assignments of ``sdfg`` (nested SDFGs apart) up until none moves; returns how many moves happened."""
    moved = 0
    changed = True
    while changed:
        definitions = definition_counts(sdfg)
        step = 0
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, ConditionalBlock):
                continue
            step += sum(sift_through(region, block) for block in list(region.nodes()))
            if leaves(region):
                step += sift_out_of(region, definitions)
        moved += step
        changed = step > 0
    return moved


@properties.make_properties
@transformation.explicit_cf_compatible
class SiftAssignmentsUp(ppl.Pass):
    """Moves interstate assignments up through the blocks that neither read their symbol nor change what they read,
    and out of the loops and conditional branches they do not depend on. Returns the number of assignment moves."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.States))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> int | None:
        moved = sum(sift_sdfg(nested) for nested in sdfg.all_sdfgs_recursive())
        return moved or None
