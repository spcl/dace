# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PerfectLoopNesting`` -- distribute a loop over the strongly-connected components of its body's dependence graph, so
each statement group ends up in its own complete nest and parallelizes on its own axes."""

import weakref
from collections.abc import Callable
from typing import Any

from dace import SDFG, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
    is_read,
    is_written,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.distribute_producer_consumer import (
    _forward_flow_groups,
    _interstate_reads,
    _rw_subsets,
)
from dace.transformation.passes.canonicalize.sift_statements_into_perfect_nest import sift_imperfect_nests
from dace.transformation.passes.loop_fission import LoopFission, _linear_blocks
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

#: Safety bound on fixpoint rounds -- a perfect nest is at most this deep in
#: practice; the loop breaks as soon as a round changes nothing.
MAX_ROUNDS = 8

#: A loop's iterator and its init, condition and update statements as text, and whether a jump targets it.
LoopHeader = tuple[str | None, str | None, str | None, str | None, bool]


def loop_header(loop: LoopRegion) -> LoopHeader:
    """Everything ``TrivialLoopElimination.can_be_applied`` reads off ``loop``."""
    init, condition, update = loop.init_statement, loop.loop_condition, loop.update_statement
    return (
        loop.loop_variable,
        None if init is None else init.as_string,
        None if condition is None else condition.as_string,
        None if update is None else update.as_string,
        # A single trip is spliced only without a ``break`` or ``continue``: two loops of one header differ here
        bool(loop_analysis.loop_jumps(loop)),
    )


def trivial_loop_accepts(xform: TrivialLoopElimination, region: ControlFlowRegion, loop: LoopRegion) -> bool:
    """One ``can_be_applied`` probe, refusing on an exception exactly as the pattern matcher does."""
    try:
        xform.setup_match(region.sdfg, -1, -1, {TrivialLoopElimination.loop: loop}, 0, override=True)
        return bool(xform.can_be_applied(region, 0, region.sdfg, permissive=False))
    except Exception as err:
        if Config.get_bool("optimizer", "match_exception"):
            raise
        print(f"WARNING: TrivialLoopElimination::can_be_applied triggered a {type(err).__name__} exception: {err}")
        return False


def first_trivial_loop(
    sdfg: SDFG, xform: TrivialLoopElimination, verdicts: dict[LoopHeader, bool]
) -> tuple[ControlFlowRegion, LoopRegion] | None:
    """The first loop the pattern matcher would apply ``TrivialLoopElimination`` to, in its order."""
    for region in sdfg.all_control_flow_regions(recursive=True):
        for block in region.nodes():
            if not isinstance(block, LoopRegion):
                continue
            header = loop_header(block)
            verdict = verdicts.get(header)
            if verdict is None:
                verdict = verdicts[header] = trivial_loop_accepts(xform, region, block)
            if verdict:
                return region, block
    return None


def eliminate_trivial_loops(sdfg: SDFG, verdicts: dict[LoopHeader, bool]) -> int:
    """``PatternMatchAndApplyRepeated([TrivialLoopElimination()])``, without re-probing every loop per apply."""
    xform = TrivialLoopElimination()
    applied = 0
    target = first_trivial_loop(sdfg, xform, verdicts)
    while target is not None:
        region, loop = target
        xform.setup_match(region.sdfg, region.cfg_id, -1, {TrivialLoopElimination.loop: loop}, 0, override=True)
        xform.apply(region, region.sdfg)
        applied += 1
        target = first_trivial_loop(sdfg, xform, verdicts)
    return applied


def level_parallel(blocks: list[ControlFlowBlock], loop_var: str | None, arrays: dict[str, Any]) -> bool:
    """Whether ``loop_var`` carries no dependence for ``blocks`` on their own.

    :param blocks: The body blocks to judge together.
    :param loop_var: The enclosing loop's iteration variable.
    :param arrays: The owning SDFG's data descriptors, to tell a transient from a real container.
    """
    writes: dict[str, frozenset[int]] = {}
    reads: dict[str, frozenset[int]] = {}
    for block in blocks:
        block_writes, block_reads = _rw_subsets(block, loop_var)
        for name, axes in block_writes.items():
            writes[name] = writes[name] & axes if name in writes else axes
        for name, axes in block_reads.items():
            reads[name] = reads[name] & axes if name in reads else axes
    return all(
        axes & reads.get(name, axes) for name, axes in writes.items() if name in arrays and not arrays[name].transient
    )


def parallel_level_diagnostic(loop: LoopRegion, groups: list[list[ControlFlowBlock]]) -> tuple[int, int]:
    """``(fused, distributed)`` parallel levels at ``loop``'s own level, for the split ``groups``.

    :param loop: The loop being distributed.
    :param groups: The legal group partition of its body blocks.
    """
    loop_var = loop.loop_variable
    arrays = loop.sdfg.arrays
    fused = int(level_parallel([block for group in groups for block in group], loop_var, arrays))
    return fused, sum(int(level_parallel(group, loop_var, arrays)) for group in groups)


def separates_ordered_side_effects(groups: list[list[ControlFlowBlock]]) -> bool:
    """Whether ``groups`` puts ordered side effects (:func:`~loop_analysis.ordered_side_effects`) into two loops: one
    loop would run all of its side effects before the other's, where the iterations interleaved them."""
    holding = [group for group in groups if any(loop_analysis.ordered_side_effects(block) for block in group)]
    return len(holding) > 1


def distribute_loops(
    sdfg: SDFG,
    diagnostics: list[tuple[str, int, int]] | None = None,
    grouping: Callable[[LoopRegion], list[list[ControlFlowBlock]] | None] = _forward_flow_groups,
) -> int:
    """Distribute every loop of ``sdfg`` whose body splits into more than one dependence component.

    :param sdfg: The SDFG to transform in place.
    :param diagnostics: When given, one ``(label, fused, distributed)`` parallel-level record is
                        appended per split (:func:`parallel_level_diagnostic`).
    :param grouping: The statement partition of one loop body, ``None`` when it does not split.
    :returns: The number of loops distributed.
    """
    count = 0
    changed = True
    # ``grouping`` reads the loop alone, so a refusal stands until a fission inside the loop changes its body.
    refused: set[LoopRegion] = set()
    while changed:
        changed = False
        for loop in list(sdfg.all_control_flow_regions(recursive=True)):
            if not isinstance(loop, LoopRegion) or loop in refused:
                continue
            groups = grouping(loop)
            if groups is None or separates_ordered_side_effects(groups):
                refused.add(loop)
                continue
            if diagnostics is not None:
                diagnostics.append((loop.label,) + parallel_level_diagnostic(loop, groups))
            refused -= loop_analysis.enclosing_loops(loop)
            # ``add_node`` re-homes each per-group ``copy.deepcopy(loop)`` clone, nested SDFGs included.
            LoopFission._fission_blocks(loop, groups)
            count += 1
            changed = True
            break
    return count


def groups_with_expansion(
    loop: LoopRegion, tried: "weakref.WeakSet[LoopRegion]"
) -> list[list[ControlFlowBlock]] | None:
    """:func:`_forward_flow_groups`, falling back once per loop to :func:`split_after_expansion`."""
    groups = _forward_flow_groups(loop)
    if groups is not None or loop in tried:
        return groups
    tried.add(loop)
    return split_after_expansion(loop)


def accessed_names(block: ControlFlowBlock) -> tuple[set[str], set[str]]:
    """The containers ``block`` writes and reads, control flow included: the names of :func:`_rw_subsets` without
    judging a subset, for the question asked of every loop."""
    writes: set[str] = set()
    reads = set(_interstate_reads(block))
    for state in [block] if isinstance(block, SDFGState) else block.states():
        for node in state.data_nodes():
            written = is_written(state, node)
            if written:
                writes.add(node.data)
            if is_read(state, node) or not written:
                reads.add(node.data)
    return writes, reads


def split_after_expansion(loop: LoopRegion) -> list[list[ControlFlowBlock]] | None:
    """The distribution groups of a sequential ``loop`` once the transients its blocks share get one slot each."""
    from dace.transformation.passes.buffer_expansion import (
        BufferExpansion,
        expand_iteration_private,
        iteration_private_buffers,
    )

    order = _linear_blocks(loop, allow_assignments=True) if loop.loop_variable else None
    if order is None or len(order) < 2:
        return None
    sdfg = loop.sdfg
    seen: dict[str, int] = {}
    accumulated: set[str] = set()
    for block in order:
        writes, reads = accessed_names(block)
        for name in writes | reads:
            seen[name] = seen.get(name, 0) + 1
        accumulated |= writes & reads
    shared = {name for name, count in seen.items() if count > 1}
    # Only an accumulator one block updates and another reads couples the blocks for good (the CSR row sum ``s``);
    # without one, expanding a shared value (a loop-invariant ``alpha[0]`` copy) only splits off a statement and
    # hides the nest from the library lifts (``LoopToSymm``).
    transients = {name for name in shared & accumulated if name in sdfg.arrays and sdfg.arrays[name].transient}
    if not transients or BufferExpansion._loop_mappable(sdfg, loop):
        return None
    found = iteration_private_buffers(sdfg, loop, shared)
    if found is None:
        return None
    buffers, slot = found
    groups = _forward_flow_groups(loop, allow_assignments=True, expanded=dict.fromkeys(buffers, slot))
    # The expansion is for a producer with work (CSR's row sum) to run ahead of its consumer.
    if groups is None or not any(holds_iteration(block) for block in groups[0]):
        return None
    # Only a container two groups share needs a slot per iteration.
    groups_of: dict[str, set[int]] = {}
    for index, group in enumerate(groups):
        for block in group:
            writes, reads = _rw_subsets(block, loop.loop_variable)
            for name in (*writes, *reads):
                groups_of.setdefault(name, set()).add(index)
    expand_iteration_private(sdfg, loop, [name for name in buffers if len(groups_of.get(name, ())) >= 2])
    return groups


def holds_iteration(block: ControlFlowBlock) -> bool:
    """Whether ``block`` is or holds a loop or a map."""
    if isinstance(block, SDFGState):
        return any(isinstance(node, nodes.MapEntry) for node in block.nodes())
    if not isinstance(block, AbstractControlFlowRegion):
        return False
    if any(isinstance(region, LoopRegion) for region in block.all_control_flow_regions()):
        return True
    return any(isinstance(node, nodes.MapEntry) for state in block.states() for node in state.nodes())


@properties.make_properties
@transformation.explicit_cf_compatible
class PerfectLoopNesting(ppl.Pass):
    """Form perfect nests, by DISTRIBUTING the parent loop over the dependence components of its body -- wherever that
    is legal -- and, on GPU, by SINKING the outer statements distribution cannot separate."""

    CATEGORY: str = "Optimization Preparation"

    target = properties.Property(
        dtype=str,
        default="cpu",
        choices=["cpu", "gpu"],
        desc="Target policy: 'gpu' also sinks outer statements into the inner "
        "loop to expose the outer axis; 'cpu' distributes only.",
    )

    def __init__(self, target: str = "cpu") -> None:
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        uniq = UniqueLoopIterators()
        verdicts: dict[LoopHeader, bool] = {}
        rounds = 0
        for round_index in range(MAX_ROUNDS):
            # Each step reports its own kind of result, so test each for truthiness rather than summing.
            changed = False
            if distribute_loops(sdfg):
                changed = True
            if MoveIfIntoLoop().apply_pass(sdfg, {}):
                changed = True
            if eliminate_trivial_loops(sdfg, verdicts):
                changed = True
            # Sink LAST in the round: give the distribution the first chance to separate the
            # statements outright, and only sink what is still stuck in an imperfect nest.
            if self.target == "gpu" and sift_imperfect_nests(sdfg):
                changed = True
            if not changed:
                break
            # SSA-rename cloned iterators (a pure relabelling; not a fixpoint signal, it just keeps the next round's
            # matchers clean).
            uniq.apply_pass(sdfg, {})
            rounds += 1
        return rounds or None


__all__ = ["PerfectLoopNesting", "distribute_loops", "level_parallel", "parallel_level_diagnostic"]
