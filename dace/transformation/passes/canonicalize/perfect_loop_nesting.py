# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PerfectLoopNesting`` -- distribute a loop over the strongly-connected
components of its body's dependence graph, so each statement group ends up in
its own complete nest and parallelizes on its own axes.

WHEN A DISTRIBUTION IS LEGAL
----------------------------

Allen-Kennedy: keep every strongly-connected component of the loop-body
dependence graph together, emit one loop per component, in the graph's
TOPOLOGICAL order. That is always legal, and the component partition is the
unique finest legal one. The only freedom left is the order among mutually
incomparable components, and it is pinned to ORIGINAL PROGRAM ORDER -- an
unpinned choice would make the emitted state order depend on nothing the input
determines.

The grouping is :func:`~dace.transformation.passes.canonicalize
.distribute_producer_consumer._forward_flow_groups`, which is that partition at
BLOCK granularity: a group is a top-level child of the loop body, which may be
an ``SDFGState`` or a whole ``LoopRegion``. Because the emitted order is the
original one, every forward dependence is satisfied by construction and the
only edge that can close a cycle is a BACKWARD loop-carried one -- a later
block writing what an earlier block reads or writes, or an earlier block
producing at an index a later block does not read per-iteration. That is what
merges two blocks into one group.

NEST GRANULARITY IS NOT STATEMENT GRANULARITY
---------------------------------------------

At statement granularity both statements run at the same parent index, so a
dependence is a comparison of two POINT subscripts and ``SplitStatements``'
signed-offset direction table settles it. Here a child is a whole nest: at
parent iteration ``i`` it touches its ENTIRE inner footprint, so the question is
about REGIONS parameterised by ``i``, not points. Cloning the parent moves ALL
of child A before ALL of child B, which is legal exactly when nothing depends
backwards -- B(i') must not feed or clobber A(i) for any ``i' > i``.

The condition is discharged per ACCESS rather than by unioning footprints:
``_is_per_iter_subset`` asks whether an access names one element whose
``loop_var`` coordinate IS ``loop_var``, with the remaining coordinates free.
Every access that answers yes lies in the hyperplane of its own parent
iteration, so the children's per-iteration footprints are pairwise disjoint
across ``i`` and only A(i) can reach what B(i) reads -- which is the same before
and after the split. Any access it cannot pin that way -- a range, a fixed slot,
a read one parent iteration behind, a subset the memlets do not describe --
answers no and the two blocks MERGE. Nothing is ever assumed disjoint; the
over-approximation refuses. That is why a sibling pair sharing only a READ-ONLY
array (TSVC ``s2233``'s ``cc``) is a weak test of this rule and an adversarial
pair, where one child reads at a parent-carried offset what the other writes,
is the one that decides it.

LEGAL IS THE WHOLE TEST
-----------------------

The canonical form is the FINEST LEGAL partition, taken unconditionally. There
is no profitability question here: canonicalization owns no cost model, and
"split only when it frees a parallel level" would be a cost decision dressed up
as a canonical form -- it makes the result depend on a judgement rather than on
the input, and costs idempotence. Composition is what undoes an unhelpful split:
the downstream ``loop_fuse`` and ``fuse`` stages fuse back every pair that buys
nothing.

The parallel-level comparison is still computed, as a DIAGNOSTIC rather than a
gate (:func:`parallel_level_diagnostic`): for a given split it reports how many
parallel levels the fused parent had against how many the distributed children
have. Fused, the parent level is parallel only where it is parallel for EVERY
child -- the intersection; distributed, each child gets its own set. On TSVC
``s2233`` / ``s233`` the intersection is empty and each child recovers its own
level, so the split frees one; on ``s2275`` / ``s235`` both children were
already parallel over the parent, so it frees none and fusion is free to put
them back. Neither answer decides anything.

The composition runs to a fixpoint (each round exposes the next nesting level;
``MapFusion`` later re-fuses whatever should recombine):

* the distribution above, rewired by ``LoopFission._fission_blocks`` (CFG
  surgery only -- this pass no longer runs ``LoopFission``'s own grouping,
  whose union-find on shared written container names carried no dependence
  direction or distance and is what disabled this pass);
* ``MoveIfIntoLoop`` -- push a guarding conditional down into each distributed
  loop so a per-statement guard travels with its statement;
* ``TrivialLoopElimination`` -- drop a now-single-iteration loop wrapper;
* ``UniqueLoopIterators`` -- SSA-rename the cloned iterators.

Statement-granularity fission inside ONE body state is not this pass's job:
``SplitStatements`` owns it and runs earlier, in the 'prep' stage.

A no-op when the body is a single block, when a body edge carries an assignment
(a cross-loop induction variable such as TSVC ``s126``'s ``k``, which cloning
the loop would increment once per clone), or when the groups do not separate.
"""
from typing import Any

from dace import SDFG, properties
from dace.sdfg.state import LoopRegion
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize.distribute_producer_consumer import _forward_flow_groups, _rw_subsets
from dace.transformation.passes.canonicalize.sift_statements_into_perfect_nest import sift_imperfect_nests
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

#: Safety bound on fixpoint rounds -- a perfect nest is at most this deep in
#: practice; the loop breaks as soon as a round changes nothing.
_MAX_ROUNDS = 8


def level_parallel(blocks: list, loop_var: str | None, arrays: dict[str, Any]) -> bool:
    """Whether ``loop_var`` carries no dependence for ``blocks`` on their own.

    A container the blocks WRITE is dependence-free across iterations of ``loop_var`` while every
    access to it -- the writes and the reads alike -- names one element that moves WITH the
    iteration (``_is_per_iter_subset``). A write at a fixed slot, a read one iteration behind or
    ahead, or a subset that cannot be read off the memlets at all (an interstate-edge read) all
    answer no. A container the blocks only READ carries nothing and is not consulted.

    TRANSIENTS are not consulted either. The frontend spells every scalar of a statement as one --
    ``aa_index``, ``aa_slice_plus_cc_slice`` -- written at the fixed slot ``[0]`` each iteration,
    which is the shape of a carry and would report every body ever written as carried. What such a
    temporary really does is decided by whether an iteration reads it before writing it, which is
    ``carried_local_transients``' question, and a loop whose sole carry is a transient accumulator
    belongs to ``SplitStatements`` / ``LoopToReduce`` at statement granularity anyway.

    NOTHING is decided on this answer -- ``_forward_flow_groups`` owns legality and the split is
    unconditional. It feeds :func:`parallel_level_diagnostic`, which only reports.

    :param blocks: The body blocks to judge together.
    :param loop_var: The enclosing loop's iteration variable.
    :param arrays: The owning SDFG's data descriptors, to tell a transient from a real container.
    """
    writes: dict[str, bool] = {}
    reads: dict[str, bool] = {}
    for block in blocks:
        block_writes, block_reads = _rw_subsets(block, loop_var)
        for name, aligned in block_writes.items():
            writes[name] = writes.get(name, True) and aligned
        for name, aligned in block_reads.items():
            reads[name] = reads.get(name, True) and aligned
    return all(aligned and reads.get(name, True) for name, aligned in writes.items()
               if name in arrays and not arrays[name].transient)


def parallel_level_diagnostic(loop: LoopRegion, groups: list[list]) -> tuple[int, int]:
    """``(fused, distributed)`` parallel levels at ``loop``'s own level, for the split ``groups``.

    A DIAGNOSTIC, never a gate. Fused, the parent level is parallel only where it is parallel for
    EVERY group -- the intersection -- so ``fused`` is 0 or 1. Distributed, each group gets its own
    set, so ``distributed`` counts the groups that are parallel on their own. ``distributed >
    fused * len(groups)`` says the split freed a level somewhere; the pass splits either way and
    leaves the recombining to the downstream fusion stages.

    :param loop: The loop being distributed.
    :param groups: The legal group partition of its body blocks.
    """
    loop_var = loop.loop_variable
    arrays = loop.sdfg.arrays
    fused = int(level_parallel([block for group in groups for block in group], loop_var, arrays))
    return fused, sum(int(level_parallel(group, loop_var, arrays)) for group in groups)


def distribute_loops(sdfg: SDFG, diagnostics: list | None = None) -> int:
    """Distribute every loop of ``sdfg`` whose body splits into more than one dependence component.

    :param sdfg: The SDFG to transform in place.
    :param diagnostics: When given, one ``(label, fused, distributed)`` parallel-level record is
                        appended per split (:func:`parallel_level_diagnostic`).
    :returns: The number of loops distributed.
    """
    count = 0
    changed = True
    while changed:
        changed = False
        for loop in list(sdfg.all_control_flow_regions(recursive=True)):
            if not isinstance(loop, LoopRegion):
                continue
            groups = _forward_flow_groups(loop)
            if groups is None:
                continue
            if diagnostics is not None:
                diagnostics.append((loop.label, ) + parallel_level_diagnostic(loop, groups))
            LoopFission._fission_blocks(loop, groups)
            # The per-group ``copy.deepcopy(loop)`` clones leave any nested SDFG inside the body
            # with a stale ``parent_sdfg``; reattach before the next scan re-reads the CFG.
            set_nested_sdfg_parent_references(sdfg)
            count += 1
            changed = True
            break
    return count


@properties.make_properties
@transformation.explicit_cf_compatible
class PerfectLoopNesting(ppl.Pass):
    """Form perfect nests, by DISTRIBUTING the parent loop over the dependence components of its
    body -- wherever that is legal -- and, on GPU, by SINKING the outer statements
    distribution cannot separate.

    The two directions are complementary, which is why one pass owns both. Distribution splits
    ``for i: {S1; S2}`` into one parent loop per dependence component, so a component the parent
    level is parallel for gets that level back instead of inheriting the other's recurrence. It is
    a no-op on ``for i: {pre; for j: body; post}`` whenever ``pre`` feeds ``body`` at an index
    ``body`` does not read per-iteration -- the usual case, since that is why they share a nest --
    and even where it does fire it yields separate loops rather than the single 2-D nest a GPU grid
    needs. So ``target='gpu'`` additionally sinks the surviving pre/post blocks into the inner loop
    under boundary guards (``sift_imperfect_nests``; see that module for the S1 / S7 soundness
    gates). On CPU the sink is skipped -- burying the pre/post work under a ``j == <boundary>``
    guard destroys the sequential-fusion locality the outer level otherwise has.
    """

    CATEGORY: str = 'Optimization Preparation'

    target = properties.Property(dtype=str,
                                 default='cpu',
                                 choices=['cpu', 'gpu'],
                                 desc="Target policy: 'gpu' also sinks outer statements into the inner "
                                 "loop to expose the outer axis; 'cpu' distributes only.")

    def __init__(self, target: str = 'cpu'):
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        uniq = UniqueLoopIterators(assign_loop_iterator_post_value=False)
        trivial = PatternMatchAndApplyRepeated([TrivialLoopElimination()])
        rounds = 0
        for _ in range(_MAX_ROUNDS):
            # ``apply_pass`` returns differ by pass type (an int count for MoveIfIntoLoop, a results
            # ``defaultdict`` for the PatternMatchAndApplyRepeated-wrapped TrivialLoopElimination),
            # so test each for truthiness rather than summing.
            changed = False
            if distribute_loops(sdfg):
                changed = True
            if MoveIfIntoLoop().apply_pass(sdfg, {}):
                changed = True
            if trivial.apply_pass(sdfg, {}):
                changed = True
            # Sink LAST in the round: give the distribution the first chance to separate the
            # statements outright, and only sink what is still stuck in an imperfect nest.
            if self.target == 'gpu' and sift_imperfect_nests(sdfg):
                changed = True
            if not changed:
                break
            # SSA-rename cloned iterators (a pure relabelling; not a fixpoint signal, it just keeps
            # the next round's matchers clean). Runs ONLY after a round that changed something --
            # otherwise a refusing pass would still rename every iterator and drop names from
            # ``sdfg.symbols``, mutating an SDFG it did not apply to (a purity violation the empty
            # early-out avoids).
            uniq.apply_pass(sdfg, {})
            rounds += 1
        return rounds or None


__all__ = ['PerfectLoopNesting', 'distribute_loops', 'level_parallel', 'parallel_level_diagnostic']
