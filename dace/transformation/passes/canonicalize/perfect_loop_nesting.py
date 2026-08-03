# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PerfectLoopNesting`` -- duplicate the loop parent per data-independent
statement group, to a fixpoint, forming a perfect nest of single-statement
loops so each statement parallelizes independently.

This is the loop-side analog of the map-side ``PerfLoopNesting``
(``dace/transformation/dataflow/perf_loop_nesting.py``). It runs **just after
statement fission** (``SplitStatements``) and **after ``BreakAntiDependence``**:
BreakAntiDependence first snapshot-breaks a forward-read anti-dependence
(``a[i] = ..; d[i] = a[i] + a[i+1]``) so the two statements become
data-independent, letting this pass then distribute the parent loop per
statement (break-antidep first ⇒ nest more).

The distribution is done by composing the existing, proven primitives to a
fixpoint (each round exposes the next nesting level; ``MapFusion`` later re-fuses
whatever should recombine):

* ``LoopFission`` -- partition the loop body into data-independent groups
  (``_independent_groups`` / ``_independent_block_groups``: dataflow union-find,
  merge on any shared *written* container, sever per-iteration ``a[i]``-only
  bridges) and clone the parent loop once per group. This is the "duplicate the
  loop parent per data-independent statement" step; its group analysis is the
  soundness crux (an unsound merge under-fissions and misses parallelism; an
  unsound split would corrupt a real recurrence -- so a shared written container
  that is *not* provably per-iteration-independent keeps the statements together).
* ``MoveIfIntoLoop`` -- push a guarding conditional down into each distributed
  loop so a per-statement guard travels with its statement.
* ``TrivialLoopElimination`` -- drop a now-single-iteration loop wrapper.
* ``UniqueLoopIterators`` -- SSA-rename the cloned iterators.

A no-op when the body is already a single statement per loop.
"""
from typing import Any, Dict, Optional

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize.sift_statements_into_perfect_nest import sift_imperfect_nests
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

#: Safety bound on fixpoint rounds -- a perfect nest is at most this deep in
#: practice; the loop breaks as soon as a round changes nothing.
_MAX_ROUNDS = 8


@properties.make_properties
@transformation.explicit_cf_compatible
class PerfectLoopNesting(ppl.Pass):
    """Form perfect nests, by DISTRIBUTING the parent loop per data-independent statement
    group and -- on GPU -- by SINKING the outer statements fission cannot separate.

    The two directions are complementary, which is why one pass owns both. Fission handles
    ``for i: {S1; S2}`` where the statements are independent, splitting the parent. It is a
    no-op on ``for i: {pre; for j: body; post}`` whenever ``pre`` feeds ``body`` -- the usual
    case, since that is why they share a nest -- because the group analysis merges on any
    shared written container. And even when they ARE independent, distributing yields three
    loops rather than the single 2-D nest a GPU grid needs. So ``target='gpu'`` additionally
    sinks the surviving pre/post blocks into the inner loop under boundary guards
    (``sift_imperfect_nests``; see that module for the S1 / S7 soundness gates). On CPU the
    sink is skipped -- burying the pre/post work under a ``j == <boundary>`` guard destroys
    the sequential-fusion locality the outer level otherwise has.
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

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        uniq = UniqueLoopIterators(assign_loop_iterator_post_value=False)
        trivial = PatternMatchAndApplyRepeated([TrivialLoopElimination()])
        rounds = 0
        for _ in range(_MAX_ROUNDS):
            # ``apply_pass`` returns differ by pass type (int count for
            # LoopFission/MoveIfIntoLoop, a results ``defaultdict`` for the
            # PatternMatchAndApplyRepeated-wrapped TrivialLoopElimination), so
            # test each for truthiness rather than summing.
            changed = False
            if LoopFission().apply_pass(sdfg, {}):
                changed = True
            if MoveIfIntoLoop().apply_pass(sdfg, {}):
                changed = True
            if trivial.apply_pass(sdfg, {}):
                changed = True
            # Sink LAST in the round: give fission the first chance to separate the statements
            # outright, and only sink what is still stuck in an imperfect nest.
            if self.target == 'gpu' and sift_imperfect_nests(sdfg):
                changed = True
            if not changed:
                break
            # SSA-rename cloned iterators (a pure relabelling; not a fixpoint
            # signal, it just keeps the next round's matchers clean). Runs ONLY after
            # a round that changed something -- otherwise a refusing pass would still
            # rename every iterator and drop names from ``sdfg.symbols``, mutating an
            # SDFG it did not apply to (a purity violation the empty early-out avoids).
            uniq.apply_pass(sdfg, {})
            rounds += 1
        return rounds or None


__all__ = ['PerfectLoopNesting']
