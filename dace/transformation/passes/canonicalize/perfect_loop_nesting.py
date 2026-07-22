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

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

#: Safety bound on fixpoint rounds -- a perfect nest is at most this deep in
#: practice; the loop breaks as soon as a round changes nothing.
_MAX_ROUNDS = 8


@transformation.explicit_cf_compatible
class PerfectLoopNesting(ppl.Pass):
    """Distribute the parent loop per data-independent statement group to a
    fixpoint, forming a perfect nest of single-statement loops."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

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
