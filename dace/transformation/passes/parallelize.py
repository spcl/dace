# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-shot pipeline that turns sequential loops into parallel maps.

It composes standalone passes in order, applied once:

0a. :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll` --
    fully unroll constant-trip loops with ``<= unroll_limit`` iterations, so small
    recurrence / reduction loops become inline straight-line code rather than
    atomically-parallelized maps.
0b. :class:`~dace.transformation.passes.parallelization_prep.BestEffortLoopPeeling`
    -- index-set-split a loop at a point derived from its body (an ``if i == x``
    guard, a broadcast read's conflicting index), and peel a wrapping body modulo
    to its floor-correct affine form. Runs before scalar fission so the
    freshly-introduced straight-line bodies' scalars get renamed.
1.  ``PrivatizeScalars`` -- privatize loop-local scalars (drop false carried deps).
2.  ``SymbolPropagation`` + ``ConstantPropagation`` -- peeling / unrolling fold
    concrete iteration indices into the bodies; propagating them simplifies
    bounds and conditions enough to expose more maps.
3.  ``TrivialTaskletElimination`` -- drop the ``__out = __inp`` copy tasklets the
    frontend emits, exposing the bare read-modify-write spine of accumulators.
4.  ``AugAssignToWCR`` -- rewrite ``arr[S] += x`` (incl. the copy-wrapped form)
    into a write-conflict-resolution write so the reduction loop maps.
5.  ``LoopToReduce`` -- lift pure accumulator loops to ``Reduce`` library nodes.
6.  ``LoopToMap`` -- parallelize every loop now free of loop-carried dependencies.

``AccumulatorToMapAndReduce`` deliberately does NOT run here. It parallelizes the
accumulator loops ``LoopToReduce`` refuses, but pays for them with a per-iteration
``(trip,)`` buffer plus a second ``Reduce`` pass over it -- more memory traffic than
the scalar accumulation it replaces, and on cloudsc a net slowdown. Its Map-with-WCR
pattern also partly undoes stage 4 (``AugAssignToWCR`` creates the WCR; that pattern
rewrites it back into buffer + ``Reduce``). The canonicalization pipeline still runs
it, where the surrounding fusion/lift stages can consume the ``Reduce``.

The pipeline runs once: every stage is idempotent or internally exhaustive, so
there is nothing to re-apply. It is modelled on the canonicalization pipeline
(a single-shot ``ppl.Pass``) rather than ``Pipeline``/``FixedPointPipeline``,
which would forbid re-using a pass type and re-run on a fixed-point loop.

Transformation classes are imported lazily inside ``apply_pass``: importing them
at module load would cycle (this module is imported by
``dace.transformation.passes`` whose subpackages those transformations import).
"""
from typing import Any, Dict, List, Optional

from dace import properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.parallelization_prep import (BestEffortLoopPeeling, ShortLoopUnroll, DEFAULT_PEEL_LIMIT,
                                                             DEFAULT_UNROLL_LIMIT)


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops, lifting reductions on the way.

    Composes the parallelization passes once, imperatively. Does not re-run.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: Forwarded to :class:`ShortLoopUnroll` (0 disables).
    :param peel_limit: Forwarded to :class:`BestEffortLoopPeeling` (0 disables).
    """

    CATEGORY: str = 'Optimization Preparation'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')
    unroll_limit = properties.Property(dtype=int,
                                       default=DEFAULT_UNROLL_LIMIT,
                                       desc='See ShortLoopUnroll (0 disables).')
    peel_limit = properties.Property(dtype=int,
                                     default=DEFAULT_PEEL_LIMIT,
                                     desc='See BestEffortLoopPeeling (0 disables).')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 peel_limit: int = DEFAULT_PEEL_LIMIT):
        self.validate = validate
        self.validate_all = validate_all
        self.unroll_limit = unroll_limit
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _stages(self) -> List[ppl.Pass]:
        from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.loop_to_reduce import LoopToReduce
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        from dace.sdfg.propagation import propagate_memlets_sdfg

        # Wrapper so memlet propagation slots into the Pass list. Memlet propagation
        # is value-preserving and computes scope-summary memlets that ``LoopToMap``'s
        # uniqueness analysis later relies on; re-running after unroll+simplify rebuilds
        # those summaries with the now-correct ``symbols_defined_at`` (which folds in
        # enclosing-LoopRegion loop variables -- the fix that kept ``tendency_loc_cld``
        # from being widened to its full array extent in cloudsc).
        class _PropagateMemlets(ppl.Pass):

            def modifies(self_):
                return ppl.Modifies.Memlets

            def should_reapply(self_, _modified):
                return False

            def depends_on(self_):
                return set()

            def apply_pass(self_, sdfg, _pipeline_results):
                propagate_memlets_sdfg(sdfg)
                return 1

        return [
            # Loop-structure transforms first (unroll, peel; reversal lives inside
            # peeling). Then symbol/constant propagation folds the constant
            # iteration values and guard symbols those expose -- it must run after
            # them, not before. The rest (privatize, trivial-tasklet, AugWCR,
            # reduce, loop-to-map) is order-insensitive to propagation.
            ShortLoopUnroll(self.unroll_limit),
            BestEffortLoopPeeling(self.peel_limit),
            # Re-simplify once, after unrolling. The caller simplifies before this
            # pipeline, but a loop body that guards on the iteration variable (e.g.
            # ``if jm == ncldqi``) only exposes a constant condition once unrolling pins
            # ``jm`` to a literal -- so the caller's simplify, run while the guard was
            # still symbolic, could not fold it. Unrolling (with the species constants
            # specialised) then leaves dead branches like ``if 1 == 2`` whose never-taken
            # bodies still hold constant-index writes (e.g. ``zvqx[1] = ...``) that read
            # as a loop-carried conflict and block LoopToMap. Folding the conditions here
            # drops those dead branches and their phantom writes.
            SimplifyPass(),
            # Re-propagate memlets so propagation re-runs with the unrolled / specialised
            # body and the corrected ``symbols_defined_at`` (enclosing-LoopRegion loop
            # variables now folded in). Without this, NSDFG-out connector memlets that
            # propagation had previously widened to the full array extent stay stale and
            # cause ``LoopToMap`` to refuse with "dynamic write not indexed by the
            # iteration variable" -- the cloudsc ``tendency_loc_cld`` shape.
            _PropagateMemlets(),
            SymbolPropagation(),
            ConstantPropagation(),
            PrivatizeScalars(),
            PatternMatchAndApplyRepeated([TrivialTaskletElimination()]),
            PatternMatchAndApplyRepeated([AugAssignToWCR()]),
            LoopToReduce(),
            PatternMatchAndApplyRepeated([LoopToMap()]),
        ]

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Parallelize ``sdfg`` in place.

        :param sdfg: The SDFG to parallelize.
        :returns: The number of stages applied.
        """
        stages = self._stages()
        for stage in stages:
            stage.apply_pass(sdfg, {})
            if self.validate_all:
                sdfg.validate()
        if self.validate:
            sdfg.validate()
        return len(stages)


def parallelize(sdfg: SDFG,
                validate: bool = True,
                validate_all: bool = False,
                unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                peel_limit: int = DEFAULT_PEEL_LIMIT) -> SDFG:
    """Parallelize ``sdfg``'s loops in place and return it.

    One-call recipe meant to run after ``simplify``.

    :param sdfg: The SDFG to parallelize.
    :param validate: Validate the SDFG after parallelization.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: See :class:`ShortLoopUnroll`.
    :param peel_limit: See :class:`BestEffortLoopPeeling`.
    :returns: The same ``sdfg`` instance, parallelized.
    """
    ParallelizePipeline(validate=validate, validate_all=validate_all, unroll_limit=unroll_limit,
                        peel_limit=peel_limit).apply_pass(sdfg, {})
    return sdfg
