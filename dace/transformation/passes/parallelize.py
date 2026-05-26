# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-shot pipeline that turns sequential loops into parallel maps.

The recipe is the set of pre-passes that lead up to ``LoopToMap`` (and
``LoopToMap`` itself), applied once in order:

0. ``LoopUnroll`` (short loops only) -- fully unroll every constant-trip loop
   whose iteration count is ``<= unroll_limit``, so small recurrence / reduction
   loops (e.g. the length-5 cloudsc species loop, once specialized) become inline
   straight-line code rather than being parallelized with atomics. Runs first so
   the scalar fission below freshly renames the unrolled bodies' scalars.
1. :class:`~dace.transformation.passes.scalar_fission.PrivatizeScalars` --
   privatize loop-local (thread-local) scalars so a loop body no longer carries
   a false scalar dependency across iterations.
2. ``TrivialTaskletElimination`` -- drop the ``__out = __inp`` copy tasklets the
   frontend emits, exposing the bare read-modify-write spine of accumulators.
3. ``AugAssignToWCR`` -- rewrite ``arr[S] += x`` (including the frontend's
   copy-wrapped form) into a write-conflict-resolution write, so the reduction
   loop's write is no longer iteration-indexed and ``LoopToMap`` may map it.
4. :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce` -- lift the
   pure accumulator loops that remain to ``Reduce`` library nodes.
5. ``LoopToMap`` -- parallelize every loop that is now free of loop-carried
   dependencies.

The pipeline is meant to run **after** ``simplify`` and **once**: every stage is
either idempotent or internally exhaustive (``PatternMatchAndApplyRepeated``),
so there is nothing to re-apply. It is modelled on the canonicalization pipeline
rather than :class:`~dace.transformation.pass_pipeline.Pipeline` because the
latter forbids re-using a pass type (``PatternMatchAndApplyRepeated`` appears
several times) and would re-run stages on its fixed-point logic.
"""
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
# Import the concrete transformation modules (not the ``dataflow`` / ``interstate``
# package ``__init__``) to avoid a circular import: those packages pull in every
# transformation, some of which import ``dace.transformation.passes`` -- which is
# what loads this module.
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.scalar_fission import PrivatizeScalars

#: Default trip-count threshold below which a constant-trip loop is unrolled.
_DEFAULT_UNROLL_LIMIT = 8


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops, lifting reductions on the way.

    Applies the parallelization recipe once, imperatively. The pipeline does not
    re-run: each stage is internally exhaustive or idempotent.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: Constant-trip loops with at most this many iterations are
                         fully unrolled before parallelization; ``0`` disables it.
    """

    CATEGORY: str = 'Optimization Preparation'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')
    unroll_limit = properties.Property(
        dtype=int,
        default=_DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations before '
        'parallelization (0 disables). Keeps small recurrence/reduction loops as inline '
        'straight-line code instead of atomically-parallelized maps.')

    def __init__(self, validate: bool = False, validate_all: bool = False, unroll_limit: int = _DEFAULT_UNROLL_LIMIT):
        self.validate = validate
        self.validate_all = validate_all
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _constant_trip_count(self, loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
        """The exact iteration count of ``loop`` if it is constant, else ``None``.

        Matches the iteration count ``LoopUnroll`` would produce
        (``len(range(0, end - start + 1, stride))``).
        """
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None or not loop.loop_variable:
            return None
        if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
            return None
        try:
            stride_val = int(symbolic.evaluate(stride, sdfg.constants))
            diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
        except (TypeError, ValueError):
            return None
        if stride_val <= 0 or diff <= 0:
            return None
        return len(range(0, diff, stride_val))

    def _unroll_short_loops(self, sdfg: SDFG):
        """Fully unroll every constant-trip loop with ``<= unroll_limit`` iterations.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).
        """
        if self.unroll_limit <= 0:
            return
        changed = True
        while changed:
            changed = False
            for loop in [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                trip = self._constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                try:
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop)
                except Exception:
                    continue  # not unrollable in this context; leave it for LoopToMap
                changed = True
                break

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Parallelize ``sdfg`` in place.

        :param sdfg: The SDFG to parallelize.
        :returns: The number of stages applied.
        """
        self._unroll_short_loops(sdfg)
        if self.validate_all:
            sdfg.validate()
        stages = [
            PrivatizeScalars(),
            PatternMatchAndApplyRepeated([TrivialTaskletElimination()]),
            PatternMatchAndApplyRepeated([AugAssignToWCR()]),
            LoopToReduce(),
            PatternMatchAndApplyRepeated([LoopToMap()]),
        ]
        for stage in stages:
            stage.apply_pass(sdfg, {})
            if self.validate_all:
                sdfg.validate()
        if self.validate:
            sdfg.validate()
        return 1 + len(stages)


def parallelize(sdfg: SDFG, validate: bool = True, validate_all: bool = False) -> SDFG:
    """Parallelize ``sdfg``'s loops in place and return it.

    One-call recipe meant to run after ``simplify``.

    :param sdfg: The SDFG to parallelize.
    :param validate: Validate the SDFG after parallelization.
    :param validate_all: Validate the SDFG after each stage.
    :returns: The same ``sdfg`` instance, parallelized.
    """
    ParallelizePipeline(validate=validate, validate_all=validate_all).apply_pass(sdfg, {})
    return sdfg
