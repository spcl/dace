# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-shot pipeline that turns sequential loops into parallel maps.

The recipe is the set of pre-passes that lead up to ``LoopToMap`` (and
``LoopToMap`` itself), applied once in order:

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

from dace import properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
# Import the concrete transformation modules (not the ``dataflow`` / ``interstate``
# package ``__init__``) to avoid a circular import: those packages pull in every
# transformation, some of which import ``dace.transformation.passes`` -- which is
# what loads this module.
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.scalar_fission import PrivatizeScalars


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops, lifting reductions on the way.

    Applies the parallelization recipe once, imperatively. The pipeline does not
    re-run: each stage is internally exhaustive or idempotent.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after each stage.
    """

    CATEGORY: str = 'Optimization Preparation'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')

    def __init__(self, validate: bool = False, validate_all: bool = False):
        self.validate = validate
        self.validate_all = validate_all

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Parallelize ``sdfg`` in place.

        :param sdfg: The SDFG to parallelize.
        :returns: The number of stages applied.
        """
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
        return len(stages)


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
