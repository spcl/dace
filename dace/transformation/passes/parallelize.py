# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-shot pipeline that turns sequential loops into parallel maps.

The recipe, applied once in order:

0. ``specialize_symbols`` -- bake ``specialize_constants`` into the graph, recursively through
   nested SDFGs. Load-bearing rather than cosmetic: on cloudsc NONE of the 150 loops has a constant
   trip count until the species PARAMETERs (``nclv`` and the ``ncldq*`` indices) are baked, and 37
   do afterwards -- so without this step stage 1 has nothing to fire on. Skipped when no constants
   are given.
1. :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll` -- fully unroll
   constant-trip loops with ``<= unroll_limit`` iterations, so small recurrence / reduction loops
   become inline straight-line code rather than atomically-parallelized maps.
2. ``UniqueLoopIterators`` -- SSA-rename every loop variable to a unique ``_loop_it_<N>``. Unrolling
   an outer loop replicates its inner loops, and the copies all keep the original iterator name;
   ``LoopToMap`` then refuses each copy but the last, because the shared name is read by a LATER
   block ("loop-defined symbol used after the loop"). Without this stage the lift reaches almost
   nothing on an unrolled Fortran graph.
3. ``SimplifyPass`` -- a body guarding on the iteration variable (``if jm == ncldqi``) only exposes
   a constant condition once unrolling pins the variable to a literal, so the caller's own simplify,
   run while it was still symbolic, could not fold it. The dead branches left behind still hold
   constant-index writes that read as loop-carried conflicts and block stage 4.
4. ``LoopToMap`` -- parallelize every loop now free of loop-carried dependencies.
5. :data:`FUSE_ROUNDS` x (``StateFusionExtended`` -> ``FullMapFusion``) -- maps fuse only within a
   state, and fusing maps in turn frees the state boundaries the fused maps were pinning, so the two
   alternate. One round returns with the graph still shrinking; two is the recipe.

The pipeline runs once: every stage is idempotent or internally exhaustive, so there is nothing to
re-apply. It is modelled on the canonicalization pipeline (a single-shot ``ppl.Pass``) rather than
``Pipeline``/``FixedPointPipeline``, which would forbid re-using a pass type and re-run on a
fixed-point loop.

Transformation classes are imported lazily inside ``_stages``: importing them at module load would
cycle (this module is imported by ``dace.transformation.passes`` whose subpackages those
transformations import).
"""
from typing import Any, Dict, List, Optional

from dace import properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll, DEFAULT_UNROLL_LIMIT

#: Rounds of (StateFusionExtended -> FullMapFusion) run after the loops have become maps.
FUSE_ROUNDS: int = 2


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops: unroll, uniquify iterators, simplify, lift to maps, then fuse.

    Composes the passes once, imperatively. Does not re-run.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: Forwarded to :class:`ShortLoopUnroll` (0 disables).
    :param specialize_constants: Symbol values to bake in before anything else runs. Without them a
                                 loop whose bound is a symbolic PARAMETER has no constant trip count,
                                 so ``ShortLoopUnroll`` refuses it.
    """

    CATEGORY: str = 'Optimization Preparation'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')
    unroll_limit = properties.Property(dtype=int,
                                       default=DEFAULT_UNROLL_LIMIT,
                                       desc='See ShortLoopUnroll (0 disables).')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 specialize_constants: Optional[Dict[str, Any]] = None):
        self.validate = validate
        self.validate_all = validate_all
        self.unroll_limit = unroll_limit
        self._specialize_constants = specialize_constants or {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _stages(self) -> List[ppl.Pass]:
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
        from dace.transformation.pass_pipeline import Pipeline
        from dace.transformation.passes.full_map_fusion import FullMapFusion
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

        stages: List[ppl.Pass] = [
            ShortLoopUnroll(self.unroll_limit),
            UniqueLoopIterators(),
            SimplifyPass(),
            PatternMatchAndApplyRepeated([LoopToMap()]),
        ]
        for _ in range(FUSE_ROUNDS):
            stages.append(PatternMatchAndApplyRepeated([StateFusionExtended()]))
            # A fresh Pipeline per round: FullMapFusion declares a FindSingleUseData dependency whose
            # results the Pipeline caches, and the second round runs on a graph the first rewrote.
            stages.append(Pipeline([FullMapFusion()]))
        return stages

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Parallelize ``sdfg`` in place.

        :param sdfg: The SDFG to parallelize.
        :returns: The number of stages applied. The specialization is not one of them -- it is what
                  makes the first stage applicable, not a stage in its own right.
        """
        if self._specialize_constants:
            from dace.sdfg.utils import specialize_symbols
            specialize_symbols(sdfg, self._specialize_constants)
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
                specialize_constants: Optional[Dict[str, Any]] = None) -> SDFG:
    """Parallelize ``sdfg``'s loops in place and return it.

    :param sdfg: The SDFG to parallelize.
    :param validate: Validate the SDFG after parallelization.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: See :class:`ShortLoopUnroll`.
    :param specialize_constants: Symbol values to bake in first; see :class:`ParallelizePipeline`.
    :returns: The same ``sdfg`` instance, parallelized.
    """
    ParallelizePipeline(validate=validate,
                        validate_all=validate_all,
                        unroll_limit=unroll_limit,
                        specialize_constants=specialize_constants).apply_pass(sdfg, {})
    return sdfg
