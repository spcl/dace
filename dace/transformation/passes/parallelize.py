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
2. ``UniqueLoopIterators``, with NO post-value assignment -- SSA-rename every loop variable to a
   unique ``_loop_it_<N>``. Unrolling an outer loop replicates its inner loops, and the copies all
   keep the original iterator name; ``LoopToMap`` then refuses each copy but the last, because the
   shared name is read by a LATER block ("loop-defined symbol used after the loop"). Without this
   stage the lift reaches almost nothing on an unrolled Fortran graph -- and with the pass's default
   post-value state it would reintroduce the very read it exists to remove.
3. ``PrivatizeScalars`` -- give each use its own copy of the scalar temporaries. The Fortran
   frontend emits ONE transient per local for the whole routine, so ten different loops write the
   same ``zqadj``; ``LoopToMap`` exempts a transient that lives only inside the loop it is lifting,
   and sharing disqualifies all ten at once. Their writes then fail the ``a*i+b`` uniqueness test on
   a scalar's ``dst_subset=0``, which is 246 of the 249 refusals measured on CloudSC. The dependence
   is false -- the ``z*`` locals are thread-private and reassigned before use every iteration.
4. ``SimplifyPass`` -- a body guarding on the iteration variable (``if jm == ncldqi``) only exposes
   a constant condition once unrolling pins the variable to a literal, so the caller's own simplify,
   run while it was still symbolic, could not fold it. The dead branches left behind still hold
   constant-index writes that read as loop-carried conflicts and block stage 5. Running after the
   fission also lets it drop the private copies nothing ends up needing.
5. ``ParallelizeLoops`` -- lift every loop now free of loop-carried dependencies to a Map. It
   applies ``LoopToMap`` outermost-first, sharing the per-SDFG analysis across probes and caching
   refusals, then sweeps once more in graph order to pick up what the outermost-first order misses
   (measured on CloudSC: the same 314 maps as the plain matcher, 515.2s against 927.7s).
6. :data:`FUSE_ROUNDS` x ``Pipeline([FuseStates, FuseMaps, FuseLoops, FuseConditions])`` -- maps
   fuse only within a state, fusing maps in turn frees the state boundaries the fused maps were
   pinning, and the loop/condition passes remove the two other things that keep fusable bodies
   apart: a pair of sequential loops ``ParallelizeLoops`` had to refuse (map fusion only ever sees
   MapEntry nodes) and a pair of guards holding two halves in separate ConditionalBlocks. One round
   returns with the graph still shrinking; two is the recipe.

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

#: Rounds of (StateFusionExtended -> FuseMaps) run after the loops have become maps.
FUSE_ROUNDS: int = 2


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops: unroll, uniquify, privatize, simplify, lift to maps, then fuse.

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
        from dace.transformation.pass_pipeline import Pipeline
        from dace.transformation.passes.canonicalize.fuse_conditions import FuseConditions
        from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops
        from dace.transformation.passes.fuse_maps import FuseMaps
        from dace.transformation.passes.fusion_inline import FuseStates
        from dace.transformation.passes.parallelize_loops import ParallelizeLoops
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation.passes.symbol_ssa import SymbolSSA
        from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

        stages: List[ppl.Pass] = [
            ShortLoopUnroll(self.unroll_limit),
            # Immediately after the unroll: each replay reassigns the same frontend index symbol on
            # the edge feeding its copy, so one name carries N values and nothing may be reordered
            # or re-guarded around the chain. Versioning the definitions removes that false
            # dependence while the unrolled chains are still intact.
            SymbolSSA(),
            # ``assign_loop_iterator_post_value=False``: the post-value state materializes
            # ``<orig_var> = <exit value>`` AFTER the loop, which is a read of a loop-defined symbol
            # from a later block -- exactly what LoopToMap refuses. Emitting it here would undo the
            # rename's whole purpose.
            UniqueLoopIterators(assign_loop_iterator_post_value=False),
            # Before the simplify: fission renames each use to its own copy (1426 z* scalars ->
            # 2676 on CloudSC), and the fold then gets to clean up the copies nothing needs.
            PrivatizeScalars(),
            SimplifyPass(),
            ParallelizeLoops(),
        ]
        for _ in range(FUSE_ROUNDS):
            # One Pipeline, four fusion passes, each the pass form of its transformation:
            # ``FuseStates`` drives ``StateFusionExtended`` per CFG edge instead of re-enumerating
            # whole-SDFG matches after every apply, and additionally splices out an empty state
            # sitting next to a LoopRegion / ConditionalBlock -- an edge whose endpoints are not
            # both ``SDFGState``, which ``StateFusionExtended`` structurally cannot match.
            # ``FuseMaps`` fuses vertically and horizontally in one FindSingleUseData scan.
            # ``FuseLoops`` joins the sequential loops ``ParallelizeLoops`` had to refuse (map
            # fusion only ever sees MapEntry nodes, so those pairs are invisible to it), and
            # ``FuseConditions`` folds the guards that keep otherwise fusable bodies apart.
            # A fresh Pipeline per round: FuseMaps declares a FindSingleUseData dependency whose
            # results the Pipeline caches, and the second round runs on a graph the first rewrote.
            stages.append(Pipeline([FuseStates(), FuseMaps(), FuseLoops(), FuseConditions()]))
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
