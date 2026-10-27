# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Parallelize``: lift every loop that parallelizes, then rescue the refused ones, each by the pass for its refusal.

:class:`~.parallelize_loops.ParallelizeLoops` leaves ``{loop: LoopRefusal}`` in the pipeline results
(:data:`~.parallelize_loops.REFUSALS`); every pass after it acts on the loops refused for its reason only, and lifts what
it rescues itself:

* :class:`~.wavefront.skew.WavefrontSkew` -- a dependence between iterations, or a sibling chain held back for it;
* :class:`~.contracts.under_constraint.ParallelizeUnderConstraint` -- a store injective under a constraint;
* :class:`~.contracts.guarded_scatter.ScatterToGuardedMaps` -- a scatter conflict-free under a runtime check.
"""

from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import canon_stats


class Parallelize(ppl.Pipeline):
    """:class:`ParallelizeLoops`, then the rescues of what it refused (see the module docstring).

    :param target: ``"cpu"`` or ``"gpu"``, for the wavefront tiling.
    :param hold_sibling_chains: Leave a sibling-loop chain with a refused sibling whole, for the wavefront skew.
    :param constraints: Version loops parallel under a constraint (:class:`ParallelizeUnderConstraint`).
    :param scatters: Guard scatters by a conflict check (:class:`ScatterToGuardedMaps`).
    :param assume_guards: Assume every constraint and scatter check holds: emit the parallel map only.
    """

    CATEGORY: str = "Optimization Preparation"

    def __init__(
        self,
        target: str = "cpu",
        hold_sibling_chains: bool = True,
        constraints: bool = False,
        scatters: bool = False,
        assume_guards: bool = False,
    ):
        # avoid an import cycle
        from dace.transformation.passes.parallelization.contracts.guarded_scatter import ScatterToGuardedMaps
        from dace.transformation.passes.parallelization.contracts.under_constraint import ParallelizeUnderConstraint
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops
        from dace.transformation.passes.parallelization.sibling_chains import ParallelizeLoopsHoldingSiblingChains
        from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew

        lift = (ParallelizeLoopsHoldingSiblingChains if hold_sibling_chains else ParallelizeLoops)(propagate=False)
        # A loop refused on a scratch buffer only lifts with the buffer expanded per iteration
        lift.expand_buffers = True
        passes: list[ppl.Pass] = [lift, WavefrontSkew(target=target)]
        if constraints:
            passes.append(ParallelizeUnderConstraint(assume_constraint=assume_guards))
        if scatters:
            passes.append(ScatterToGuardedMaps(emit_unparallelized_else_branch=True, assume_no_conflicts=assume_guards))
        super().__init__(passes)

    def apply_pass(self, sdfg, pipeline_results):
        """Run the passes on results of their own: the refusals are of this run's sweep."""
        return super().apply_pass(sdfg, {})

    def apply_subpass(self, sdfg, p, state):
        """Run ``p`` as the stage the canonicalization statistics attribute its loops to."""
        stats = canon_stats.active()
        if stats is None:
            return super().apply_subpass(sdfg, p, state)
        stage = stats.stage
        stats.stage_start(stage[0], p)
        try:
            result = super().apply_subpass(sdfg, p, state)
            stats.stage_done(sdfg)
            return result
        finally:
            stats.stage = stage
