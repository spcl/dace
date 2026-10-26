# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ParallelizeLoops`` that keeps a sibling-loop chain whole for ``WavefrontSkew``.

lu's outer ``i`` loop holds two sibling ``j`` loops over adjacent ranges: the ``j < i`` column update, a recurrence,
and the ``j >= i`` row update, which LoopToMap proves DOALL. Mapped on its own, the row update leaves the column
update a lone recurrence and the ``(i, j)`` wavefront the two spell together is gone. Held back here, the chain reaches
``WavefrontSkew``, which either skews it or refuses; the LoopToMap round after the skew maps what it left.

A chain whose siblings are all DOALL is no wavefront and is mapped as usual.
"""

from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopLift
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import plan_guarded_fusion
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops


def liftable(loop: LoopRegion, permissive: bool, like: LoopLift | None = None) -> bool:
    """Whether LoopToMap accepts ``loop`` as the graph stands, its scale aside: a loop over SMALL symbols stays
    sequential for its size, not for a recurrence, so it is no wavefront half.

    :param like: a probe of a loop in the same SDFG whose shared analysis (``lift_context``, ``symbol_resolver``)
                 this one reuses instead of deriving its own.
    """
    probe = LoopLift()
    probe.keep_small_loops_sequential = False
    if like is not None:
        probe.lift_context = like.lift_context
        probe.symbol_resolver = like.symbol_resolver
    probe.loop = loop
    # The dependence probe alone: the sibling hold is what asks.
    return ParallelizeLoops(permissive=permissive).probe(probe, loop, [])


def held_sibling(loop: LoopRegion, permissive: bool, like: LoopLift | None = None) -> bool:
    """Whether ``loop`` is a sibling of a chain that also holds a loop LoopToMap refuses; ``like`` as for
    :func:`liftable`."""
    parent = loop.parent_graph
    if not isinstance(parent, LoopRegion):
        return False
    plan = plan_guarded_fusion(parent)
    if plan is None or loop not in plan.loops:
        return False
    return any(not liftable(other, permissive, like) for other in plan.loops if other is not loop)


class ParallelizeLoopsHoldingSiblingChains(ParallelizeLoops):
    """:class:`ParallelizeLoops` that leaves every sibling of a chain with a refused sibling a loop."""

    def holds_back(self, xform: LoopLift, loop: LoopRegion) -> bool:
        # After ``loop``'s own probe: re-probing its siblings for each of the many loops that probe refuses was the
        # bulk of cloudsc's canonicalize time.
        return held_sibling(loop, self.permissive, xform)
