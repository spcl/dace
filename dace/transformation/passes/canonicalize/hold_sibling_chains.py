# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ParallelizeLoops`` that keeps a sibling-loop chain whole for ``WavefrontSkew``.

lu's outer ``i`` loop holds two sibling ``j`` loops over adjacent ranges: the ``j < i`` column update, a recurrence,
and the ``j >= i`` row update, which LoopToMap proves DOALL. Mapped on its own, the row update leaves the column
update a lone recurrence and the ``(i, j)`` wavefront the two spell together is gone. Held back here, the chain reaches
``WavefrontSkew``, which either skews it or refuses; the LoopToMap round after the skew maps what it left.

A chain whose siblings are all DOALL is no wavefront and is mapped as usual.
"""

from typing import Any, Callable

from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import plan_guarded_fusion
from dace.transformation.passes.parallelize_loops import ParallelizeLoops


def liftable(loop: LoopRegion, permissive: bool) -> bool:
    """Whether LoopToMap accepts ``loop`` as the graph stands."""
    probe = LoopToMap()
    probe.setup_match(loop.sdfg, -1, -1, {LoopToMap.loop: loop}, 0, override=True)
    return probe.can_be_applied(loop.parent_graph, 0, loop.sdfg, permissive=permissive)


def held_sibling(loop: LoopRegion, permissive: bool) -> bool:
    """Whether ``loop`` is a sibling of a chain that also holds a loop LoopToMap refuses."""
    parent = loop.parent_graph
    if not isinstance(parent, LoopRegion):
        return False
    plan = plan_guarded_fusion(parent)
    if plan is None or loop not in plan.loops:
        return False
    return any(not liftable(other, permissive) for other in plan.loops if other is not loop)


class ParallelizeLoopsHoldingSiblingChains(ParallelizeLoops):
    """:class:`ParallelizeLoops` that leaves every sibling of a chain with a refused sibling a loop."""

    def lift(
        self,
        xform: LoopToMap,
        loop: LoopRegion,
        pipeline_results: dict[str, Any],
        proven: bool = False,
        before_apply: Callable[[], None] | None = None,
    ) -> bool:
        if not proven and held_sibling(loop, self.permissive):
            return False
        return super().lift(xform, loop, pipeline_results, proven, before_apply)
