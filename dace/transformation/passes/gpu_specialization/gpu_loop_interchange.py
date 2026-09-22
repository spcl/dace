# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU column form: ``for k { map i; if c: map i }`` becomes ``map i { for k { ..; if c: .. } }``.

Fork/join cost model. On a GPU every top-level map is a kernel launch, i.e. one fork/join. A loop of ``K`` trips over
``M`` maps pays ``K * M`` of them; after the interchange it pays one. Threads index the same lane axis in both forms
and each does the same work in total, so work and parallelism cancel and the launches alone decide: interchange
when ``K * M > 1``. Extents are compared with every symbol taken as the same large size ``S``: a numeric trip
count is exact, a symbolic one is ``S``.

Coalescing still vetoes: when the LOOP axis is strictly more contiguous than every lane axis, the unit-stride
accesses belong on the threads, and moving the loop inside would fix them as a sequential per-thread walk over
strided threads. Legality is :class:`~dace.transformation.interstate.move_loop_into_map.MoveLoopIntoMap` with
``cfg_body``, which refuses any dependence between lanes.
"""
import math
from typing import Any, Dict, Optional

from dace import SDFG, properties, symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap, lane_maps
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import stride_costs


def launches_saved(loop: LoopRegion) -> float:
    """Kernel launches the interchange saves, ``trips * maps - 1``; ``inf`` for a symbolic trip count.

    Every map in the body counts, branches included (an upper bound on a trip's launches).
    """
    maps = len(lane_maps(loop))
    start, end, step = (loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop),
                        loop_analysis.get_loop_stride(loop))
    if maps == 0 or start is None or end is None or step is None:
        return math.inf if maps else 0
    trips = symbolic.simplify((end - start) / step + 1)
    if not trips.is_Number:
        return math.inf
    return max(int(trips), 0) * maps - 1


def interchange_pays_on_gpu(loop: LoopRegion, sdfg: SDFG) -> bool:
    """The module's cost model: launches saved, unless the loop axis is the contiguous one."""
    costs = stride_costs(loop, sdfg)
    return costs is not None and launches_saved(loop) > 0 and not costs[0] < costs[1]


@properties.make_properties
@transformation.explicit_cf_compatible
class GPULoopInterchange(ppl.Pass):
    """Interchange every loop the fork/join cost model approves; ``MoveLoopIntoMap(cfg_body)`` judges legality."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Interchange until no loop qualifies.

        :param sdfg: The device-neutral SDFG to specialize, in place.
        :returns: The number of loops interchanged, or ``None`` if none.
        """
        xform = MoveLoopIntoMap()
        xform.cfg_body = True
        applied = 0
        changed = True
        while changed:
            changed = False
            # An interchange nests the loop in a new SDFG, so the walk restarts after each one.
            for loop in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
                if not interchange_pays_on_gpu(loop, loop.sdfg):
                    continue
                xform.setup_match(loop.sdfg, -1, -1, {MoveLoopIntoMap.loop: loop}, 0, override=True)
                if not xform.can_be_applied(loop.parent_graph, 0, loop.sdfg):
                    continue
                xform.apply(loop.parent_graph, loop.sdfg)
                applied += 1
                changed = True
                break
        return applied or None
