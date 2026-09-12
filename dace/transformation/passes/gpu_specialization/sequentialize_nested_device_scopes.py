# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resolve nested GPU device scopes left in a canonicalized graph.

The GPU counterpart of the CPU band's fork/join model -- and a different kind of decision. On CPU a
nested parallel map is a cost question; on GPU a ``GPU_Device`` map inside another ``GPU_Device``
map is an in-kernel kernel launch, which is not expressible at all. So there is no cost model here:
any device-scheduled map or library node that a device scope re-enters is pinned ``Sequential``,
unconditionally and at any size, transitively across nested-SDFG boundaries.

A library node additionally loses the device schedule inside any enclosing loop, at any trip count:
a per-iteration kernel launch is a hazard on GPU regardless of how short the loop is (the CPU band
makes that same test trip-count aware, which is a CPU-only relaxation).

Runs on the offloaded graph, after schedule inference: ``canonicalize`` leaves the parallelism in
place and device-neutral, and this is where the GPU target resolves it.
"""
from typing import Any, Dict, Optional

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl


@properties.make_properties
class SequentializeNestedDeviceScopes(ppl.Pass):
    """Pin every ``GPU_Device`` scope nested in another device scope to ``Sequential``."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Sequentialize every re-entered device scope.

        :param sdfg: the offloaded SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many scopes were pinned, or ``None`` if none were.
        """
        from dace.transformation.helpers import get_parent_map_and_loop_scopes
        pinned = 0
        for node, state in sdfg.all_nodes_recursive():
            is_map = isinstance(node, nodes.MapEntry)
            if is_map:
                schedule = node.map.schedule
            elif isinstance(node, nodes.LibraryNode):
                schedule = node.schedule
            else:
                continue
            if schedule != dtypes.ScheduleType.GPU_Device:
                continue
            for scope in get_parent_map_and_loop_scopes(sdfg, node, state):
                in_device_map = isinstance(scope,
                                           nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.GPU_Device
                if in_device_map or (not is_map and isinstance(scope, LoopRegion)):
                    if is_map:
                        node.map.schedule = dtypes.ScheduleType.Sequential
                    else:
                        node.schedule = dtypes.ScheduleType.Sequential
                    pinned += 1
                    break
        return pinned or None
