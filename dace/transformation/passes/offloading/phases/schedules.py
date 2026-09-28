# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPU_Device`` at a host level, ``Sequential`` below one; a host map keeps its body a host level."""
from typing import Dict, List, Optional

from ordered_set import OrderedSet

from dace import dtypes
from dace.sdfg import nodes, SDFG


def set_schedule(node: nodes.Node, schedule: dtypes.ScheduleType) -> None:
    if isinstance(node, nodes.MapEntry):
        node.map.schedule = schedule
    else:
        node.schedule = schedule


def assign_schedules(sdfg: SDFG,
                     host_map_entries: OrderedSet,
                     pinned_maps: OrderedSet,
                     host_level: bool = True,
                     via_host_map: bool = False) -> None:
    """Schedule every map and library node; a pinned map stays on the host with its whole subtree, and below a
    host level everything is ``Sequential`` (``Default`` can be lowered to CUDA in the wrong places)."""

    def walk(scope_children: Dict[Optional[nodes.Node], List[nodes.Node]], entry: Optional[nodes.MapEntry],
             host_level: bool, via_host_map: bool) -> None:
        for node in scope_children.get(entry, ()):
            on_host = node in host_map_entries or node in pinned_maps
            if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
                set_schedule(
                    node,
                    dtypes.ScheduleType.GPU_Device if host_level and not on_host else dtypes.ScheduleType.Sequential)
            if isinstance(node, nodes.MapEntry):
                # A host map does not consume the host level: what it launches is still host code.
                is_host_map = host_level and node in host_map_entries and node not in pinned_maps
                walk(scope_children, node, is_host_map, via_host_map or is_host_map)
            elif isinstance(node, nodes.NestedSDFG):
                # A host level only below a host map: elsewhere a nested map the copy analysis never
                # placed would become a kernel reading host memory (npbench spmv).
                assign_schedules(node.sdfg, host_map_entries, pinned_maps, host_level and via_host_map, via_host_map)

    for state in sdfg.states():
        walk(state.scope_children(), None, host_level, via_host_map)
