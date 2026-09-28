# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPU_Device`` at a host level, ``Sequential`` below one; a host map keeps its body a host level."""
from typing import Dict, List, Optional

from ordered_set import OrderedSet

from dace import dtypes
from dace.sdfg import nodes, SDFG


def set_schedule(node: nodes.Node, host_level: bool, in_kernel: bool, host_map_entries: OrderedSet,
                 sequential_innermaps: bool) -> None:
    if host_level and node not in host_map_entries:
        schedule = dtypes.ScheduleType.GPU_Device
    elif in_kernel and not sequential_innermaps:
        return
    else:
        # Sequential specifically: Default can be lowered to CUDA in the wrong places.
        schedule = dtypes.ScheduleType.Sequential
    if isinstance(node, nodes.MapEntry):
        node.map.schedule = schedule
    else:
        node.schedule = schedule


def assign_schedules(sdfg: SDFG,
                     host_map_entries: OrderedSet,
                     sequential_innermaps: bool = True,
                     host_level: bool = True,
                     via_host_map: bool = False,
                     in_kernel: bool = False) -> None:
    """Schedule every map and library node of ``sdfg``; ``sequential_innermaps`` off leaves in-kernel ones alone."""

    def walk(scope_children: Dict[Optional[nodes.Node], List[nodes.Node]], entry: Optional[nodes.MapEntry],
             host_level: bool, via_host_map: bool, in_kernel: bool) -> None:
        for node in scope_children.get(entry, ()):
            if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
                set_schedule(node, host_level, in_kernel, host_map_entries, sequential_innermaps)
            if isinstance(node, nodes.MapEntry):
                is_kernel = host_level and node not in host_map_entries
                # A host map does not consume the host level: what it launches is still host code.
                is_host_map = host_level and not is_kernel
                walk(scope_children, node, is_host_map, via_host_map or is_host_map, in_kernel or is_kernel)
            elif isinstance(node, nodes.NestedSDFG):
                # A host level only below a host map: elsewhere a nested map the copy analysis never
                # placed would become a kernel reading host memory (npbench spmv).
                assign_schedules(node.sdfg, host_map_entries, sequential_innermaps, host_level and via_host_map,
                                 via_host_map, in_kernel)

    for state in sdfg.states():
        walk(state.scope_children(), None, host_level, via_host_map, in_kernel)
