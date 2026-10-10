# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPU_Device`` at a host level, ``Sequential`` below one; a host map keeps its body a host level."""

from ordered_set import OrderedSet

from dace import dtypes
from dace.sdfg import SDFG, nodes


def assign_schedules(
    sdfg: SDFG, host_map_entries: OrderedSet, pinned_maps: OrderedSet, host_level: bool = True
) -> None:
    """Schedule every map and library node; a pinned map stays on the host with its whole subtree, and below a
    host level everything is ``Sequential`` (``Default`` can be lowered to CUDA in the wrong places).

    A host map keeps its body a host level, and so does a nested SDFG at one, whose body is placed on its own.
    """

    def walk(
        scope_children: dict[nodes.Node | None, list[nodes.Node]], entry: nodes.MapEntry | None, host_level: bool
    ) -> None:
        for node in scope_children.get(entry, ()):
            on_host = node in host_map_entries or node in pinned_maps
            if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
                node.schedule = (
                    dtypes.ScheduleType.GPU_Device if host_level and not on_host else dtypes.ScheduleType.Sequential
                )
            if isinstance(node, nodes.MapEntry):
                walk(scope_children, node, host_level and node in host_map_entries and node not in pinned_maps)
            elif isinstance(node, nodes.NestedSDFG):
                assign_schedules(node.sdfg, host_map_entries, pinned_maps, host_level)

    for state in sdfg.states():
        walk(state.scope_children(), None, host_level)
