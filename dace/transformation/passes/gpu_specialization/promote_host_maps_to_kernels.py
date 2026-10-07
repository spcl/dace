# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn a host map that only launches device work into the kernel itself.

The device offload never nests one kernel in another, so a parallel map whose body holds a device
map or a device library node stays on the HOST and launches that work once per iteration::

    map i (Sequential, host):         # M launches
        Reduce(A[i, :]) -> out[i]     # a device-wide reduction per row

The map is parallel by definition, so it becomes the ``GPU_Device`` map: one kernel, the inner device
scopes nested in it (serial per thread, or a block collective where the node has one).

Only a body the device can run is promoted: data in device-accessible storage (or transients the body
owns), Python tasklets (a native tasklet may call the host runtime, a copy among them), nested SDFGs of
the same kind, and library nodes that have an in-kernel block lowering for their shape.
"""

from typing import Any, Dict, Optional

from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.standard.block_reduce import gpu_block_implementation
from dace.sdfg import nodes
from dace.transformation import helpers as xfh, pass_pipeline as ppl

#: Storage a kernel can address directly.
DEVICE_ACCESSIBLE = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned)
#: Storage a transient the body owns can take inside a kernel (a per-thread value, or lifted out of it).
BODY_LOCAL = (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.GPU_Global)
HOST_MAP_SCHEDULES = (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)


def device_capable(node: nodes.Node, state: SDFGState, owned: bool) -> bool:
    """Whether a kernel can run ``node``; ``owned`` says a transient here belongs to the body."""
    if isinstance(node, nodes.AccessNode):
        desc = node.desc(state.sdfg)
        return desc.storage in DEVICE_ACCESSIBLE or (owned and desc.transient and desc.storage in BODY_LOCAL)
    if isinstance(node, nodes.Tasklet):
        return node.language is dtypes.Language.Python
    if isinstance(node, nodes.LibraryNode):
        return gpu_block_implementation(node, state, state.sdfg) is not None
    if isinstance(node, nodes.NestedSDFG):
        return all(
            device_capable(inner, inner_state, True)
            for inner_state in node.sdfg.states()
            for inner in inner_state.nodes()
        )
    return isinstance(node, (nodes.EntryNode, nodes.ExitNode))


def launches_device_work(node: nodes.Node) -> bool:
    if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
        return node.schedule == dtypes.ScheduleType.GPU_Device
    if isinstance(node, nodes.NestedSDFG):
        return any(launches_device_work(inner) for inner_state in node.sdfg.states() for inner in inner_state.nodes())
    return False


def promotable(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A host map, outside every kernel, whose whole body is device work a kernel can run."""
    if entry.map.schedule not in HOST_MAP_SCHEDULES:
        return False
    if any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES
        for scope in xfh.get_parent_map_and_loop_scopes(state.sdfg, entry, state)
    ):
        return False
    body = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    return any(launches_device_work(node) for node in body) and all(device_capable(node, state, False) for node in body)


@properties.make_properties
class PromoteHostMapsToKernels(ppl.Pass):
    """Make each host map that only launches device work the kernel (see the module docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote every :func:`promotable` host map, outermost first.

        :param sdfg: the offloaded SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many maps became kernels, or ``None`` if none did.
        """
        promoted = 0
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and promotable(state, node):
                node.map.schedule = dtypes.ScheduleType.GPU_Device
                promoted += 1
        return promoted or None
