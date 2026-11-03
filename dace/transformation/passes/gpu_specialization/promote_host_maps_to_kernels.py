# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn a host map that only launches device work into the kernel itself.

The device offload never nests one kernel in another, so a parallel map whose body holds a device
map or a device library node stays on the HOST and launches that work once per iteration::

    map i (Sequential, host):         # M launches
        Reduce(A[i, :]) -> out[i]     # a device-wide reduction per row

The map is parallel by definition, so it becomes the ``GPU_Device`` map: one kernel, the inner device
scopes nested in it (serial per thread, or a block collective where the node has one).

Only a launch-bound map is promoted (:func:`launch_bound`): an extent that is not ``SMALL`` whose body's device maps
are ``SMALL`` or relaunched by a sequential loop; wide device maps already amortize their launches.

Only a body the device can run is promoted: data in device-accessible storage (or transients the body
owns, or scalars it only reads), Python tasklets (a native tasklet may call the host runtime, a copy among them), nested SDFGs of
the same kind, and library nodes that have an in-kernel block lowering for their shape.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from dace import SDFG, SDFGState, data, dtypes, properties, symbolic
from dace.libraries.standard.block_reduce import gpu_block_implementation
from dace.sdfg import nodes
from dace.sdfg.sdfg import memoize_facts
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl

#: Storage a transient the body owns can take inside a kernel (a per-thread value, or lifted out of it).
BODY_LOCAL = (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.GPU_Global)
HOST_MAP_SCHEDULES = (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)


def memoized(scan: Callable[..., Any]) -> Callable[..., Any]:
    """``scan`` once per arguments, in the memo that is its last one: nothing writes the SDFG until the analysis ends."""

    @wraps(scan)
    def cached(*args: Any) -> Any:
        key = (scan, *args[:-1])
        if key not in args[-1]:
            args[-1][key] = scan(*args)
        return args[-1][key]

    return cached


@memoized
def constants_of(sdfg: SDFG, memo: dict[Any, Any]) -> dict[str, Any]:
    return sdfg.constants


@memoized
def device_capable(node: nodes.Node, state: SDFGState, owned: bool, memo: dict[Any, Any]) -> bool:
    """Whether a kernel can run ``node``; ``owned`` says a transient here belongs to the body."""
    if isinstance(node, nodes.AccessNode):
        desc = node.desc(state.sdfg)
        if isinstance(desc, data.Scalar) and state.in_degree(node) == 0:
            # A kernel takes a scalar it only reads by value, wherever it lives
            return True
        if owned and not desc.transient and desc.storage == dtypes.StorageType.Default:
            # A nested SDFG's argument has no storage of its own: it is the caller's array, checked out there.
            return True
        return (
            desc.storage in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
            or (owned and desc.transient and desc.storage in BODY_LOCAL)
            or node.data in constants_of(state.sdfg, memo)  # Written into the code that reads it: nothing to refuse
        )
    if isinstance(node, nodes.Tasklet):
        return node.language is dtypes.Language.Python
    if isinstance(node, nodes.LibraryNode):
        return gpu_block_implementation(node, state, state.sdfg) is not None
    if isinstance(node, nodes.NestedSDFG):
        return all(
            device_capable(inner, inner_state, True, memo)
            for inner_state in node.sdfg.states()
            for inner in inner_state.nodes()
        )
    return isinstance(node, (nodes.EntryNode, nodes.ExitNode))


@memoized
def launches_device_work(node: nodes.Node, memo: dict[Any, Any]) -> bool:
    if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
        return node.schedule == dtypes.ScheduleType.GPU_Device
    if isinstance(node, nodes.NestedSDFG):
        return any(
            launches_device_work(inner, memo) for inner_state in node.sdfg.states() for inner in inner_state.nodes()
        )
    return False


def promotable(state: SDFGState, entry: nodes.MapEntry, scopes: list[Any], memo: dict[Any, Any]) -> bool:
    """A host map under ``scopes``, outside every kernel, whose whole body is device work a kernel can run."""
    if any(isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES for scope in scopes):
        return False
    body = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    if not any(launches_device_work(node, memo) for node in body) or not all(
        device_capable(node, state, False, memo) for node in body
    ):
        return False
    return launch_bound(state, entry, body, memo)


def launch_bound(state: SDFGState, entry: nodes.MapEntry, body: list[nodes.Node], memo: dict[Any, Any]) -> bool:
    """Whether the host map's launches cost more than one kernel over it: its extent is not ``SMALL``, and every device
    map in its body is ``SMALL`` or relaunched by a sequential loop. A body whose device maps are wide amortizes its
    launches, and inside one kernel each thread would run them serially."""
    if symbolic.is_small_extent(map_extent(entry), state.sdfg.facts()):
        return False
    return not any(wide_device_map(node, state, False, memo) for node in body)


def map_extent(entry: nodes.MapEntry) -> symbolic.SymbolicType:
    return entry.map.range.num_elements()


@memoized
def wide_device_map(node: nodes.Node, state: SDFGState, in_loop: bool, memo: dict[Any, Any]) -> bool:
    """Whether ``node`` is, or nests, a device map that is neither ``SMALL`` nor under a sequential loop."""
    if isinstance(node, nodes.MapEntry):
        return (
            node.map.schedule == dtypes.ScheduleType.GPU_Device
            and not in_loop
            and not symbolic.is_small_extent(map_extent(node), state.sdfg.facts())
        )
    if isinstance(node, nodes.NestedSDFG):
        return any(
            wide_device_map(inner, inner_state, in_loop or in_a_loop(inner_state), memo)
            for inner_state in node.sdfg.states()
            for inner in inner_state.nodes()
        )
    return False


def in_a_loop(block: SDFGState) -> bool:
    """A ``LoopRegion`` of the block's own SDFG encloses it."""
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


@properties.make_properties
class PromoteHostMapsToKernels(ppl.Pass):
    """Make each host map that only launches device work the kernel (see the module docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Promote the outermost :func:`promotable` host maps: choose all of them, then write the schedules.

        :param sdfg: the offloaded SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many maps became kernels, or ``None`` if none did.
        """
        hosts = [
            (node, state, xfh.get_parent_map_and_loop_scopes(state.sdfg, node, state))
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule in HOST_MAP_SCHEDULES
        ]
        memo: dict[Any, Any] = {}
        chosen: set[nodes.MapEntry] = set()
        # A parent has fewer scopes than its child, so it is decided first; a map under a chosen one is in a kernel.
        with memoize_facts(sdfg):
            for entry, state, scopes in sorted(hosts, key=lambda host: len(host[2])):
                if chosen.isdisjoint(scopes) and promotable(state, entry, scopes, memo):
                    chosen.add(entry)
        for entry in chosen:
            entry.map.schedule = dtypes.ScheduleType.GPU_Device
        return len(chosen) or None
