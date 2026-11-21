# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared utilities of the GPU-specialization passes."""

from dace import dtypes
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.scope import is_in_scope
from dace.transformation.helpers import get_parent_maps

# The legacy ambient-stream symbol, so expanded code is valid under either codegen.
STREAM_CONNECTOR = CURRENT_STREAM_NAME

#: Threads in one warp; a map nested in a kernel with provably fewer iterations stays one thread's loop.
WARP_THREADS = 32


def statically_narrower_than_warp(entry: nodes.MapEntry) -> bool:
    """Whether ``entry`` provably has fewer iterations than one warp has lanes (a symbolic size is assumed big)."""
    return (entry.map.range.num_elements() < WARP_THREADS) == True


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def is_stream_wiring_applied(sdfg: SDFG) -> bool:
    """Whether wiring already produced ``gpu_streams``; scheduling persists in ``Node.gpu_stream_id``."""
    return get_gpu_stream_array_name() in sdfg.arrays


def enclosing_map_chain(state: SDFGState, node: nodes.Node, schedule: dtypes.ScheduleType) -> list[nodes.MapEntry]:
    """Outermost-first chain of the schedule maps enclosing node in state; earlier passes may leave the
    scope cache stale."""
    state._clear_scopedict_cache()
    return [
        entry
        for entry, entry_state in reversed(get_parent_maps(state, node))
        if entry_state is state and isinstance(entry, nodes.MapEntry) and entry.map.schedule == schedule
    ]


def innermost_enclosing_map(state: SDFGState, node: nodes.Node, schedule: dtypes.ScheduleType) -> nodes.MapEntry | None:
    """Innermost ``MapEntry`` with ``schedule`` enclosing ``node``, or None."""
    chain = enclosing_map_chain(state, node, schedule)
    return chain[-1] if chain else None


def is_inside_gpu_device_kernel(sub_sdfg: SDFG) -> bool:
    """Whether ``sub_sdfg`` is, transitively, the body of a GPU_Device map."""
    return is_in_scope(
        sub_sdfg.parent_sdfg, sub_sdfg.parent, sub_sdfg.parent_nsdfg_node, [dtypes.ScheduleType.GPU_Device]
    )


def in_scope_of(state: SDFGState, node: nodes.Node, schedules) -> bool:
    """Whether ``node`` is, or is enclosed by, a map with one of ``schedules``, across nested SDFGs."""
    if isinstance(node, nodes.MapEntry) and node.map.schedule in schedules:
        return True
    return is_in_scope(state.sdfg, state, node, schedules)


def weakly_connected_node_sets(graph) -> list[OrderedSet]:
    """Weakly connected components of ``graph``, in node order (networkx yields hash order, which varies per run)."""
    import networkx as nx

    order = {node: index for index, node in enumerate(graph.nodes())}
    components = [sorted(c, key=order.__getitem__) for c in nx.weakly_connected_components(graph.nx)]
    return [OrderedSet(c) for c in sorted(components, key=lambda c: order[c[0]])]


def is_gpu_copy_or_fill_libnode(node, sdfg: SDFG, state: SDFGState) -> bool:
    """``CopyLibraryNode`` / ``FillLibraryNode`` whose storage involves GPU memory."""
    from dace.libraries.standard.nodes.copy import CopyLibraryNode
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    if isinstance(node, CopyLibraryNode):
        return (
            node.src_storage(state) in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
            or node.dst_storage(state) in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
        )
    if isinstance(node, FillLibraryNode):
        for e in state.out_edges(node):
            if e.data and e.data.data and sdfg.arrays[e.data.data].storage in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES:
                return True
    return False


def is_gpu_kernel_launcher(node) -> bool:
    """True for a ``GPU_Device`` kernel entry, which binds the stream handle on entry."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """A kernel entry, a GPU copy or fill library node, or a lowered runtime-call tasklet."""
    return (
        is_gpu_kernel_launcher(node)
        or is_gpu_copy_or_fill_libnode(node, sdfg, state)
        or is_already_lowered_gpu_runtime_call(node)
    )


def is_already_lowered_gpu_runtime_call(node) -> bool:
    """A tasklet issuing a stream-bound runtime call: a ``gpuStream_t`` in-connector or :data:`STREAM_CONNECTOR` in its
    body. Pipeline sync tasklets are excluded."""
    if not isinstance(node, nodes.Tasklet):
        return False
    if is_pipeline_sync_tasklet(node):
        return False
    if has_stream_connector(node):
        return True
    return STREAM_CONNECTOR in node.code.as_string


SYNC_TASKLET_LABELS = ("gpu_streams_synchronization", "gpu_stream_synchronization")


def is_pipeline_sync_tasklet(node) -> bool:
    """A sync tasklet emitted by the stream pipeline, identified by its canonical label."""
    return isinstance(node, nodes.Tasklet) and node.label in SYNC_TASKLET_LABELS


def is_gpu_relevant_node(node, sdfg: SDFG, state: SDFGState) -> bool:
    """Nodes implying the enclosing component involves GPU work: the stream consumers plus the
    AccessNodes of ``GPU_Global`` arrays."""
    if is_gpu_stream_consumer(node, sdfg, state):
        return True
    if isinstance(node, nodes.AccessNode):
        return sdfg.arrays[node.data].storage == dtypes.StorageType.GPU_Global
    return False


def has_stream_connector(node) -> bool:
    """Whether ``node`` carries an in-connector typed ``gpuStream_t``, whatever its name."""
    return any(t is not None and t == dtypes.gpuStream_t for t in node.in_connectors.values())


def add_gpu_stream_connector(node, conn_name: str, *, single_stream: bool):
    """Add a GPU-stream input connector: a scalar ``gpuStream_t`` under ``single_stream``, else a
    pointer to the whole ``gpu_streams`` array, which the consumer indexes by id."""
    dtype = dtypes.gpuStream_t if single_stream else dtypes.pointer(dtypes.gpuStream_t)
    node.add_in_connector(conn_name, dtype)


def find_inner_gpu_consumers(sdfg: SDFG):
    """Yield ``(node, sdfg, state)`` for every GPU stream consumer in ``sdfg`` and its nested SDFGs."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if is_gpu_stream_consumer(node, nsdfg, state):
                    yield node, nsdfg, state


def persisted_stream_assignments(sdfg: SDFG) -> dict[nodes.Node, int]:
    """Every ``Node.gpu_stream_id`` set across the hierarchy; the per-node property is the durable record."""
    return {
        n: n.gpu_stream_id
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Node) and n.gpu_stream_id is not None
    }
