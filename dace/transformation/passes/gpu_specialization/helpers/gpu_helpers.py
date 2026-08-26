# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared utilities for the GPU-specialization passes: stream names, node and connector
predicates, and the stream-wiring idempotency signal."""
from typing import List, Optional, Set

from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes
from dace.libraries.standard.helper import CURRENT_STREAM_NAME

# Imported from the libnode layer so producers and the scheduler cannot drift. Named after the
# legacy ambient-stream symbol, so the expanded IR is valid under either codegen.
STREAM_CONNECTOR = CURRENT_STREAM_NAME


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def dependency_edge():
    """Return a fresh empty ``Memlet`` used as a control-dependency edge."""
    from dace.memlet import Memlet
    return Memlet()


def written_by_gpu_map_exit(sdfg: SDFG, name: str) -> bool:
    """Whether ``name`` is written across a GPU-scheduled map's ``MapExit``, i.e. is a kernel output."""
    for state in sdfg.states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            for in_edge in state.in_edges(node):
                src = in_edge.src
                if not isinstance(src, nodes.ExitNode):
                    continue
                entry = state.entry_node(src)
                if entry is not None and entry.map.schedule in dtypes.GPU_SCHEDULES:
                    return True
    return False


def is_stream_wiring_applied(sdfg: SDFG) -> bool:
    """Whether wiring already produced the ``gpu_streams`` array. Only wiring is single-shot;
    scheduling persists per node in ``Node.gpu_stream_id``."""
    return get_gpu_stream_array_name() in sdfg.arrays


def enclosing_map_chain(state: SDFGState, node: nodes.Node, schedule: dtypes.ScheduleType) -> List[nodes.MapEntry]:
    """Outermost-first chain of ``MapEntry`` nodes with ``schedule`` enclosing ``node``.

    The ``scope_dict`` cache is invalidated first: earlier passes may have left it stale.
    """
    state._clear_scopedict_cache()
    sdict = state.scope_dict()
    chain: List[nodes.MapEntry] = []
    scope = sdict.get(node)
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule == schedule:
            chain.append(scope)
        scope = sdict.get(scope)
    chain.reverse()
    return chain


def innermost_enclosing_map(state: SDFGState, node: nodes.Node,
                            schedule: dtypes.ScheduleType) -> Optional[nodes.MapEntry]:
    """Innermost ``MapEntry`` with ``schedule`` enclosing ``node``, or None."""
    chain = enclosing_map_chain(state, node, schedule)
    return chain[-1] if chain else None


def is_inside_gpu_device_kernel(sub_sdfg: SDFG) -> bool:
    """Whether ``sub_sdfg`` is, transitively, the body of a GPU_Device map."""
    cur = sub_sdfg
    while cur.parent_nsdfg_node is not None:
        if innermost_enclosing_map(cur.parent, cur.parent_nsdfg_node, dtypes.ScheduleType.GPU_Device) is not None:
            return True
        cur = cur.parent_sdfg
    return False


def weakly_connected_node_sets(graph) -> List[Set[nodes.Node]]:
    """Weakly-connected components of ``graph``'s dataflow, as node sets."""
    import networkx as nx
    return [set(c) for c in nx.weakly_connected_components(graph.nx)]


def is_gpu_copy_or_memset_libnode(node, sdfg: SDFG, state: SDFGState) -> bool:
    """``CopyLibraryNode`` / ``FillLibraryNode`` whose storage involves GPU memory."""
    from dace.libraries.standard.nodes.copy import CopyLibraryNode
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    if isinstance(node, CopyLibraryNode):
        return (node.src_storage(state) in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
                or node.dst_storage(state) in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES)
    if isinstance(node, FillLibraryNode):
        for e in state.out_edges(node):
            if e.data and e.data.data and sdfg.arrays[e.data.data].storage in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES:
                return True
    return False


def is_gpu_kernel_launcher(node) -> bool:
    """True for a ``GPU_Device`` kernel ``MapEntry`` -- the launcher binds the stream handle on enter."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """Nodes that *take* a GPU stream: a kernel ``MapEntry``, a GPU Copy/Fill libnode, or a lowered
    runtime-call Tasklet. AccessNodes are memory references, not consumers."""
    return (is_gpu_kernel_launcher(node) or is_gpu_copy_or_memset_libnode(node, sdfg, state)
            or is_already_lowered_gpu_runtime_call(node))


def is_already_lowered_gpu_runtime_call(node) -> bool:
    """A Tasklet issuing a stream-bound GPU runtime call, detected by a ``gpuStream_t``
    in-connector or a :data:`STREAM_CONNECTOR` reference in its body. Pipeline sync tasklets are
    excluded -- they are not consumers in the WCC sense."""
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
