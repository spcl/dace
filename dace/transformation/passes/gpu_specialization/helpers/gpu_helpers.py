# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared utilities for the GPU-specialization passes: canonical stream names,
node/connector predicates (single source of truth so passes don't reimplement
scope walks), and the stream-wiring idempotency signal."""
from typing import List, Optional, Set

from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes
from dace.libraries.standard.helper import CURRENT_STREAM_NAME

# Stream in-connector name, imported from the libnode layer so producers and
# the scheduler cannot drift. Named after the legacy ambient-stream symbol so
# the same expanded IR is valid under both the legacy codegen (which declares
# it) and the experimental codegen (whose type-based prelude binds it).
STREAM_CONNECTOR = CURRENT_STREAM_NAME

# On-GPU / on-CPU storage sets. Deliberately duplicated from the libnode-layer
# sets: the experimental GPU path owns its own constants rather than reaching
# across into ``dace.libraries.standard.helper``.
GPU_RESIDENT_STORAGES = frozenset({dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared})
CPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
})


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def dependency_edge():
    """Return a fresh empty ``Memlet`` used as a control-dependency edge."""
    from dace.memlet import Memlet
    return Memlet()


def written_by_gpu_map_exit(sdfg: SDFG, name: str) -> bool:
    """``True`` iff ``name`` is written across a GPU-scheduled map's ``MapExit`` -- i.e. a kernel output.

    Single source of truth for the "is a kernel output" test: ``PromoteGPUScalarsToArrays`` uses it to
    promote a scalar, and ``DemoteKernelInternalArraysToScalars`` uses it (negated) to leave genuine
    kernel outputs alone.
    """
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
    """True iff stream-wiring already produced the ``gpu_streams`` array. Only the *wiring* step is
    single-shot; scheduling is persisted per node via ``Node.gpu_stream_id`` and survives
    serialisation. Used by :class:`GPUStreamWiring` to skip re-wiring.
    """
    return get_gpu_stream_array_name() in sdfg.arrays


def enclosing_map_chain(state: SDFGState, node: nodes.Node, schedule: dtypes.ScheduleType) -> List[nodes.MapEntry]:
    """Outermost-first chain of ``MapEntry`` nodes with ``schedule`` that enclose ``node`` (empty when none).

    Invalidates the state's ``scope_dict`` cache first: earlier pipeline passes can mutate topology
    in ways that leave the cache stale.
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
    """True iff ``sub_sdfg`` is (transitively) the body of a GPU_Device map.

    Walks ``parent_nsdfg_node`` / ``parent_sdfg`` directly, so the result is robust against stale
    ``scope_dict`` caches.
    """
    cur = sub_sdfg
    while cur.parent_nsdfg_node is not None:
        if innermost_enclosing_map(cur.parent, cur.parent_nsdfg_node, dtypes.ScheduleType.GPU_Device) is not None:
            return True
        cur = cur.parent_sdfg
    return False


def weakly_connected_node_sets(graph) -> List[Set[nodes.Node]]:
    """Weakly-connected components of ``graph``'s dataflow, as node sets.

    Single source of truth for the WCC partition used by both the stream scheduler and the
    state-splitter, via ``OrderedDiGraph.nx`` (tracks DaCe's graph internals rather than
    re-deriving connectivity)."""
    import networkx as nx
    return [set(c) for c in nx.weakly_connected_components(graph.nx)]


# Storages that mark a copy/memset library node as "GPU-relevant" (its expansion emits a
# cudaMemcpy / cudaMemset). Hoisted to module scope because it is consulted per node visited
# and rebuilding the set on every call shows up in profiles.
_GPU_COPY_STORAGES = frozenset(
    {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned})


def is_gpu_copy_or_memset_libnode(node, sdfg: SDFG, state: SDFGState) -> bool:
    """``CopyLibraryNode`` / ``MemsetLibraryNode`` whose storage involves GPU memory."""
    from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
    from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

    if isinstance(node, CopyLibraryNode):
        return (node.src_storage(state) in _GPU_COPY_STORAGES or node.dst_storage(state) in _GPU_COPY_STORAGES)
    if isinstance(node, MemsetLibraryNode):
        for e in state.out_edges(node):
            if e.data and e.data.data and sdfg.arrays[e.data.data].storage in _GPU_COPY_STORAGES:
                return True
    return False


def is_gpu_kernel_launcher(node) -> bool:
    """True for a ``GPU_Device`` kernel ``MapEntry`` -- the launcher binds the stream handle on enter."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """True for nodes that *take* a GPU stream: kernel ``MapEntry``, GPU Copy/Memset libnode, or a
    lowered runtime-call Tasklet.

    AccessNodes are excluded (memory references, not stream consumers); use
    :func:`is_gpu_relevant_node` for the broader "involves GPU work" question.
    """
    return (is_gpu_kernel_launcher(node) or is_gpu_copy_or_memset_libnode(node, sdfg, state)
            or is_already_lowered_gpu_runtime_call(node))


def is_already_lowered_gpu_runtime_call(node) -> bool:
    """True for a Tasklet that issues a stream-bound GPU runtime call.

    Detected either by a ``gpuStream_t`` in-connector (cuBLAS / cuSolver expansions that wire one)
    or by a :data:`STREAM_CONNECTOR` reference in the body (Copy/Memset libnode expansions, which
    carry no connector and rely on the scheduler binding it post-expansion). Pipeline-emitted sync
    tasklets are excluded -- they are not consumers in the WCC sense.
    """
    if not isinstance(node, nodes.Tasklet):
        return False
    if is_pipeline_sync_tasklet(node):
        return False
    if any(t == dtypes.gpuStream_t for t in node.in_connectors.values() if t is not None):
        return True
    return STREAM_CONNECTOR in node.code.as_string


SYNC_TASKLET_LABELS = ("gpu_streams_synchronization", "gpu_stream_synchronization")


def is_pipeline_sync_tasklet(node) -> bool:
    """True iff ``node`` is a sync tasklet emitted by the stream pipeline (identified by its canonical
    label). Excluded from consumer re-detection despite its ``gpuStream_t`` connector.
    """
    return isinstance(node, nodes.Tasklet) and node.label in SYNC_TASKLET_LABELS


def is_gpu_relevant_node(node, sdfg: SDFG, state: SDFGState) -> bool:
    """True for nodes implying the enclosing component/SDFG involves GPU work.

    The union of stream consumers and AccessNodes for ``GPU_Global`` arrays. Only stream consumers
    get a stream connector wired; AccessNodes have none to bind.
    """
    if is_gpu_stream_consumer(node, sdfg, state):
        return True
    if isinstance(node, nodes.AccessNode):
        return sdfg.arrays[node.data].storage == dtypes.StorageType.GPU_Global
    return False


def has_stream_connector(node) -> bool:
    """True if ``node`` carries any in-connector typed ``gpuStream_t``.

    Type-based, so it accepts whatever name the libnode expansion chose.
    """
    return any(t is not None and t == dtypes.gpuStream_t for t in node.in_connectors.values())


def add_gpu_stream_connector(node, conn_name: str, *, single_stream: bool):
    """Add a GPU-stream input connector with the right dtype.

    ``single_stream=True`` types it as a scalar ``gpuStream_t`` (consumer takes one stream value);
    ``False`` types it as ``pointer(gpuStream_t)`` (consumer receives the full ``gpu_streams`` array
    and indexes it by id).
    """
    dtype = dtypes.gpuStream_t if single_stream else dtypes.pointer(dtypes.gpuStream_t)
    node.add_in_connector(conn_name, dtype)


def find_inner_gpu_consumers(sdfg: SDFG):
    """Yield ``(node, sdfg, state)`` for every GPU stream consumer reachable inside ``sdfg``, recursing
    into nested SDFGs. Used by the stream-wiring passes to enumerate kernels and library nodes that
    need a stream bound.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if is_gpu_stream_consumer(node, nsdfg, state):
                    yield node, nsdfg, state
