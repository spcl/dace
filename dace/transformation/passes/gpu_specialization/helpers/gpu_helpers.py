# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared utilities for the GPU-specialization passes.

Canonical stream-threading names, node/connector predicates (single
source of truth so passes don't reimplement scope walks), and the
:func:`is_gpu_lowering_applied` idempotency signal.
"""
from typing import List, Optional

from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes

# Canonical GPU-stream in-connector name. Every stream consumer uses it.
# Detection is type-based (``gpuStream_t``-typed in-connector); this
# constant only keeps producers naming the connector consistently.
STREAM_CONNECTOR = "__stream"

# Back-compat alias. Pre-existing callers use this name; new code should
# use :data:`STREAM_CONNECTOR`.
COPY_MEMSET_STREAM_CONNECTOR = STREAM_CONNECTOR


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def dependency_edge():
    """Return a fresh empty ``Memlet`` used as a control-dependency edge (centralised for a
    single future migration point)."""
    from dace.memlet import Memlet
    return Memlet()


def is_gpu_lowering_applied(sdfg: SDFG) -> bool:
    """True iff the gpu_specialization lowering has already run on ``sdfg``, signalled by the
    ``gpu_streams`` transient. Used to short-circuit a re-application.
    """
    return get_gpu_stream_array_name() in sdfg.arrays


def enclosing_map_chain(state: SDFGState, node: nodes.Node, schedule: dtypes.ScheduleType) -> List[nodes.MapEntry]:
    """Outermost-first chain of ``MapEntry`` nodes with ``schedule`` that
    enclose ``node`` in ``state``. Empty when none. Invalidates the
    state's ``scope_dict`` cache first because earlier pipeline passes
    can mutate topology in ways that leave the cache stale."""
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

    Walks ``parent_nsdfg_node`` / ``parent_sdfg`` directly via
    :func:`innermost_enclosing_map`, so the result is robust against stale
    ``scope_dict`` caches.
    """
    cur = sub_sdfg
    while cur.parent_nsdfg_node is not None:
        if innermost_enclosing_map(cur.parent, cur.parent_nsdfg_node, dtypes.ScheduleType.GPU_Device) is not None:
            return True
        cur = cur.parent_sdfg
    return False


# Storages that mark a copy/memset library node as "GPU-relevant" -- i.e.
# its expansion wires the ``stream`` connector to a cudaMemcpy /
# cudaMemset runtime call. Hoisted to module scope because
# :func:`is_gpu_copy_or_memset_libnode` is called per node visited and
# rebuilding the set on every call shows up in profiles.
_GPU_COPY_STORAGES = frozenset(
    {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned})


def is_gpu_copy_or_memset_libnode(node, sdfg: SDFG, state: SDFGState) -> bool:
    """``CopyLibraryNode`` / ``MemsetLibraryNode`` whose storage involves GPU
    memory. These are the library nodes whose expansion wires the
    ``stream`` connector to the cudaMemcpy / cudaMemset runtime call."""
    from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
    from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

    if isinstance(node, CopyLibraryNode):
        return (node.src_storage(state, sdfg) in _GPU_COPY_STORAGES
                or node.dst_storage(state, sdfg) in _GPU_COPY_STORAGES)
    if isinstance(node, MemsetLibraryNode):
        for e in state.out_edges(node):
            if e.data and e.data.data and sdfg.arrays[e.data.data].storage in _GPU_COPY_STORAGES:
                return True
    return False


def is_gpu_kernel_launcher(node) -> bool:
    """``GPU_Device`` kernel ``MapEntry`` -- the launcher binds the stream
    handle via the ``__stream_<i>`` connector on enter."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """True for nodes that *take* a GPU stream: kernel ``MapEntry`` (:func:`is_gpu_kernel_launcher`),
    GPU Copy/Memset libnode (:func:`is_gpu_copy_or_memset_libnode`), or a lowered runtime-call
    Tasklet (:func:`is_already_lowered_gpu_runtime_call`).

    AccessNodes are excluded (memory references, not stream consumers); use
    :func:`is_gpu_relevant_node` for the broader "involves GPU work" question.
    """
    return (is_gpu_kernel_launcher(node) or is_gpu_copy_or_memset_libnode(node, sdfg, state)
            or is_already_lowered_gpu_runtime_call(node))


def is_already_lowered_gpu_runtime_call(node) -> bool:
    """True for a Tasklet that issues a stream-bound GPU runtime call (the post-libnode-expansion
    shape), detected by either a ``gpuStream_t`` in-connector or a ``__dace_current_stream``
    reference in its body.

    Pipeline-emitted sync tasklets (:func:`is_pipeline_sync_tasklet`) are excluded -- they aren't
    consumers in the WCC sense.
    """
    if not isinstance(node, nodes.Tasklet):
        return False
    if is_pipeline_sync_tasklet(node):
        return False
    if any(t == dtypes.gpuStream_t for t in node.in_connectors.values() if t is not None):
        return True
    code = node.code.as_string if hasattr(node.code, 'as_string') else str(node.code)
    return '__dace_current_stream' in code


SYNC_TASKLET_LABELS = ("gpu_streams_synchronization", "gpu_stream_synchronization")


def is_pipeline_sync_tasklet(node) -> bool:
    """True iff ``node`` is a sync tasklet emitted by the stream pipeline (identified by its
    canonical label). Excluded from consumer re-detection despite its ``gpuStream_t`` connector.
    """
    return isinstance(node, nodes.Tasklet) and node.label in SYNC_TASKLET_LABELS


def is_gpu_relevant_node(node, sdfg: SDFG, state: SDFGState) -> bool:
    """True for nodes implying the enclosing component/SDFG involves GPU work: the union of stream
    consumers (:func:`is_gpu_stream_consumer`) and AccessNodes for ``GPU_Global`` arrays. Only
    ``is_gpu_stream_consumer`` nodes get a stream connector wired; AccessNodes have none to bind.
    """
    if is_gpu_stream_consumer(node, sdfg, state):
        return True
    if isinstance(node, nodes.AccessNode):
        return sdfg.arrays[node.data].storage == dtypes.StorageType.GPU_Global
    return False


def is_stream_typed_connector(node, conn_name: str) -> bool:
    """True iff ``conn_name`` is an in-connector on ``node`` typed ``gpuStream_t``. The codebase
    uses one connector name (:data:`STREAM_CONNECTOR`) for all consumers, but detection is
    type-based -- the type is the contract.
    """
    t = node.in_connectors.get(conn_name)
    return t is not None and t == dtypes.gpuStream_t


def has_stream_connector(node) -> bool:
    """Return True if ``node`` already carries any GPU-stream in-connector
    -- i.e. any in-connector typed ``gpuStream_t``. Type-based, so it
    accepts whatever name the libnode expansion chose."""
    return any(t is not None and t == dtypes.gpuStream_t for t in node.in_connectors.values())


def add_gpu_stream_connector(node, conn_name: str, *, single_stream: bool):
    """Add a GPU-stream input connector with the right dtype.

    ``single_stream=True`` types it as a scalar ``gpuStream_t`` -- the
    consumer takes one stream value (kernel maps, libnodes that bind one
    stream). ``False`` types it as ``pointer(gpuStream_t)`` -- the consumer
    receives the full ``gpu_streams`` array and indexes it by id.
    """
    dtype = dtypes.gpuStream_t if single_stream else dtypes.pointer(dtypes.gpuStream_t)
    node.add_in_connector(conn_name, dtype)


def find_inner_gpu_consumers(sdfg: SDFG):
    """Yield ``(node, sdfg, state)`` for every GPU stream consumer reachable
    inside ``sdfg`` (recursively). Used by the stream-wiring passes to
    enumerate kernels and library nodes that need a stream bound.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if is_gpu_stream_consumer(node, nsdfg, state):
                    yield node, nsdfg, state


def read_stream_assignments_from_wired_sdfg(sdfg: SDFG):
    """Recover ``{node: stream_id}`` from a post-pipeline SDFG by reading the ``gpu_streams[<i>]``
    subset wired into each consumer's stream in-connector. Re-running the scheduler instead would
    differ because pipeline-internal nodes stitch otherwise-independent components together.
    Returns ``{}`` if the lowering hasn't run yet.
    """
    if not is_gpu_lowering_applied(sdfg):
        return {}
    stream_array = get_gpu_stream_array_name()
    assignments = {}
    for node, parent_sdfg, state in find_inner_gpu_consumers(sdfg):
        for edge in state.in_edges(node):
            if not edge.dst_conn or not is_stream_typed_connector(node, edge.dst_conn):
                continue
            if edge.data is None or edge.data.data != stream_array or edge.data.subset is None:
                continue
            # The wired memlet is ``gpu_streams[<i>]`` -- a single-element
            # ``Range`` whose start equals its end. Read the start.
            try:
                stream_id = int(edge.data.subset[0][0])
            except (TypeError, ValueError, IndexError):
                continue
            assignments[node] = stream_id
            break
    return assignments
