# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared utilities for the GPU-specialization passes.

Three groups of helpers:

* **Names**: the canonical strings the pipeline uses when threading GPU
  streams through an SDFG — the array name, the per-kernel connector
  prefix, the legacy runtime placeholder symbol, and the connector name
  added to ``CopyLibraryNode`` / ``MemsetLibraryNode`` by stream wiring.

* **Predicates**: "this node needs a stream wired to it",
  "this connector name is a stream connector", "this NestedSDFG executes
  as device code". Single source of truth so passes don't reimplement
  scope walks.

* **State signal**: ``is_gpu_lowering_applied`` for pipeline idempotency.
"""
from typing import List, Optional

from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes

# Canonical name of the GPU-stream in-connector across the codebase.
# Every stream consumer (kernel ``MapEntry``, Copy/Memset libnode,
# pre-expanded runtime Tasklet, sync tasklet, Sequential-map
# pass-through) uses this single name. Detection is type-based
# (``gpuStream_t``-typed in-connector); the constant exists so
# producers can name the connector consistently when they create one.
STREAM_CONNECTOR = "__stream"

# Back-compat alias. Pre-existing callers use this name; new code should
# use :data:`STREAM_CONNECTOR`.
COPY_MEMSET_STREAM_CONNECTOR = STREAM_CONNECTOR


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def dependency_edge():
    """Return a fresh empty ``Memlet`` used as a control-dependency edge.

    Multiple call sites across the gpu_specialization package use a bare
    ``dace.Memlet()`` to mean "this edge carries no data, only an
    ordering dependency". Centralising the construction gives us a
    single place to migrate to a typed dependency-edge primitive when
    the framework adds one.
    """
    from dace.memlet import Memlet
    return Memlet()


def is_gpu_lowering_applied(sdfg: SDFG) -> bool:
    """True iff the gpu_specialization lowering has already run on ``sdfg``.

    The signal is the presence of the ``gpu_streams`` transient on the top
    SDFG — created by ``InsertGPUStreams``, the only pipeline pass that
    introduces it, and stable across the rest of the pipeline. Used by
    ``GPUSpecializationPipeline`` to short-circuit a re-application.
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


def get_gpu_stream_connector_name() -> str:
    """Deprecated. Use :data:`STREAM_CONNECTOR` directly. Returned name is
    the single canonical connector name (no per-id suffix anymore)."""
    return STREAM_CONNECTOR


# Storages that mark a copy/memset library node as "GPU-relevant" — i.e.
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
    """``GPU_Device`` kernel ``MapEntry`` — the launcher binds the stream
    handle via the ``__stream_<i>`` connector on enter."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """Return True for nodes that *take* a GPU stream.

    Composed of three named predicates, one per class:

    * :func:`is_gpu_kernel_launcher` — kernel ``MapEntry``.
    * :func:`is_gpu_copy_or_memset_libnode` — Copy/Memset libnode
      touching GPU-side storage.
    * :func:`is_already_lowered_gpu_runtime_call` — Tasklet that takes a
      ``gpuStream_t`` connector or references the legacy
      ``__dace_current_stream`` placeholder.

    AccessNodes are *excluded* — they're memory references, not stream
    consumers. Use :func:`is_gpu_relevant_node` for the broader "does
    this state/component involve GPU work" question.
    """
    return (is_gpu_kernel_launcher(node) or is_gpu_copy_or_memset_libnode(node, sdfg, state)
            or is_already_lowered_gpu_runtime_call(node))


def is_already_lowered_gpu_runtime_call(node) -> bool:
    """True for a Tasklet that issues a stream-bound GPU runtime call —
    the canonical post-libnode-expansion shape.

    Two structural signals (whichever fires first):

    1. The Tasklet has an in-connector typed ``gpuStream_t`` — a libnode
       expansion with a wired stream input produces this shape.
    2. The Tasklet's body references the legacy ``__dace_current_stream``
       placeholder — a libnode expanded *before* stream wiring relies on
       this symbol; the codegen prelude / wrapper Scalar binds it.

    Pipeline-emitted sync tasklets (:func:`is_pipeline_sync_tasklet`)
    are excluded — they were added by the pipeline itself and carry a
    ``__stream_<i>`` connector that doesn't make them consumers in the
    WCC sense.
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
    """True iff ``node`` is a sync tasklet emitted by the stream pipeline.

    The pipeline's own sync tasklets carry a ``gpuStream_t`` in-connector
    (so they can call ``cudaStreamSynchronize``) — they're not stream
    consumers in the WCC sense and must be excluded from re-detection.
    Identified by their canonical labels.
    """
    return isinstance(node, nodes.Tasklet) and node.label in SYNC_TASKLET_LABELS


def is_gpu_relevant_node(node, sdfg: SDFG, state: SDFGState) -> bool:
    """Return True for nodes whose presence implies the enclosing component
    or SDFG involves GPU work — i.e. the union of stream consumers
    (:func:`is_gpu_stream_consumer`) and AccessNodes for GPU_Global arrays.

    Used by the stream scheduler's component-level check and by
    ``InsertGPUStreams``' "does this child SDFG need the stream array?"
    sweep. Wire stream connectors only to ``is_gpu_stream_consumer`` —
    AccessNodes don't have a stream connector to bind.
    """
    if is_gpu_stream_consumer(node, sdfg, state):
        return True
    if isinstance(node, nodes.AccessNode):
        return sdfg.arrays[node.data].storage == dtypes.StorageType.GPU_Global
    return False


def is_stream_typed_connector(node, conn_name: str) -> bool:
    """True iff ``conn_name`` is an in-connector on ``node`` typed
    ``gpuStream_t`` — the type IS the contract.

    The codebase uses a single connector name (:data:`STREAM_CONNECTOR`,
    ``"__stream"``) for stream inputs across all consumer classes
    (kernels, Copy/Memset libnodes, runtime tasklets, sync tasklets,
    Sequential-map pass-through). Detection is always type-based.
    """
    t = node.in_connectors.get(conn_name)
    return t is not None and t == dtypes.gpuStream_t


def has_stream_connector(node) -> bool:
    """Return True if ``node`` already carries any GPU-stream in-connector
    — i.e. any in-connector typed ``gpuStream_t``. Type-based, so it
    accepts whatever name the libnode expansion chose."""
    return any(t is not None and t == dtypes.gpuStream_t for t in node.in_connectors.values())


def add_gpu_stream_connector(node, conn_name: str, *, single_stream: bool):
    """Add a GPU-stream input connector with the right dtype.

    ``single_stream=True`` types it as a scalar ``gpuStream_t`` — the
    consumer takes one stream value (kernel maps, libnodes that bind one
    stream). ``False`` types it as ``pointer(gpuStream_t)`` — the consumer
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
    """Recover ``{node: stream_id}`` from a post-pipeline SDFG by reading the
    ``gpu_streams[<i>]`` subset wired into each consumer's stream
    in-connector. The pipeline already encoded the assignments into the
    graph — re-running the scheduler on the post-pipeline graph would give
    different results because pipeline-internal AccessNodes / sync
    tasklets stitch otherwise-independent components together. Returns
    ``{}`` if the lowering hasn't run yet.
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
            # The wired memlet is ``gpu_streams[<i>]`` — a single-element
            # ``Range`` whose start equals its end. Read the start.
            try:
                stream_id = int(edge.data.subset[0][0])
            except (TypeError, ValueError, IndexError):
                continue
            assignments[node] = stream_id
            break
    return assignments


