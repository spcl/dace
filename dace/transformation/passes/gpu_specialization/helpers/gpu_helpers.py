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

# Name of the `stream` in-connector on CopyLibraryNode / MemsetLibraryNode.
# Kept in sync with the ``_STREAM_CONN`` constant in the library-node
# modules so the stream passes can add the connector without importing
# the private constant.
COPY_MEMSET_STREAM_CONNECTOR = "stream"


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
    return "__stream_"


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


def is_gpu_stream_consumer(node, sdfg: SDFG, state: SDFGState) -> bool:
    """Return True for nodes that *take* a GPU stream as an in-connector.

    The two classes are GPU_Device kernel ``MapEntry`` (the kernel launcher
    binds the stream handle on enter) and copy/memset library nodes touching
    GPU-side storage (their expansions wire ``stream`` to the runtime call).
    AccessNodes are *excluded* — they're memory references, not stream
    consumers. Use :func:`is_gpu_relevant_node` for the broader "does this
    state/component involve GPU work" question.
    """
    if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
        return True
    return is_gpu_copy_or_memset_libnode(node, sdfg, state)


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


def is_stream_connector(conn_name: str) -> bool:
    """Return True for the two GPU-stream connector naming conventions:
    the per-kernel-id ``__stream_<id>`` connectors added by
    ``ConnectGPUStreamsToNodes`` and the library-node ``stream`` connector
    used by ``CopyLibraryNode`` / ``MemsetLibraryNode``."""
    return conn_name == COPY_MEMSET_STREAM_CONNECTOR or conn_name.startswith(get_gpu_stream_connector_name())


def has_stream_connector(node) -> bool:
    """Return True if ``node`` already carries any GPU-stream input connector
    (``stream`` or ``__stream_<id>``). Use to skip nodes a prior pass wired."""
    return any(is_stream_connector(c) for c in node.in_connectors)


def add_gpu_stream_connector(node, conn_name: str, *, single_stream: bool):
    """Add a GPU-stream input connector with the right dtype.

    ``single_stream=True`` types it as a scalar ``gpuStream_t`` — the
    consumer takes one stream value (kernel maps, libnodes that bind one
    stream). ``False`` types it as ``pointer(gpuStream_t)`` — the consumer
    receives the full ``gpu_streams`` array and indexes it by id.
    """
    dtype = dtypes.gpuStream_t if single_stream else dtypes.pointer(dtypes.gpuStream_t)
    node.add_in_connector(conn_name, dtype)


def is_expanded_libnode_nsdfg(nsdfg_node) -> bool:
    """Return True if ``nsdfg_node`` is a NestedSDFG that inherited a single
    ``stream`` connector from a ``CopyLibraryNode`` / ``MemsetLibraryNode``
    expansion. Such NestedSDFGs already carry one bound stream and should
    reuse it for every internal GPU consumer instead of receiving a fresh
    ``gpu_streams`` array."""
    return COPY_MEMSET_STREAM_CONNECTOR in nsdfg_node.in_connectors


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
