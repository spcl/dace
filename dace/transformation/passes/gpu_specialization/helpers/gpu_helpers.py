# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared naming / code-snippet helpers for the GPU-specialization passes.

Provides the canonical names used when threading per-kernel GPU stream
handles through an SDFG -- the stream array, the per-node connector
prefix, the legacy runtime placeholder symbol (``__dace_current_stream``)
recognized by the DaCe codegen, and the connector name added to
``CopyLibraryNode`` / ``MemsetLibraryNode`` by the stream-connection
pass.

Also exports a single source of truth for the predicates "this node needs
a GPU stream wired to it" and "this connector name is a stream
connector". Both were independently reimplemented in four passes before
this module centralized them.
"""
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes

# Storages whose copies are serviced by the GPU stream pipeline.
_GPU_SIDE_STORAGES = frozenset({
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
    dtypes.StorageType.CPU_Pinned,
})

# Name of the `stream` in-connector on CopyLibraryNode / MemsetLibraryNode.
# Kept in sync with the ``_STREAM_CONN`` constant in the library-node
# modules so the stream passes can add the connector without importing
# the private constant.
COPY_MEMSET_STREAM_CONNECTOR = "stream"


def get_gpu_stream_array_name() -> str:
    return "gpu_streams"


def get_gpu_stream_connector_name() -> str:
    return "__stream_"


def get_dace_runtime_gpu_stream_name() -> str:
    return "__dace_current_stream"


def get_default_gpu_stream_name() -> str:
    return "__default_stream"


def is_gpu_copy_or_memset_libnode(node, sdfg: SDFG, state: SDFGState) -> bool:
    """``CopyLibraryNode`` / ``MemsetLibraryNode`` whose storage involves GPU
    memory. These are the library nodes whose expansion wires the
    ``stream`` connector to the cudaMemcpy / cudaMemset runtime call."""
    from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
    from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

    if isinstance(node, CopyLibraryNode):
        return (node.src_storage(state, sdfg) in _GPU_SIDE_STORAGES
                or node.dst_storage(state, sdfg) in _GPU_SIDE_STORAGES)
    if isinstance(node, MemsetLibraryNode):
        for e in state.out_edges(node):
            if e.data and e.data.data and sdfg.arrays[e.data.data].storage in _GPU_SIDE_STORAGES:
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
