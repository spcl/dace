# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Small shared helpers for the experimental CUDA codegen (block-size math, schedule checks)."""

from typing import Iterator, Tuple

from dace import Config
from dace.codegen import common
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name
from dace.sdfg.state import SDFGState


def host_read_device_copies(state: SDFGState, consumer: nodes.Node) -> Iterator[Tuple[nodes.AccessNode, nodes.Node]]:
    """Yield ``(access node, producer)`` per host value ``consumer`` reads that a GPU copy wrote.

    A producer bound to a GPU stream writing host-visible memory is a device-to-host copy: its
    completion is only ordered by the stream, so the host has to synchronize before reading the
    destination. Host-to-device copies write GPU memory and are filtered out here.
    """
    for edge in state.in_edges(consumer):
        if edge.data is None or edge.data.data is None:
            continue
        source = state.memlet_path(edge)[0].src
        if not isinstance(source, nodes.AccessNode):
            continue
        if state.sdfg.arrays[source.data].storage in GPU_RESIDENT_STORAGES:
            continue
        for producer_edge in state.in_edges(source):
            producer = producer_edge.src
            if producer.gpu_stream_id is not None:
                yield source, producer


def generate_sync_debug_call() -> str:
    """Return backend sync + error-check calls when ``compiler.cuda.syncdebug`` is set, else empty string."""
    if not Config.get_bool('compiler', 'cuda', 'syncdebug'):
        return ""
    backend: str = common.get_gpu_backend()
    return (f"DACE_GPU_CHECK({backend}GetLastError());\n"
            f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")


def assigned_stream_expr(node: nodes.Node) -> str:
    """The expression of the GPU stream the stream pipeline assigned to ``node``.

    :raises ValueError: If the node was never assigned a stream.
    """
    if node.gpu_stream_id is None:
        raise ValueError(f"No GPU stream assigned to node {node}. Check whether the node is relevant for GPU "
                         "stream assignment and, if it is, why the GPU stream pipeline assigned none.")
    return common.gpu_stream_expr(node.gpu_stream_id)


def num_gpu_streams(sdfg: SDFG) -> int:
    """The length of the ``gpu_streams`` array: the descriptor, not the largest stream id, which depends on
    the graph shape and changes when the pipeline is re-applied."""
    stream_array = get_gpu_stream_array_name()
    return int(sdfg.arrays[stream_array].shape[0]) if stream_array in sdfg.arrays else 0
