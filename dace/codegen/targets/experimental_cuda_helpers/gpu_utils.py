# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Small shared helpers for the experimental CUDA codegen (block-size math, schedule checks)."""

from typing import Iterator, Tuple

from dace import Config
from dace.codegen import common
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState

# CUDA/HIP launch grids and blocks always have three dimensions (x, y, z).
CUDA_GRID_DIMS = 3


def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx >= CUDA_GRID_DIMS:
        raise ValueError(f'idx must be in 0..{CUDA_GRID_DIMS - 1}, got {idx}')
    return ('x', 'y', 'z')[idx]


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
