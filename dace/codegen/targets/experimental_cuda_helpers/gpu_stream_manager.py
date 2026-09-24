# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tracks GPU stream slots and maps stream-using nodes to their assigned ``gpuStream_t``."""
from dace import SDFG, nodes
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name


class GPUStreamManager:
    """Resolve backend GPU streams (CUDA/HIP, not DaCe data streams) from ``Node.gpu_stream_id``."""

    def __init__(self, sdfg: SDFG):
        # Stream count = descriptor shape, not ``max(gpu_stream_id) + 1``, which is not invariant
        # under pipeline re-application.
        stream_array = get_gpu_stream_array_name()
        if stream_array in sdfg.arrays:
            self._num_gpu_streams = int(sdfg.arrays[stream_array].shape[0])
        else:
            self._num_gpu_streams = 0

    def get_stream_node(self, node: nodes.Node) -> str:
        """Access expression for the stream assigned to ``node``, e.g. ``__state->gpu_context->streams[0]``.

        Raises if the node was never assigned.
        """
        if node.gpu_stream_id is not None:
            return f"__state->gpu_context->streams[{node.gpu_stream_id}]"
        raise ValueError(f"No GPU stream assigned to node {node}. "
                         "Check whether the node is relevant for GPU stream assignment and, if it is, "
                         "inspect the GPU stream pipeline to see why no stream was assigned.")

    @property
    def num_gpu_streams(self) -> int:
        """Number of GPU streams in use (stream IDs start at 0)."""
        return self._num_gpu_streams
