# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tracks GPU stream slots and maps stream-using nodes to their assigned ``gpuStream_t``."""
from dace import SDFG, nodes
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name


class GPUStreamManager:
    """Manage the backend GPU streams (CUDA/HIP, not DaCe data streams) of an SDFG's nodes,
    resolving assignments from ``Node.gpu_stream_id``."""

    def __init__(self, sdfg: SDFG):
        self.sdfg = sdfg
        self._stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"
        # The descriptor shape, not ``max(gpu_stream_id) + 1``: the latter is graph-shape
        # dependent and not invariant under pipeline re-application.
        stream_array = get_gpu_stream_array_name()
        if stream_array in sdfg.arrays:
            self._num_gpu_streams = int(sdfg.arrays[stream_array].shape[0])
        else:
            self._num_gpu_streams = 0

    def get_stream_node(self, node: nodes.Node) -> str:
        """The access expression for ``node``'s GPU stream, e.g. ``__state->gpu_context->streams[0]``.

        :raises ValueError: If the node was never assigned a stream.
        """
        if node.gpu_stream_id is not None:
            return self._stream_access_template.format(gpu_stream=node.gpu_stream_id)
        raise ValueError(f"No GPU stream assigned to node {node}. "
                         "Check whether the node is relevant for GPU stream assignment and, if it is, "
                         "inspect the GPU stream pipeline to see why no stream was assigned.")

    @property
    def num_gpu_streams(self) -> int:
        """Number of GPU streams in use (stream IDs start at 0)."""
        return self._num_gpu_streams

    @property
    def num_gpu_events(self) -> int:
        """Always 0: events are not wired through the new pipeline yet, but the codegen template
        still emits create/destroy loops over this count."""
        return 0
