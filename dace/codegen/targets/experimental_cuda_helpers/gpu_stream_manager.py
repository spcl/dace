# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tracks GPU stream slots and maps stream-using nodes to their assigned ``gpuStream_t``."""
from typing import Dict
from dace import SDFG, nodes
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name


class GPUStreamManager:
    """
    Manage GPU backend streams (CUDA/HIP) for SDFG nodes.

    Given the per-node stream IDs assigned by ``NaiveGPUStreamScheduler``, provides their access
    expressions and the stream count. GPU events are not yet supported. "Stream" here means a
    backend GPU stream, not a DaCe data stream.
    """

    def __init__(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        self.sdfg = sdfg
        self._stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"
        self._assignments = assignments
        # Stream count comes from the ``gpu_streams`` descriptor shape (set by the GPU stream
        # scheduler via ``allocate_stream_array``), not from
        # ``max(assignments) + 1`` -- the latter is not invariant under pipeline re-application
        # (the scheduler's WCC walk is graph-shape-dependent and the pipeline mutates the graph).
        stream_array = get_gpu_stream_array_name()
        if stream_array in sdfg.arrays:
            self._num_gpu_streams = int(sdfg.arrays[stream_array].shape[0])
        else:
            self._num_gpu_streams = 0

    def get_stream_node(self, node: nodes.Node) -> str:
        """Return the access expression for the GPU stream assigned to ``node``,
        e.g. ``__state->gpu_context->streams[0]``. Reads ``node.gpu_stream_id``
        (persisted via the Node Property) so a deserialised SDFG round-trips
        without re-running the scheduler. Raises if the node was never assigned.
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
        """Always 0 -- events aren't wired through the new pipeline yet, but the
        codegen template still emits create/destroy loops over this count."""
        return 0

    @property
    def assignments(self) -> Dict[nodes.Node, int]:
        """Mapping of nodes to assigned GPU stream IDs (not all nodes necessarily have one)."""
        return self._assignments
