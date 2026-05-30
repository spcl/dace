# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tracks GPU stream slots and maps stream-using nodes to their assigned ``gpuStream_t``."""
from typing import Dict
from dace import SDFG, nodes
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (get_gpu_event_array_name,
                                                                               get_gpu_stream_array_name)


class GPUStreamManager:
    """
    Manage GPU backend streams (CUDA/HIP) for SDFG nodes.

    Given the per-node stream IDs assigned by the active scheduling strategy, provides their
    access expressions and the stream / event counts. "Stream" here means a backend GPU stream,
    not a DaCe data stream.
    """

    def __init__(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        self.sdfg = sdfg
        self._stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"
        self._assignments = assignments
        # Stream / event counts come from the descriptors' shapes (set by the scheduling
        # strategy via ``allocate_stream_array`` / ``allocate_event_array``), not from
        # ``max(assignments) + 1`` -- the latter is not invariant under pipeline re-application
        # (the WCC walk is graph-shape-dependent and the pipeline mutates the graph).
        stream_array = get_gpu_stream_array_name()
        if stream_array in sdfg.arrays:
            self._num_gpu_streams = int(sdfg.arrays[stream_array].shape[0])
        else:
            self._num_gpu_streams = 0

        event_array = get_gpu_event_array_name()
        if event_array in sdfg.arrays:
            self._num_gpu_events = int(sdfg.arrays[event_array].shape[0])
        else:
            self._num_gpu_events = 0

    def get_stream_node(self, node: nodes.Node) -> str:
        """Return the access expression for the GPU stream assigned to ``node``,
        e.g. ``__state->gpu_context->streams[0]``. Raises if the node is not
        in the scheduler's assignment map."""
        if node in self._assignments:
            return self._stream_access_template.format(gpu_stream=self._assignments[node])
        raise ValueError(f"No GPU stream assigned to node {node}. "
                         "Check whether the node is relevant for GPU stream assignment and, if it is, "
                         "inspect the GPU stream pipeline to see why no stream was assigned.")

    @property
    def num_gpu_streams(self) -> int:
        """Number of GPU streams in use (stream IDs start at 0)."""
        return self._num_gpu_streams

    @property
    def num_gpu_events(self) -> int:
        """Number of GPU events in use (event IDs start at 0). Sourced from the IR-level
        ``gpu_events`` descriptor shape (allocated by :func:`allocate_event_array`)."""
        return self._num_gpu_events

    @property
    def assignments(self) -> Dict[nodes.Node, int]:
        """Mapping of nodes to assigned GPU stream IDs (not all nodes necessarily have one)."""
        return self._assignments
