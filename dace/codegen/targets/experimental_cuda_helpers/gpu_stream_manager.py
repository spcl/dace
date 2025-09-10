# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, Union
from dace import SDFG, nodes


class GPUStreamManager:
    """
    Manage GPU backend streams (e.g., CUDA or HIP) for nodes in an SDFG.

    Nodes are assigned stream IDs by the NaiveGPUStreamScheduler Pass, and 
    this class provides their access expressions and tracks the number of streams 
    in use. GPU events are not (yet) supported.

    Note
    ----
    "Stream" refers to backend GPU streams, not DaCe data streams.
    """

    def __init__(self, sdfg: SDFG, gpustream_assignments: Dict[nodes.Node, int]):
        self.sdfg = sdfg
        self._stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"
        self._gpustream_assignments = gpustream_assignments
        self._num_gpu_streams = max(gpustream_assignments.values()) + 1 if gpustream_assignments else 0
        self._num_gpu_events = 0
        


    def get_stream_node(self, node: nodes.Node) -> str:
        """
        Return the access expression for the GPU stream assigned to a node.

        Parameters
        ----------
        node : nodes.Node
            The node for which to return the access expression of its assigned CUDA stream.

        Returns
        -------
        str
            The GPU stream access expression, e.g.,
            "__state->gpu_context->streams[0]".

        Raises
        ------
        ValueError
            If the given node does not have an assigned stream.
        """
        if node in self.gpustream_assignments:
            return self._stream_access_template.format(
                gpu_stream=self.gpustream_assignments[node]
            )
        else:
            raise ValueError(
                f"No GPU stream assigned to node {node}. "
                "Check whether the node is relevant for GPU stream assignment and, if it is, "
                "inspect the GPU stream pipeline to see why no stream was assigned."
            )
        
    def get_stream_edge(self, src_node: nodes.Node, dst_node: nodes.Node) -> str:
        """
        Returns the GPU stream access expression for an edge.

        Currently unused: edge-level streams were only needed for asynchronous
        memory-copy operations (e.g., cudaMemcpyAsync). These copies are now
        modeled via tasklets in the SDFG, so edges do not carry stream info.
        Implement this if the design changes and edges need streams again.
        """
        raise NotImplementedError(
            "Edge-level GPU streams are not supported. "
            "They were previously used for asynchronous memory copies (e.g., cudaMemcpyAsync), "
            "but these are now modeled via tasklets in the SDFG. "
            "Implement this if the design changes and edges must carry GPU stream information."
        )

    @property
    def num_gpu_events(self) -> int:
        """Number of GPU events (currently always 0, left here for potential future support)."""
        return 0

    @property
    def num_gpu_streams(self) -> int:
        """Number of GPU streams in use (stream IDs start at 0)."""
        return self._num_gpu_streams
    
    @property
    def gpustream_assignments(self) -> Dict[nodes.Node, int]:
        """Mapping of nodes to assigned GPU stream IDs (not all nodes necessarily have a GPU stream ID)."""
        return self._gpustream_assignments
    