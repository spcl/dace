from typing import Dict, Union
from dace import SDFG, nodes


class GPUStreamManager:
    """
    Manages GPU backend streams (e.g., CUDA or HIP streams) for nodes in an SDFG.
    Assumes that the initialization inputs come from the NaiveGPUScheduler pass.

    NOTE: "Stream" here refers to backend GPU streams, not DaCe data streams.
    """

    def __init__(self, sdfg: SDFG, assigned_streams: Dict[nodes.Node, Union[int, str]], stream_access_template: str):
        self.sdfg = sdfg
        self.assigned_streams = assigned_streams
        self.stream_access_template = stream_access_template

        # Placeholder for future support of backend events (e.g., CUDA events)
        self.num_gpu_events = 0

        # Determine the number of streams used (stream IDs start from 0)
        # Only count integer stream IDs (ignore string values like "nullptr")
        int_stream_ids = [v for v in assigned_streams.values() if isinstance(v, int)]
        self.num_gpu_streams = max(int_stream_ids, default=0) + 1

    def get_stream_node(self, node: nodes.Node) -> str:
        """
        Returns the GPU stream access expression for a given node.

        If the node has an assigned stream not equal the default "nullptr", returns
        the formatted stream expression. Otherwise, returns "nullptr".
        """
        if node in self.assigned_streams and self.assigned_streams[node] != "nullptr":
            return self.stream_access_template.format(gpu_stream=self.assigned_streams[node])
        return "nullptr"

    def get_stream_edge(self, src_node: nodes.Node, dst_node: nodes.Node) -> str:
        """
        Returns the stream access expression for an edge based on either the
        source or destination node. If one of the nodes has an assigned stream not equal
        to the default 'nullptr', that stream is returned (should be symmetric
        when using the NaiveGPUStreamScheduler pass). Otherwise, returns 'nullptr'.
        """
        if src_node in self.assigned_streams and self.assigned_streams[src_node] != "nullptr":
            stream_id = self.assigned_streams[src_node]
            return self.stream_access_template.format(gpu_stream=stream_id)
        elif dst_node in self.assigned_streams and self.assigned_streams[dst_node] != "nullptr":
            stream_id = self.assigned_streams[dst_node]
            return self.stream_access_template.format(gpu_stream=stream_id)
        else:
            return "nullptr"
