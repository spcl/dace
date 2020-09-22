# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from dace import registry
from dace.data import Array, Stream
from dace.sdfg import nodes, utils as sdutil
from dace.transformation import helpers, pattern_matching


@registry.autoregister_params(singlestate=True)
class PipelineFusion(pattern_matching.Transformation):
    """ For a transient array that splits a dataflow graph into two
        disconnected subgraphs, PipelineFusion replaces the array with a stream
        that is produced by the writing component, and consumed by the reading
        component.
        This is useful for SDFGs targeting FPGAs, where the two components will
        be scheduled as asynchronous, pipeline parallel components, scheduled in
        parallel.
    """
    _node_before = nodes.Node()
    _array_node = nodes.AccessNode("_")
    _node_after = nodes.Node()

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                PipelineFusion._node_before,
                PipelineFusion._array_node,
                PipelineFusion._node_after,
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):

        array_node = graph.nodes()[candidate[PipelineFusion._array_node]]
        array_desc = sdfg.data(array_node.data)
        if (not isinstance(array_desc, Array) or not array_desc.transient):
            return False

        scope_dict = graph.scope_dict(array_node)
        if array_node not in scope_dict[None]:
            return False  # Must be in outermost scope

        if len([n for n in graph.nodes() if isinstance(n, nodes.AccessNode)
               and n.data == array_node.data]) > 1:
            return False  # Can only occur once

        node_before = graph.nodes()[candidate[PipelineFusion._node_before]]
        node_after = graph.nodes()[candidate[PipelineFusion._node_after]]

        # Don't allow any other edges going into/out of the array node
        for e in graph.in_edges(array_node):
            if e.src != node_before:
                return False
        for e in graph.out_edges(array_node):
            if e.dst != node_after:
                return False

        if not helpers.is_cut_vertex(graph, array_node):
            return False  # Node must cut the graph into two or more components

        # Success
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        array_node = graph.nodes()[candidate[PipelineFusion._array_node]]
        node_before = graph.nodes()[candidate[PipelineFusion._node_before]]
        node_after = graph.nodes()[candidate[PipelineFusion._node_after]]
        return f"{node_before} -> {array_node} -> {node_after}"

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        array_node = graph.nodes()[self.subgraph[PipelineFusion._array_node]]
        array_desc = sdfg.data(array_node.data)
        node_before = graph.nodes()[self.subgraph[PipelineFusion._node_before]]
        node_after = graph.nodes()[self.subgraph[PipelineFusion._node_after]]

        edges_before = graph.in_edges(array_node)
        edges_after = graph.out_edges(array_node)

        graph.remove_node(array_node)
        sdfg.remove_data(array_node.data)

        # Create the stream transient that will replace the array
        stream_desc = Stream(array_desc.dtype,
                             0,
                             storage=array_desc.storage,
                             transient=True,
                             location=array_desc.location,
                             lifetime=array_desc.lifetime)
        sdfg.add_datadesc(array_node.data, stream_desc)

        # Instantiate producer and consumer
        stream_produce = graph.add_write(array_node.data)
        stream_consume = graph.add_read(array_node.data)

        # Move edges to point to stream nodes instead
        for src, src_conn, dst, dst_conn, memlet in edges_before:
            graph.add_edge(src, src_conn, stream_produce, dst_conn, memlet)
        for src, src_conn, dst, dst_conn, memlet in edges_after:
            graph.add_edge(stream_consume, src_conn, dst, dst_conn, memlet)
