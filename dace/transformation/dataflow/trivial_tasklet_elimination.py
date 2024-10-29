# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial-tasklet-elimination transformation. """

from dace import data, registry
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class TrivialTaskletElimination(transformation.SingleStateTransformation):
    """ Implements the Trivial-Tasklet Elimination pattern.

        Trivial-Tasklet Elimination removes tasklets that just copy the input
        to the output without WCR.
    """

    read = transformation.PatternNode(nodes.AccessNode)
    read_map = transformation.PatternNode(nodes.MapEntry)
    tasklet = transformation.PatternNode(nodes.Tasklet)
    write = transformation.PatternNode(nodes.AccessNode)
    write_map = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.read, cls.tasklet, cls.write),
            sdutil.node_path_graph(cls.read_map, cls.tasklet, cls.write),
            sdutil.node_path_graph(cls.read, cls.tasklet, cls.write_map),
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        read = self.read_map if expr_index == 1 else self.read
        tasklet = self.tasklet
        write = self.write_map if expr_index == 2 else self.write
        if len(tasklet.in_connectors) != 1:
            return False
        if len(graph.in_edges(tasklet)) != 1:
            return False
        if len(tasklet.out_connectors) != 1:
            return False
        if len(graph.out_edges(tasklet)) != 1:
            return False
        in_conn = list(tasklet.in_connectors.keys())[0]
        out_conn = list(tasklet.out_connectors.keys())[0]
        if tasklet.code.as_string != f'{out_conn} = {in_conn}':
            return False
        read_memlet = graph.edges_between(read, tasklet)[0].data
        read_desc = sdfg.arrays[read_memlet.data]
        write_memlet = graph.edges_between(tasklet, write)[0].data
        if write_memlet.wcr:
            return False
        write_desc = sdfg.arrays[write_memlet.data]
        # Do not apply on streams
        if isinstance(read_desc, data.Stream):
            return False
        if isinstance(write_desc, data.Stream):
            return False
        # Keep copy-tasklet connected to map node if source and destination nodes
        # have different data type (implicit type cast)
        if expr_index != 0 and read_desc.dtype != write_desc.dtype:
            return False
    
        return True

    def apply(self, graph, sdfg):
        read = self.read_map if self.expr_index == 1 else self.read
        tasklet = self.tasklet
        write = self.write_map if self.expr_index == 2 else self.write

        in_edge = graph.edges_between(read, tasklet)[0]
        out_edge = graph.edges_between(tasklet, write)[0]
        graph.remove_edge(in_edge)
        graph.remove_edge(out_edge)
        out_edge.data.other_subset = in_edge.data.subset
        graph.add_edge(read, in_edge.src_conn, write, out_edge.dst_conn, out_edge.data)
        graph.remove_node(tasklet)
