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
    tasklet = transformation.PatternNode(nodes.Tasklet)
    write = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(TrivialTaskletElimination.read, TrivialTaskletElimination.tasklet,
                                   TrivialTaskletElimination.write)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        read = graph.nodes()[candidate[TrivialTaskletElimination.read]]
        tasklet = graph.nodes()[candidate[TrivialTaskletElimination.tasklet]]
        write = graph.nodes()[candidate[TrivialTaskletElimination.write]]
        # Do not apply on Streams
        if isinstance(sdfg.arrays[read.data], data.Stream):
            return False
        if isinstance(sdfg.arrays[write.data], data.Stream):
            return False
        if len(graph.in_edges(tasklet)) != 1:
            return False
        if len(graph.out_edges(tasklet)) != 1:
            return False
        if graph.edges_between(tasklet, write)[0].data.wcr:
            return False
        if len(tasklet.in_connectors) != 1:
            return False
        if len(tasklet.out_connectors) != 1:
            return False
        in_conn = list(tasklet.in_connectors.keys())[0]
        out_conn = list(tasklet.out_connectors.keys())[0]
        if tasklet.code.as_string != f'{out_conn} = {in_conn}':
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = graph.nodes()[candidate[TrivialTaskletElimination.tasklet]]
        return tasklet.label

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        read = graph.nodes()[self.subgraph[TrivialTaskletElimination.read]]
        tasklet = graph.nodes()[self.subgraph[TrivialTaskletElimination.tasklet]]
        write = graph.nodes()[self.subgraph[TrivialTaskletElimination.write]]

        in_edge = graph.edges_between(read, tasklet)[0]
        out_edge = graph.edges_between(tasklet, write)[0]
        graph.remove_edge(in_edge)
        graph.remove_edge(out_edge)
        out_edge.data.other_subset = in_edge.data.subset
        graph.add_nedge(read, write, out_edge.data)
        graph.remove_node(tasklet)
