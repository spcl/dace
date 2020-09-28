# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" State elimination transformations """

import networkx as nx

from dace import dtypes, registry, sdfg
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.config import Config


@registry.autoregister_params(strict=True)
class EndStateElimination(transformation.Transformation):
    """ 
    End-state elimination removes a redundant state that has one incoming edge
    and no contents.
    """

    _end_state = sdfg.SDFGState()

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(EndStateElimination._end_state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        state = graph.nodes()[candidate[EndStateElimination._end_state]]

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # If this is an end state, there are no outgoing edges
        if len(out_edges) != 0:
            return False

        # We only match end states with one source and no conditions
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]
        if not edge.data.is_unconditional():
            return False

        # Only empty states can be eliminated
        if state.number_of_nodes() > 0:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[EndStateElimination._end_state]]
        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[EndStateElimination._end_state]]
        sdfg.remove_node(state)


@registry.autoregister
class StateAssignElimination(transformation.Transformation):
    """ 
    State assign elimination removes all assignments into the final state
    and subsumes the assigned value into its contents.
    """

    _end_state = sdfg.SDFGState()

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(StateAssignElimination._end_state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        state = graph.nodes()[candidate[StateAssignElimination._end_state]]

        out_edges = graph.out_edges(state)
        in_edges = graph.in_edges(state)

        # If this is an end state, there are no outgoing edges
        if len(out_edges) != 0:
            return False

        # We only match end states with one source and at least one assignment
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]
        if len(edge.data.assignments) == 0:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[StateAssignElimination._end_state]]
        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[StateAssignElimination._end_state]]
        edge = sdfg.in_edges(state)[0]
        # Since inter-state assignments that use an assigned value leads to
        # undefined behavior (e.g., {m: n, n: m}), we can replace each
        # assignment separately.
        for varname, assignment in edge.data.assignments.items():
            state.replace(varname, assignment)
        # Remove assignments from edge
        edge.data.assignments = {}