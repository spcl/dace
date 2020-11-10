# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" State elimination transformations """

import copy
import networkx as nx

from dace import dtypes, registry, sdfg, symbolic
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


def _assignments_to_consider(sdfg, edge):
    assignments_to_consider = {}
    for var, assign in edge.data.assignments.items():
        as_symbolic = symbolic.pystr_to_symbolic(assign)
        # Assignments cannot access a data container
        if not symbolic.contains_sympy_functions(as_symbolic):  # via subscript
            # Assignments cannot use scalar values
            for sym in as_symbolic.free_symbols:
                if str(sym) in sdfg.arrays:
                    break
            else:
                assignments_to_consider[var] = assign
    return assignments_to_consider


@registry.autoregister_params(strict=True)
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

        # We only match end states with one source and at least one assignment
        if len(in_edges) != 1:
            return False
        edge = in_edges[0]

        assignments_to_consider = _assignments_to_consider(sdfg, edge)

        # No assignments to eliminate
        if len(assignments_to_consider) == 0:
            return False

        # If this is an end state, there are no other edges to consider
        if len(out_edges) == 0:
            return True

        # Otherwise, ensure the symbols are never set/used again in edges
        akeys = set(assignments_to_consider.keys())
        for e in sdfg.edges():
            if e is edge:
                continue
            if e.data.free_symbols & akeys:
                return False

        # If used in any state that is not the current one, fail
        for s in sdfg.nodes():
            if s is state:
                continue
            if s.free_symbols & akeys:
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
        keys_to_remove = set()
        assignments_to_consider = _assignments_to_consider(sdfg, edge)
        for varname, assignment in assignments_to_consider.items():
            state.replace(varname, assignment)
            keys_to_remove.add(varname)

        for varname in keys_to_remove:
            # Remove assignments from edge
            del edge.data.assignments[varname]

            for e in sdfg.edges():
                if varname in e.data.free_symbols:
                    break
            else:
                # If removed assignment does not appear in any other edge,
                # remove symbol
                if varname in sdfg.symbols:
                    sdfg.remove_symbol(varname)
