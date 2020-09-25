# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

import networkx as nx

from dace import dtypes, registry, sdfg
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.config import Config


@registry.autoregister_params(strict=True)
class StateFusion(transformation.Transformation):
    """ Implements the state-fusion transformation.
        
        State-fusion takes two states that are connected through a single edge,
        and fuses them into one state. If strict, only applies if no memory 
        access hazards are created.
    """

    _states_fused = 0
    _first_state = sdfg.SDFGState()
    _edge = sdfg.InterstateEdge()
    _second_state = sdfg.SDFGState()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(StateFusion._first_state,
                                   StateFusion._second_state)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_state = graph.nodes()[candidate[StateFusion._first_state]]
        second_state = graph.nodes()[candidate[StateFusion._second_state]]

        out_edges = graph.out_edges(first_state)
        in_edges = graph.in_edges(first_state)

        # First state must have only one output edge (with dst the second
        # state).
        if len(out_edges) != 1:
            return False
        # The interstate edge must not have a condition.
        if not out_edges[0].data.is_unconditional():
            return False
        # The interstate edge may have assignments, as long as there are input
        # edges to the first state that can absorb them.
        if out_edges[0].data.assignments:
            if not in_edges:
                return False
            # Fail if symbol is set before the state to fuse
            # TODO: Also fail if symbol is used in the dataflow of that state
            new_assignments = set(out_edges[0].data.assignments.keys())
            if any((new_assignments & set(e.data.assignments.keys()))
                   for e in in_edges):
                return False

        # There can be no state that have output edges pointing to both the
        # first and the second state. Such a case will produce a multi-graph.
        for src, _, _ in in_edges:
            for _, dst, _ in graph.out_edges(src):
                if dst == second_state:
                    return False

        if strict:
            # If second state has other input edges, there might be issues
            # Exceptions are when none of the states contain dataflow, unless
            # the first state is an initial state (in which case the new initial
            # state would be ambiguous).
            first_in_edges = graph.in_edges(first_state)
            second_in_edges = graph.in_edges(second_state)
            if ((not second_state.is_empty() or not first_state.is_empty()
                 or len(first_in_edges) == 0) and len(second_in_edges) != 1):
                return False

            # Get connected components.
            first_cc = [
                cc_nodes
                for cc_nodes in nx.weakly_connected_components(first_state._nx)
            ]
            second_cc = [
                cc_nodes
                for cc_nodes in nx.weakly_connected_components(second_state._nx)
            ]

            # Find source/sink (data) nodes
            first_input = {
                node
                for node in sdutil.find_source_nodes(first_state)
                if isinstance(node, nodes.AccessNode)
            }
            first_output = {
                node
                for node in first_state.nodes() if
                isinstance(node, nodes.AccessNode) and node not in first_input
            }
            second_input = {
                node
                for node in sdutil.find_source_nodes(second_state)
                if isinstance(node, nodes.AccessNode)
            }
            second_output = {
                node
                for node in second_state.nodes() if
                isinstance(node, nodes.AccessNode) and node not in second_input
            }

            # Find source/sink (data) nodes by connected component
            first_cc_input = [cc.intersection(first_input) for cc in first_cc]
            first_cc_output = [cc.intersection(first_output) for cc in first_cc]
            second_cc_input = [
                cc.intersection(second_input) for cc in second_cc
            ]
            second_cc_output = [
                cc.intersection(second_output) for cc in second_cc
            ]

            # Apply transformation in case all paths to the second state's
            # nodes go through the same access node, which implies sequential
            # behavior in SDFG semantics.
            check_strict = len(first_cc)
            for cc_output in first_cc_output:
                out_nodes = [
                    n for n in first_state.sink_nodes() if n in cc_output
                ]
                # Branching exists, multiple paths may involve same access node
                # potentially causing data races
                if len(out_nodes) > 1:
                    continue

                # Otherwise, check if any of the second state's connected
                # components for matching input
                for node in out_nodes:
                    if (next((x for x in second_input if x.label == node.label),
                             None) is not None):
                        check_strict -= 1
                        break

            if check_strict > 0:
                # Check strict conditions
                # RW dependency
                for node in first_input:
                    if (next(
                        (x for x in second_output if x.label == node.label),
                            None) is not None):
                        return False
                # WW dependency
                for node in first_output:
                    if (next(
                        (x for x in second_output if x.label == node.label),
                            None) is not None):
                        return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_state = graph.nodes()[candidate[StateFusion._first_state]]
        second_state = graph.nodes()[candidate[StateFusion._second_state]]

        return " -> ".join(state.label for state in [first_state, second_state])

    def apply(self, sdfg):
        first_state = sdfg.nodes()[self.subgraph[StateFusion._first_state]]
        second_state = sdfg.nodes()[self.subgraph[StateFusion._second_state]]

        # Remove interstate edge(s)
        edges = sdfg.edges_between(first_state, second_state)
        for edge in edges:
            if edge.data.assignments:
                for src, dst, other_data in sdfg.in_edges(first_state):
                    other_data.assignments.update(edge.data.assignments)
            sdfg.remove_edge(edge)

        # Special case 1: first state is empty
        if first_state.is_empty():
            sdutil.change_edge_dest(sdfg, first_state, second_state)
            sdfg.remove_node(first_state)
            return

        # Special case 2: second state is empty
        if second_state.is_empty():
            sdutil.change_edge_src(sdfg, second_state, first_state)
            sdutil.change_edge_dest(sdfg, second_state, first_state)
            sdfg.remove_node(second_state)
            return

        # Normal case: both states are not empty

        # Find source/sink (data) nodes
        first_input = [
            node for node in sdutil.find_source_nodes(first_state)
            if isinstance(node, nodes.AccessNode)
        ]
        first_output = [
            node for node in sdutil.find_sink_nodes(first_state)
            if isinstance(node, nodes.AccessNode)
        ]
        second_input = [
            node for node in sdutil.find_source_nodes(second_state)
            if isinstance(node, nodes.AccessNode)
        ]

        # first input = first input - first output
        first_input = [
            node for node in first_input
            if next((x for x in first_output
                     if x.label == node.label), None) is None
        ]

        # Merge second state to first state
        # First keep a backup of the topological sorted order of the nodes
        order = [
            x for x in reversed(list(nx.topological_sort(first_state._nx)))
            if isinstance(x, nodes.AccessNode)
        ]
        for node in second_state.nodes():
            first_state.add_node(node)
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        # Merge common (data) nodes
        for node in second_input:
            if first_state.in_degree(node) == 0:
                n = next((x for x in order if x.label == node.label), None)
                if n:
                    sdutil.change_edge_src(first_state, node, n)
                    first_state.remove_node(node)
                    n.access = dtypes.AccessType.ReadWrite

        # Redirect edges and remove second state
        sdutil.change_edge_src(sdfg, second_state, first_state)
        sdfg.remove_node(second_state)
        if Config.get_bool("debugprint"):
            StateFusion._states_fused += 1
