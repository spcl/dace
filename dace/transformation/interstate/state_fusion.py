"""State fusion transformation"""

import networkx as nx

from dace import sdfg, symbolic
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching


class StateFusion(pattern_matching.Transformation):
    """ Implements the state-fusion transformation.
        
        State-fusion takes two states that are connected through a single edge,
        and fuses them into one state. If strict, only applies if no memory 
        access hazards are created.
    """

    _first_state = sdfg.SDFGState()
    _edge = edges.InterstateEdge()
    _second_state = sdfg.SDFGState()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(StateFusion._first_state,
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
        if out_edges[0].data.condition.as_string != '':
            return False
        # The interstate edge may have assignments, as long as there are input
        # edges to the first state, that can absorb them.
        if out_edges[0].data.assignments and not in_edges:
            return False
        # There can be no state that have output edges pointing to both the
        # first and the second state. Such a case will produce a multi-graph.
        for src, _, _ in in_edges:
            for _, dst, _ in graph.out_edges(src):
                if dst == second_state:
                    return False

        if strict:

            # If second state has other input edges, there might be issues
            second_in_edges = graph.in_edges(second_state)
            if ((not second_state.is_empty() or not first_state.is_empty())
                    and len(second_in_edges) != 1):
                return False

            # Get connected components.
            first_cc = [
                cc_nodes
                for cc_nodes in nx.weakly_connected_components(first_state._nx)
            ]
            second_cc = [
                cc_nodes for cc_nodes in nx.weakly_connected_components(
                    second_state._nx)
            ]

            # Find source/sink (data) nodes
            first_input = {
                node
                for node in nxutil.find_source_nodes(first_state)
                if isinstance(node, nodes.AccessNode)
            }
            first_output = {
                node
                for node in first_state.nodes() if
                isinstance(node, nodes.AccessNode) and node not in first_input
            }
            second_input = {
                node
                for node in nxutil.find_source_nodes(second_state)
                if isinstance(node, nodes.AccessNode)
            }
            second_output = {
                node
                for node in second_state.nodes() if
                isinstance(node, nodes.AccessNode) and node not in second_input
            }

            # Find source/sink (data) nodes by connected component
            first_cc_input = [cc.intersection(first_input) for cc in first_cc]
            first_cc_output = [
                cc.intersection(first_output) for cc in first_cc
            ]
            second_cc_input = [
                cc.intersection(second_input) for cc in second_cc
            ]
            second_cc_output = [
                cc.intersection(second_output) for cc in second_cc
            ]

            check_strict = len(first_cc)
            for cc_output in first_cc_output:
                for node in cc_output:
                    if next((x for x in second_input
                             if x.label == node.label), None) is not None:
                        check_strict -= 1
                        break

            if check_strict > 0:
                # Check strict conditions
                # RW dependency
                for node in first_input:
                    if next((x for x in second_output
                             if x.label == node.label), None) is not None:
                        return False
                # WW dependency
                for node in first_output:
                    if next((x for x in second_output
                             if x.label == node.label), None) is not None:
                        return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_state = graph.nodes()[candidate[StateFusion._first_state]]
        second_state = graph.nodes()[candidate[StateFusion._second_state]]

        return ' -> '.join(
            state.label for state in [first_state, second_state])

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
            nxutil.change_edge_dest(sdfg, first_state, second_state)
            sdfg.remove_node(first_state)
            return

        # Special case 2: second state is empty
        if second_state.is_empty():
            nxutil.change_edge_src(sdfg, second_state, first_state)
            nxutil.change_edge_dest(sdfg, second_state, first_state)
            sdfg.remove_node(second_state)
            return

        # Normal case: both states are not empty

        # Find source/sink (data) nodes
        first_input = [
            node for node in nxutil.find_source_nodes(first_state)
            if isinstance(node, nodes.AccessNode)
        ]
        first_output = [
            node for node in nxutil.find_sink_nodes(first_state)
            if isinstance(node, nodes.AccessNode)
        ]
        second_input = [
            node for node in nxutil.find_source_nodes(second_state)
            if isinstance(node, nodes.AccessNode)
        ]

        # first input = first input - first output
        first_input = [
            node for node in first_input
            if next((x for x in first_output
                     if x.label == node.label), None) is None
        ]

        # Merge second state to first state
        for node in second_state.nodes():
            first_state.add_node(node)
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        # Merge common (data) nodes
        for node in first_input:
            try:
                old_node = next(
                    x for x in second_input if x.label == node.label)
            except StopIteration:
                continue
            nxutil.change_edge_src(first_state, old_node, node)
            first_state.remove_node(old_node)
        for node in first_output:
            try:
                new_node = next(
                    x for x in second_input if x.label == node.label)
            except StopIteration:
                continue
            nxutil.change_edge_dest(first_state, node, new_node)
            first_state.remove_node(node)

        # Redirect edges and remove second state
        nxutil.change_edge_src(sdfg, second_state, first_state)
        sdfg.remove_node(second_state)


pattern_matching.Transformation.register_stateflow_pattern(StateFusion)
