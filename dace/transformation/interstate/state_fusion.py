# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from dace.sdfg.state import SDFGState
from typing import List, Set
import networkx as nx

from dace import dtypes, registry, sdfg, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.config import Config


# Helper class for finding connected component correspondences
class CCDesc:
    def __init__(self, first_inputs: Set[str], first_outputs: Set[str],
                 second_inputs: Set[str], second_outputs: Set[str]) -> None:
        self.first_inputs = first_inputs
        self.first_outputs = first_outputs
        self.second_inputs = second_inputs
        self.second_outputs = second_outputs


@registry.autoregister_params(strict=True)
class StateFusion(transformation.Transformation):
    """ Implements the state-fusion transformation.
        
        State-fusion takes two states that are connected through a single edge,
        and fuses them into one state. If strict, only applies if no memory 
        access hazards are created.
    """

    _first_state = sdfg.SDFGState()
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
    def find_fused_components(first_cc_input, first_cc_output, second_cc_input,
                              second_cc_output) -> List[CCDesc]:
        # Make a bipartite graph out of the first and second components
        g = nx.DiGraph()
        g.add_nodes_from((0, i) for i in range(len(first_cc_output)))
        g.add_nodes_from((1, i) for i in range(len(second_cc_output)))
        # Find matching nodes in second state
        for i, cc1 in enumerate(first_cc_output):
            outnames1 = {n.data for n in cc1}
            for j, cc2 in enumerate(second_cc_input):
                inpnames2 = {n.data for n in cc2}
                if len(outnames1 & inpnames2) > 0:
                    g.add_edge((0, i), (1, j))

        # Construct result out of connected components of the bipartite graph
        result = []
        for cc in nx.weakly_connected_components(g):
            input1, output1, input2, output2 = set(), set(), set(), set()
            for gind, cind in cc:
                if gind == 0:
                    input1 |= {n.data for n in first_cc_input[cind]}
                    output1 |= {n.data for n in first_cc_output[cind]}
                else:
                    input2 |= {n.data for n in second_cc_input[cind]}
                    output2 |= {n.data for n in second_cc_output[cind]}
            result.append(CCDesc(input1, output1, input2, output2))

        return result

    @staticmethod
    def memlets_intersect(graph_a: SDFGState, group_a: List[nodes.AccessNode],
                          inputs_a: bool, graph_b: SDFGState,
                          group_b: List[nodes.AccessNode],
                          inputs_b: bool) -> bool:
        """ 
        Performs an all-pairs check for subset intersection on two
        groups of nodes. If group intersects or result is indeterminate, 
        returns True as a precaution.
        :param graph_a: The graph in which the first set of nodes reside.
        :param group_a: The first set of nodes to check.
        :param inputs_a: If True, checks inputs of the first group.
        :param graph_b: The graph in which the second set of nodes reside.
        :param group_b: The second set of nodes to check.
        :param inputs_b: If True, checks inputs of the second group.
        :returns True if subsets intersect or result is indeterminate.
        """
        # Set traversal functions
        src_subset = lambda e: (e.data.src_subset if e.data.src_subset is
                                not None else e.data.dst_subset)
        dst_subset = lambda e: (e.data.dst_subset if e.data.dst_subset is
                                not None else e.data.src_subset)
        if inputs_a:
            edges_a = [e for n in group_a for e in graph_a.out_edges(n)]
            subset_a = src_subset
        else:
            edges_a = [e for n in group_a for e in graph_a.in_edges(n)]
            subset_a = dst_subset
        if inputs_b:
            edges_b = [e for n in group_b for e in graph_b.out_edges(n)]
            subset_b = src_subset
        else:
            edges_b = [e for n in group_b for e in graph_b.in_edges(n)]
            subset_b = dst_subset

        # Simple all-pairs check
        for ea in edges_a:
            for eb in edges_b:
                result = subsets.intersects(subset_a(ea), subset_b(eb))
                if result is True or result is None:
                    return True
        return False

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
            new_assignments = set(out_edges[0].data.assignments.keys())
            if any((new_assignments & set(e.data.assignments.keys()))
                   for e in in_edges):
                return False
            # Fail if symbol is used in the dataflow of that state
            if len(new_assignments & first_state.free_symbols) > 0:
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
            first_output_names = {node.data for node in first_output}
            second_input_names = {node.data for node in second_input}

            # If any second input appears more than once, fail
            if len(second_input) > len(second_input_names):
                return False

            # If any first output that is an input to the second state
            # appears in more than one CC, fail
            matches = first_output_names & second_input_names
            for match in matches:
                cc_appearances = 0
                for cc in first_cc_output:
                    if len([n for n in cc if n.data == match]) > 0:
                        cc_appearances += 1
                if cc_appearances > 1:
                    return False

            # Recreate fused connected component correspondences, and then
            # check for hazards
            resulting_ccs: List[CCDesc] = StateFusion.find_fused_components(
                first_cc_input, first_cc_output, second_cc_input,
                second_cc_output)

            # Check for data races
            for fused_cc in resulting_ccs:
                # Write-Write hazard - data is output of both first and second
                # states, without a read in between
                write_write_candidates = (
                    (fused_cc.first_outputs & fused_cc.second_outputs) -
                    fused_cc.second_inputs)
                if len(write_write_candidates) > 0:
                    # If we have potential candidates, check if there is a
                    # path from the first write to the second write (in that
                    # case, there is no hazard):
                    # Find the leaf (topological) instances of the matches
                    order = [
                        x for x in reversed(
                            list(nx.topological_sort(first_state._nx)))
                        if isinstance(x, nodes.AccessNode)
                        and x.data in fused_cc.first_outputs
                    ]
                    # Those nodes will be the connection points upon fusion
                    match_nodes = {
                        next(n for n in order if n.data == match)
                        for match in (fused_cc.first_outputs
                                      & fused_cc.second_inputs)
                    }
                else:
                    match_nodes = set()

                for cand in write_write_candidates:
                    nodes_first = [n for n in first_output if n.data == cand]
                    nodes_second = [n for n in second_output if n.data == cand]

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    fail = False
                    path_found = False
                    for match in match_nodes:
                        for node in nodes_first:
                            path_to = nx.has_path(first_state._nx, node, match)
                            if not path_to:
                                continue
                            path_found = True
                            node2 = next(n for n in second_input
                                         if n.data == match.data)
                            if not all(
                                    nx.has_path(second_state._nx, node2, n)
                                    for n in nodes_second):
                                fail = True
                                break
                        if fail or path_found:
                            break

                    # Check for intersection (if None, fusion is ok)
                    if fail or not path_found:
                        if StateFusion.memlets_intersect(
                                first_state, nodes_first, False, second_state,
                                nodes_second, False):
                            return False
                # End of write-write hazard check

                first_inout = fused_cc.first_inputs | fused_cc.first_outputs
                for other_cc in resulting_ccs:
                    if other_cc is fused_cc:
                        continue
                    # If an input/output of a connected component in the first
                    # state is an output of another connected component in the
                    # second state, we have a potential data race (Read-Write
                    # or Write-Write)
                    for d in first_inout:
                        if d in other_cc.second_outputs:
                            # Check for intersection (if None, fusion is ok)
                            nodes_second = [
                                n for n in second_output if n.data == d
                            ]
                            # Read-Write race
                            if d in fused_cc.first_inputs:
                                nodes_first = [
                                    n for n in first_input if n.data == d
                                ]
                                if StateFusion.memlets_intersect(
                                        first_state, nodes_first, True,
                                        second_state, nodes_second, False):
                                    return False
                            # Write-Write race
                            if d in fused_cc.first_outputs:
                                nodes_first = [
                                    n for n in first_output if n.data == d
                                ]
                                if StateFusion.memlets_intersect(
                                        first_state, nodes_first, False,
                                        second_state, nodes_second, False):
                                    return False
                    # End of data race check

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
