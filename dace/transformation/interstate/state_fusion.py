# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from typing import Dict, List, Set

import networkx as nx

from dace import data as dt, dtypes, registry, sdfg, subsets
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation


# Helper class for finding connected component correspondences
class CCDesc:

    def __init__(self, first_input_nodes: Set[nodes.AccessNode], first_output_nodes: Set[nodes.AccessNode],
                 second_input_nodes: Set[nodes.AccessNode], second_output_nodes: Set[nodes.AccessNode]) -> None:
        self.first_inputs = {n.data for n in first_input_nodes}
        self.first_input_nodes = first_input_nodes
        self.first_outputs = {n.data for n in first_output_nodes}
        self.first_output_nodes = first_output_nodes
        self.second_inputs = {n.data for n in second_input_nodes}
        self.second_input_nodes = second_input_nodes
        self.second_outputs = {n.data for n in second_output_nodes}
        self.second_output_nodes = second_output_nodes


def top_level_nodes(state: SDFGState):
    return state.scope_children()[None]


class StateFusion(transformation.MultiStateTransformation):
    """ Implements the state-fusion transformation.

        State-fusion takes two states that are connected through a single edge,
        and fuses them into one state. If permissive, also applies if potential memory
        access hazards are created.
    """

    first_state = transformation.PatternNode(sdfg.SDFGState)
    second_state = transformation.PatternNode(sdfg.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_state, cls.second_state)]

    @staticmethod
    def find_fused_components(first_cc_input, first_cc_output, second_cc_input, second_cc_output) -> List[CCDesc]:
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
                    input1 |= first_cc_input[cind]
                    output1 |= first_cc_output[cind]
                else:
                    input2 |= second_cc_input[cind]
                    output2 |= second_cc_output[cind]
            result.append(CCDesc(input1, output1, input2, output2))

        return result

    @staticmethod
    def memlets_intersect(graph_a: SDFGState, group_a: List[nodes.AccessNode], inputs_a: bool, graph_b: SDFGState,
                          group_b: List[nodes.AccessNode], inputs_b: bool) -> bool:
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
        src_subset = lambda e: (e.data.src_subset if e.data.src_subset is not None else e.data.dst_subset)
        dst_subset = lambda e: (e.data.dst_subset if e.data.dst_subset is not None else e.data.src_subset)
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

    def has_path(self, first_state: SDFGState, second_state: SDFGState,
                 match_nodes: Dict[nodes.AccessNode, nodes.AccessNode], node_a: nodes.Node, node_b: nodes.Node) -> bool:
        """ Check for paths between the two states if they are fused. """
        for match_a, match_b in match_nodes.items():
            if nx.has_path(first_state._nx, node_a, match_a) and nx.has_path(second_state._nx, match_b, node_b):
                return True
        return False

    def _check_all_paths(self, first_state: SDFGState, second_state: SDFGState,
                         match_nodes: Dict[nodes.AccessNode, nodes.AccessNode], nodes_first: List[nodes.AccessNode],
                         nodes_second: List[nodes.AccessNode], first_read: bool, second_read: bool) -> bool:
        for node_a in nodes_first:
            succ_a = first_state.successors(node_a)
            for node_b in nodes_second:
                if all(self.has_path(first_state, second_state, match_nodes, sa, node_b) for sa in succ_a):
                    return True
        # Path not found, check memlets
        if StateFusion.memlets_intersect(first_state, nodes_first, first_read, second_state, nodes_second, second_read):
            return False
        return True

    def _check_paths(self, first_state: SDFGState, second_state: SDFGState, match_nodes: Dict[nodes.AccessNode,
                                                                                              nodes.AccessNode],
                     nodes_first: List[nodes.AccessNode], nodes_second: List[nodes.AccessNode],
                     second_input: Set[nodes.AccessNode], first_read: bool, second_read: bool) -> bool:
        fail = False
        path_found = False
        for match in match_nodes:
            for node in nodes_first:
                path_to = nx.has_path(first_state._nx, node, match)
                if not path_to:
                    continue
                path_found = True
                node2 = next(n for n in second_input if n.data == match.data)
                if not all(nx.has_path(second_state._nx, node2, n) for n in nodes_second):
                    fail = True
                    break
            if fail or path_found:
                break

        # Check for intersection (if None, fusion is ok)
        if fail or not path_found:
            if StateFusion.memlets_intersect(first_state, nodes_first, first_read, second_state, nodes_second,
                                             second_read):
                return False
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

        out_edges = graph.out_edges(first_state)
        in_edges = graph.in_edges(first_state)

        # First state must have only one output edge (with dst the second
        # state).
        if len(out_edges) != 1:
            return False
        # If both states have more than one incoming edge, some control flow
        # may become ambiguous
        if len(in_edges) > 1 and graph.in_degree(second_state) > 1:
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
            if any((new_assignments & set(e.data.assignments.keys())) for e in in_edges):
                return False
            # Fail if symbol is used in the dataflow of that state
            if len(new_assignments & first_state.free_symbols) > 0:
                return False
            # Fail if assignments have free symbols that are updated in the
            # first state
            freesyms = out_edges[0].data.free_symbols
            if freesyms and any(n.data in freesyms for n in first_state.nodes()
                                if isinstance(n, nodes.AccessNode) and first_state.in_degree(n) > 0):
                return False
            # Fail if symbols assigned on the first edge are free symbols on the
            # second edge
            symbols_used = set(out_edges[0].data.free_symbols)
            for e in in_edges:
                if e.data.assignments.keys() & symbols_used:
                    return False
                # Also fail in the inverse; symbols assigned on the second edge are free symbols on the first edge
                if new_assignments & set(e.data.free_symbols):
                    return False

        # There can be no state that have output edges pointing to both the
        # first and the second state. Such a case will produce a multi-graph.
        for src, _, _ in in_edges:
            for _, dst, _ in graph.out_edges(src):
                if dst == second_state:
                    return False

        if not permissive:
            # Strict mode that inhibits state fusion if Python callbacks are involved
            if Config.get_bool('frontend', 'dont_fuse_callbacks'):
                for node in (first_state.data_nodes() + second_state.data_nodes()):
                    if node.data == '__pystate':
                        return False

            # NOTE: This is quick fix for MPI Waitall (probably also needed for
            # Wait), until we have a better SDFG representation of the buffer
            # dependencies.
            try:
                next(node for node in first_state.nodes()
                     if (isinstance(node, nodes.LibraryNode) and type(node).__name__ == 'Waitall')
                     or node.label == '_Waitall_')
                return False
            except StopIteration:
                pass
            try:
                next(node for node in second_state.nodes()
                     if (isinstance(node, nodes.LibraryNode) and type(node).__name__ == 'Waitall')
                     or node.label == '_Waitall_')
                return False
            except StopIteration:
                pass

            # If second state has other input edges, there might be issues
            # Exceptions are when none of the states contain dataflow, unless
            # the first state is an initial state (in which case the new initial
            # state would be ambiguous).
            first_in_edges = graph.in_edges(first_state)
            second_in_edges = graph.in_edges(second_state)
            if ((not second_state.is_empty() or not first_state.is_empty() or len(first_in_edges) == 0)
                    and len(second_in_edges) != 1):
                return False

            # Get connected components.
            first_cc = [cc_nodes for cc_nodes in nx.weakly_connected_components(first_state._nx)]
            second_cc = [cc_nodes for cc_nodes in nx.weakly_connected_components(second_state._nx)]

            # Find source/sink (data) nodes
            first_input = {node for node in first_state.source_nodes() if isinstance(node, nodes.AccessNode)}
            first_output = {
                node
                for node in first_state.scope_children()[None]
                if isinstance(node, nodes.AccessNode) and node not in first_input
            }
            second_input = {node for node in second_state.source_nodes() if isinstance(node, nodes.AccessNode)}
            second_output = {
                node
                for node in second_state.scope_children()[None]
                if isinstance(node, nodes.AccessNode) and node not in second_input
            }

            # Find source/sink (data) nodes by connected component
            first_cc_input = [cc.intersection(first_input) for cc in first_cc]
            first_cc_output = [cc.intersection(first_output) for cc in first_cc]
            second_cc_input = [cc.intersection(second_input) for cc in second_cc]
            second_cc_output = [cc.intersection(second_output) for cc in second_cc]

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
            resulting_ccs: List[CCDesc] = StateFusion.find_fused_components(first_cc_input, first_cc_output,
                                                                            second_cc_input, second_cc_output)

            # Check for data races
            for fused_cc in resulting_ccs:
                # Write-Write hazard - data is output of both first and second
                # states, without a read in between
                write_write_candidates = ((fused_cc.first_outputs & fused_cc.second_outputs) - fused_cc.second_inputs)

                # Find the leaf (topological) instances of the matches
                order = [
                    x for x in reversed(list(nx.topological_sort(first_state._nx)))
                    if isinstance(x, nodes.AccessNode) and x.data in fused_cc.first_outputs
                ]
                # Those nodes will be the connection points upon fusion
                match_nodes: Dict[nodes.AccessNode, nodes.AccessNode] = {
                    next(n for n in order
                         if n.data == match): next(n for n in fused_cc.second_input_nodes if n.data == match)
                    for match in (fused_cc.first_outputs
                                  & fused_cc.second_inputs)
                }

                # If we have potential candidates, check if there is a
                # path from the first write to the second write (in that
                # case, there is no hazard):
                for cand in write_write_candidates:
                    nodes_first = [n for n in first_output if n.data == cand]
                    nodes_second = [n for n in second_output if n.data == cand]

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(first_state, second_state, match_nodes, nodes_first, nodes_second,
                                             second_input, False, False):
                        return False
                # End of write-write hazard check

                first_inout = fused_cc.first_inputs | fused_cc.first_outputs
                for other_cc in resulting_ccs:
                    # NOTE: Special handling for `other_cc is fused_cc`
                    if other_cc is fused_cc:
                        # Checking for potential Read-Write data races
                        for d in first_inout:
                            if d in other_cc.second_outputs:
                                nodes_second = [n for n in second_output if n.data == d]
                                # Read-Write race
                                if d in fused_cc.first_inputs:
                                    nodes_first = [n for n in first_input if n.data == d]
                                else:
                                    nodes_first = []
                                for n2 in nodes_second:
                                    for e in second_state.in_edges(n2):
                                        path = second_state.memlet_path(e)
                                        src = path[0].src
                                        if src in second_input and src.data in fused_cc.first_outputs:
                                            for n1 in fused_cc.first_output_nodes:
                                                if n1.data == src.data:
                                                    for n0 in nodes_first:
                                                        if not nx.has_path(first_state._nx, n0, n1):
                                                            return False
                                # Read-write hazard where an access node is connected
                                # to more than one output at once: (a) -> (b)  |  (d) -> [code] -> (d)
                                #                                     \-> (c)  |
                                # in the first state, and the same memory is inout in the second state
                                # All paths need to lead to `src`
                                if not self._check_all_paths(first_state, second_state, match_nodes, nodes_first,
                                                             nodes_second, True, False):
                                    return False

                        continue
                    # If an input/output of a connected component in the first
                    # state is an output of another connected component in the
                    # second state, we have a potential data race (Read-Write
                    # or Write-Write)
                    for d in first_inout:
                        if d in other_cc.second_outputs:
                            # Check for intersection (if None, fusion is ok)
                            nodes_second = [n for n in second_output if n.data == d]
                            # Read-Write race
                            if d in fused_cc.first_inputs:
                                nodes_first = [n for n in first_input if n.data == d]
                                if StateFusion.memlets_intersect(first_state, nodes_first, True, second_state,
                                                                 nodes_second, False):
                                    return False
                            # Write-Write race
                            if d in fused_cc.first_outputs:
                                nodes_first = [n for n in first_output if n.data == d]
                                if StateFusion.memlets_intersect(first_state, nodes_first, False, second_state,
                                                                 nodes_second, False):
                                    return False
                    # End of data race check

                # Read-after-write dependencies: if there is an output of the
                # second state that is an input of the first, ensure all paths
                # from the input of the first state lead to the output.
                # Otherwise, there may be a RAW due to topological sort or
                # concurrency.
                second_inout = ((fused_cc.first_inputs | fused_cc.first_outputs) & fused_cc.second_outputs)
                for inout in second_inout:
                    nodes_first = [n for n in match_nodes if n.data == inout]
                    if any(first_state.out_degree(n) > 0 for n in nodes_first):
                        return False

                    # If we have potential candidates, check if there is a
                    # path from the first read to the second write (in that
                    # case, there is no hazard):
                    nodes_first = {
                        n
                        for n in fused_cc.first_input_nodes
                        | fused_cc.first_output_nodes if n.data == inout
                    }
                    nodes_second = {n for n in fused_cc.second_output_nodes if n.data == inout}

                    # If there is a path for the candidate that goes through
                    # the match nodes in both states, there is no conflict
                    if not self._check_paths(first_state, second_state, match_nodes, nodes_first, nodes_second,
                                             second_input, True, False):
                        return False

                # End of read-write hazard check

                # Read-after-write dependencies: if there is more than one first
                # output with the same data, make sure it can be unambiguously
                # connected to the second state
                if (len(fused_cc.first_output_nodes) > len(fused_cc.first_outputs)):
                    for inpnode in fused_cc.second_input_nodes:
                        found = None
                        for outnode in fused_cc.first_output_nodes:
                            if outnode.data != inpnode.data:
                                continue
                            if StateFusion.memlets_intersect(first_state, [outnode], False, second_state, [inpnode],
                                                             True):
                                # If found more than once, either there is a
                                # path from one to another or it is ambiguous
                                if found is not None:
                                    if nx.has_path(first_state.nx, outnode, found):
                                        # Found is a descendant, continue
                                        continue
                                    elif nx.has_path(first_state.nx, found, outnode):
                                        # New node is a descendant, set as found
                                        found = outnode
                                    else:
                                        # No path: ambiguous match
                                        return False
                                found = outnode

        # Do not fuse FPGA and NON-FPGA states (unless one of them is empty)
        if first_state.number_of_nodes() > 0 and second_state.number_of_nodes() > 0 and sdutil.is_fpga_kernel(
                sdfg, first_state) != sdutil.is_fpga_kernel(sdfg, second_state):
            return False

        return True

    def apply(self, _, sdfg):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

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
            if sdfg.start_state == first_state:
                sdfg.start_state = sdfg.node_id(second_state)
            return

        # Special case 2: second state is empty
        if second_state.is_empty():
            sdutil.change_edge_src(sdfg, second_state, first_state)
            sdutil.change_edge_dest(sdfg, second_state, first_state)
            sdfg.remove_node(second_state)
            if sdfg.start_state == second_state:
                sdfg.start_state = sdfg.node_id(first_state)
            return

        # Normal case: both states are not empty

        # Find source/sink (data) nodes
        first_input = [node for node in first_state.source_nodes() if isinstance(node, nodes.AccessNode)]
        first_output = [node for node in first_state.sink_nodes() if isinstance(node, nodes.AccessNode)]
        second_input = [node for node in second_state.source_nodes() if isinstance(node, nodes.AccessNode)]

        top2 = top_level_nodes(second_state)

        # first input = first input - first output
        first_input = [
            node for node in first_input if next((x for x in first_output if x.data == node.data), None) is None
        ]

        # NOTE: We exclude Views from the process of merging common data nodes because it may lead to double edges.
        second_mid = [
            x for x in list(nx.topological_sort(second_state._nx)) if isinstance(x, nodes.AccessNode)
            and second_state.out_degree(x) > 0 and not isinstance(sdfg.arrays[x.data], dt.View)
        ]

        # Merge second state to first state
        # First keep a backup of the topological sorted order of the nodes
        sdict = first_state.scope_dict()
        order = [
            x for x in reversed(list(nx.topological_sort(first_state._nx)))
            if isinstance(x, nodes.AccessNode) and sdict[x] is None
        ]
        for node in second_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                # update parent information
                node.sdfg.parent = first_state
            first_state.add_node(node)
        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        top = top_level_nodes(first_state)

        # Merge common (data) nodes
        merged_nodes = set()
        for node in second_mid:

            # merge only top level nodes, skip everything else
            if node not in top2:
                continue

            candidates = [x for x in order if x.data == node.data and x in top and x not in merged_nodes]
            source_node = first_state.in_degree(node) == 0

            # If not source node, try to connect every memlet-intersecting candidate
            if not source_node:
                for cand in candidates:
                    if StateFusion.memlets_intersect(first_state, [cand], False, second_state, [node], True):
                        if nx.has_path(first_state._nx, cand, node):  # Do not create cycles
                            continue
                        sdutil.change_edge_src(first_state, cand, node)
                        sdutil.change_edge_dest(first_state, cand, node)
                        first_state.remove_node(cand)
                continue

            if len(candidates) == 0:
                continue
            elif len(candidates) == 1:
                n = candidates[0]
            else:
                # Choose first candidate that intersects memlets
                for cand in candidates:
                    if StateFusion.memlets_intersect(first_state, [cand], False, second_state, [node], True):
                        n = cand
                        break
                else:
                    # No node intersects, use topologically-last node
                    n = candidates[0]

            sdutil.change_edge_src(first_state, node, n)
            sdutil.change_edge_dest(first_state, node, n)
            first_state.remove_node(node)
            merged_nodes.add(n)

        # Redirect edges and remove second state
        sdutil.change_edge_src(sdfg, second_state, first_state)
        sdfg.remove_node(second_state)
        if sdfg.start_state == second_state:
            sdfg.start_state = sdfg.node_id(first_state)
