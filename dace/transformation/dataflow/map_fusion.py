# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace import data, dtypes, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx


class MapFusion(transformation.SingleStateTransformation):
    """ Implements the MapFusion transformation.
        It wil check for all patterns MapExit -> AccessNode -> MapEntry, and
        based on the following rules, fuse them and remove the transient in
        between. There are several possibilities of what it does to this
        transient in between.

        Essentially, if there is some other place in the
        sdfg where it is required, or if it is not a transient, then it will
        not be removed. In such a case, it will be linked to the MapExit node
        of the new fused map.

        Rules for fusing maps:
          0. The map range of the second map should be a permutation of the
             first map range.
          1. Each of the access nodes that are adjacent to the first map exit
             should have an edge to the second map entry. If it doesn't, then the
             second map entry should not be reachable from this access node.
          2. Any node that has a wcr from the first map exit should not be
             adjacent to the second map entry.
          3. Access pattern for the access nodes in the second map should be
             the same permutation of the map parameters as the map ranges of the
             two maps. Alternatively, this access node should not be adjacent to
             the first map entry.
    """
    first_map_exit = transformation.PatternNode(nodes.ExitNode)
    array = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.EntryNode)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    @staticmethod
    def find_permutation(first_map: nodes.Map, second_map: nodes.Map) -> Union[List[int], None]:
        """ Find permutation between two map ranges.
            :param first_map: First map.
            :param second_map: Second map.
            :return: None if no such permutation exists, otherwise a list of
                     indices L such that L[x]'th parameter of second map has the same range as x'th
                     parameter of the first map.
            """
        result = []

        if len(first_map.range) != len(second_map.range):
            return None

        # Match map ranges with reduce ranges
        for i, tmap_rng in enumerate(first_map.range):
            found = False
            for j, rng in enumerate(second_map.range):
                if tmap_rng == rng and j not in result:
                    result.append(j)
                    found = True
                    break
            if not found:
                break

        # Ensure all map ranges matched
        if len(result) != len(first_map.range):
            return None

        return result

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        first_map_exit = self.first_map_exit
        first_map_entry = graph.entry_node(first_map_exit)
        second_map_entry = self.second_map_entry
        second_map_exit = graph.exit_node(second_map_entry)

        for _in_e in graph.in_edges(first_map_exit):
            if _in_e.data.wcr is not None:
                for _out_e in graph.out_edges(second_map_entry):
                    if _out_e.data.data == _in_e.data.data:
                        # wcr is on a node that is used in the second map, quit
                        return False
        # Check whether there is a pattern map -> access -> map.
        intermediate_nodes = set()
        intermediate_data = set()
        for _, _, dst, _, _ in graph.out_edges(first_map_exit):
            if isinstance(dst, nodes.AccessNode):
                intermediate_nodes.add(dst)
                intermediate_data.add(dst.data)

                # If array is used anywhere else in this state.
                num_occurrences = len([
                    n for s in sdfg.nodes() for n in s.nodes() if isinstance(n, nodes.AccessNode) and n.data == dst.data
                ])
                if num_occurrences > 1:
                    return False
            else:
                return False
        # Check map ranges
        perm = self.find_permutation(first_map_entry.map, second_map_entry.map)
        if perm is None:
            return False

        # Check if any intermediate transient is also going to another location
        second_inodes = set(e.src for e in graph.in_edges(second_map_entry) if isinstance(e.src, nodes.AccessNode))
        transients_to_remove = intermediate_nodes & second_inodes
        # if any(e.dst != second_map_entry for n in transients_to_remove
        #        for e in graph.out_edges(n)):
        if any(graph.out_degree(n) > 1 for n in transients_to_remove):
            return False

        # Create a dict that maps parameters of the first map to those of the
        # second map.
        params_dict = {}
        for _index, _param in enumerate(second_map_entry.map.params):
            params_dict[_param] = first_map_entry.map.params[perm[_index]]

        out_memlets = [e.data for e in graph.in_edges(first_map_exit)]

        # Check that input set of second map is provided by the output set
        # of the first map, or other unrelated maps
        for second_edge in graph.out_edges(second_map_entry):
            # Memlets that do not come from one of the intermediate arrays
            if second_edge.data.data not in intermediate_data:
                # however, if intermediate_data eventually leads to
                # second_memlet.data, need to fail.
                for _n in intermediate_nodes:
                    source_node = _n
                    destination_node = graph.memlet_path(second_edge)[0].src
                    # NOTE: Assumes graph has networkx version
                    if destination_node in nx.descendants(graph._nx, source_node):
                        return False
                continue

            provided = False

            # Compute second subset with respect to first subset's symbols
            sbs_permuted = dcpy(second_edge.data.subset)
            if sbs_permuted:
                # Create intermediate dicts to avoid conflicts, such as {i:j, j:i}
                symbolic.safe_replace(params_dict, lambda m: sbs_permuted.replace(m))

            for first_memlet in out_memlets:
                if first_memlet.data != second_edge.data.data:
                    continue

                # If there is a covered subset, it is provided
                if first_memlet.subset.covers(sbs_permuted):
                    provided = True
                    break

            # If none of the output memlets of the first map provide the info,
            # fail.
            if provided is False:
                return False

        # Checking for stencil pattern and common input/output data
        # (after fusing the maps)
        first_map_inputnodes = {
            e.src: e.src.data
            for e in graph.in_edges(first_map_entry) if isinstance(e.src, nodes.AccessNode)
        }
        input_views = set()
        viewed_inputnodes = dict()
        for n in first_map_inputnodes.keys():
            if isinstance(n.desc(sdfg), data.View):
                input_views.add(n)
        for v in input_views:
            del first_map_inputnodes[v]
            e = sdutil.get_view_edge(graph, v)
            if e:
                first_map_inputnodes[e.src] = e.src.data
                viewed_inputnodes[e.src.data] = v
        second_map_outputnodes = {
            e.dst: e.dst.data
            for e in graph.out_edges(second_map_exit) if isinstance(e.dst, nodes.AccessNode)
        }
        output_views = set()
        viewed_outputnodes = dict()
        for n in second_map_outputnodes:
            if isinstance(n.desc(sdfg), data.View):
                output_views.add(n)
        for v in output_views:
            del second_map_outputnodes[v]
            e = sdutil.get_view_edge(graph, v)
            if e:
                second_map_outputnodes[e.dst] = e.dst.data
                viewed_outputnodes[e.dst.data] = v
        common_data = set(first_map_inputnodes.values()).intersection(set(second_map_outputnodes.values()))
        if common_data:
            input_data = [viewed_inputnodes[d].data if d in viewed_inputnodes.keys() else d for d in common_data]
            input_accesses = [
                graph.memlet_path(e)[-1].data.src_subset for e in graph.out_edges(first_map_entry)
                if e.data.data in input_data
            ]
            if len(input_accesses) > 1:
                for i, a in enumerate(input_accesses[:-1]):
                    for b in input_accesses[i + 1:]:
                        if isinstance(a, subsets.Indices):
                            c = subsets.Range.from_indices(a)
                            c.offset(b, negative=True)
                        else:
                            c = a.offset_new(b, negative=True)
                        for r in c:
                            if r != (0, 0, 1):
                                return False

            output_data = [viewed_outputnodes[d].data if d in viewed_outputnodes.keys() else d for d in common_data]
            output_accesses = [
                graph.memlet_path(e)[0].data.dst_subset for e in graph.in_edges(second_map_exit)
                if e.data.data in output_data
            ]

            # Compute output accesses with respect to first map's symbols
            oacc_permuted = [dcpy(a) for a in output_accesses]
            for a in oacc_permuted:
                # Create intermediate dicts to avoid conflicts, such as {i:j, j:i}
                symbolic.safe_replace(params_dict, lambda m: a.replace(m))

            a = input_accesses[0]
            for b in oacc_permuted:
                if isinstance(a, subsets.Indices):
                    c = subsets.Range.from_indices(a)
                    c.offset(b, negative=True)
                else:
                    c = a.offset_new(b, negative=True)
                for r in c:
                    if r != (0, 0, 1):
                        return False

        # Success
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        """
            This method applies the mapfusion transformation.
            Other than the removal of the second map entry node (SME), and the first
            map exit (FME) node, it has the following side effects:

            1.  Any transient adjacent to both FME and SME with degree = 2 will be removed.
                The tasklets that use/produce it shall be connected directly with a
                scalar/new transient (if the dataflow is more than a single scalar)

            2.  If this transient is adjacent to FME and SME and has other
                uses, it will be adjacent to the new map exit post fusion.
                Tasklet-> Tasklet edges will ALSO be added as mentioned above.

            3.  If an access node is adjacent to FME but not SME, it will be
                adjacent to new map exit post fusion.

            4.  If an access node is adjacent to SME but not FME, it will be
                adjacent to the new map entry node post fusion.

        """
        first_exit = self.first_map_exit
        first_entry = graph.entry_node(first_exit)
        second_entry = self.second_map_entry
        second_exit = graph.exit_node(second_entry)

        intermediate_nodes = set()
        for _, _, dst, _, _ in graph.out_edges(first_exit):
            intermediate_nodes.add(dst)
            assert isinstance(dst, nodes.AccessNode)

        # Check if an access node refers to non transient memory, or transient
        # is used at another location (cannot erase)
        do_not_erase = set()
        for node in intermediate_nodes:
            if sdfg.arrays[node.data].transient is False:
                do_not_erase.add(node)
            else:
                for edge in graph.in_edges(node):
                    if edge.src != first_exit:
                        do_not_erase.add(node)
                        break
                else:
                    for edge in graph.out_edges(node):
                        if edge.dst != second_entry:
                            do_not_erase.add(node)
                            break

        # Find permutation between first and second scopes
        perm = self.find_permutation(first_entry.map, second_entry.map)
        params_dict = {}
        for index, param in enumerate(first_entry.map.params):
            params_dict[param] = second_entry.map.params[perm[index]]

        # Replaces (in memlets and tasklet) the second scope map
        # indices with the permuted first map indices.
        # This works in two passes to avoid problems when e.g., exchanging two
        # parameters (instead of replacing (j,i) and (i,j) to (j,j) and then
        # i,i).
        second_scope = graph.scope_subgraph(second_entry)
        for firstp, secondp in params_dict.items():
            if firstp != secondp:
                replace(second_scope, secondp, '__' + secondp + '_fused')
        for firstp, secondp in params_dict.items():
            if firstp != secondp:
                replace(second_scope, '__' + secondp + '_fused', firstp)

        # Isolate First exit node
        ############################
        edges_to_remove = set()
        nodes_to_remove = set()
        for edge in graph.in_edges(first_exit):
            tree = graph.memlet_tree(edge)
            access_node = tree.root().edge.dst
            if access_node not in do_not_erase:
                out_edges = [e for e in graph.out_edges(access_node) if e.dst == second_entry]
                # In this transformation, there can only be one edge to the
                # second map
                assert len(out_edges) == 1

                # Get source connector to the second map
                connector = out_edges[0].dst_conn[3:]

                new_dsts = []
                # Look at the second map entry out-edges to get the new
                # destinations
                for e in graph.out_edges(second_entry):
                    if e.src_conn[4:] == connector:
                        new_dsts.append(e)
                if not new_dsts:  # Access node is not used in the second map
                    nodes_to_remove.add(access_node)
                    continue

                # Add a transient scalar/array
                self.fuse_nodes(sdfg, graph, edge, new_dsts[0].dst, new_dsts[0].dst_conn, new_dsts[1:])

                edges_to_remove.add(edge)

                # Remove transient node between the two maps
                nodes_to_remove.add(access_node)
            else:  # The case where intermediate array node cannot be removed
                # Node will become an output of the second map exit
                out_e = tree.parent.edge
                conn = second_exit.next_connector()
                graph.add_edge(
                    second_exit,
                    'OUT_' + conn,
                    out_e.dst,
                    out_e.dst_conn,
                    dcpy(out_e.data),
                )
                second_exit.add_out_connector('OUT_' + conn)

                graph.add_edge(edge.src, edge.src_conn, second_exit, 'IN_' + conn, dcpy(edge.data))
                second_exit.add_in_connector('IN_' + conn)

                edges_to_remove.add(out_e)
                edges_to_remove.add(edge)

                # If the second map needs this node, link the connector
                # that generated this to the place where it is needed, with a
                # temp transient/scalar for memlet to be generated
                for out_e in graph.out_edges(second_entry):
                    second_memlet_path = graph.memlet_path(out_e)
                    source_node = second_memlet_path[0].src
                    if source_node == access_node:
                        self.fuse_nodes(sdfg, graph, edge, out_e.dst, out_e.dst_conn)

        ###
        # First scope exit is isolated and can now be safely removed
        for e in edges_to_remove:
            graph.remove_edge(e)
        graph.remove_nodes_from(nodes_to_remove)
        graph.remove_node(first_exit)

        # Isolate second_entry node
        ###########################
        for edge in graph.in_edges(second_entry):
            tree = graph.memlet_tree(edge)
            access_node = tree.root().edge.src
            if access_node in intermediate_nodes:
                # Already handled above, can be safely removed
                graph.remove_edge(edge)
                continue

            # This is an external input to the second map which will now go
            # through the first map.
            conn = first_entry.next_connector()
            graph.add_edge(edge.src, edge.src_conn, first_entry, 'IN_' + conn, dcpy(edge.data))
            first_entry.add_in_connector('IN_' + conn)
            graph.remove_edge(edge)
            for out_enode in tree.children:
                out_e = out_enode.edge
                graph.add_edge(
                    first_entry,
                    'OUT_' + conn,
                    out_e.dst,
                    out_e.dst_conn,
                    dcpy(out_e.data),
                )
                graph.remove_edge(out_e)
            first_entry.add_out_connector('OUT_' + conn)

        ###
        # Second node is isolated and can now be safely removed
        graph.remove_node(second_entry)

        # Fix scope exit to point to the right map
        second_exit.map = first_entry.map

    def fuse_nodes(self, sdfg, graph, edge, new_dst, new_dst_conn, other_edges=None):
        """ Fuses two nodes via memlets and possibly transient arrays. """
        other_edges = other_edges or []
        memlet_path = graph.memlet_path(edge)
        access_node = memlet_path[-1].dst

        local_name = "__s%d_n%d%s_n%d%s" % (
            self.state_id,
            graph.node_id(edge.src),
            edge.src_conn,
            graph.node_id(edge.dst),
            edge.dst_conn,
        )
        # Add intermediate memory between subgraphs. If a scalar,
        # uses direct connection. If an array, adds a transient node
        if edge.data.subset.num_elements() == 1:
            local_name, _ = sdfg.add_scalar(
                local_name,
                dtype=access_node.desc(graph).dtype,
                transient=True,
                storage=dtypes.StorageType.Register,
                find_new_name=True,
            )
            edge.data.data = local_name
            edge.data.subset = "0"

            # If source of edge leads to multiple destinations, redirect all through an access node.
            out_edges = list(graph.out_edges_by_connector(edge.src, edge.src_conn))
            if len(out_edges) > 1:
                local_node = graph.add_access(local_name)
                src_connector = None

                # Add edge that leads to transient node
                graph.add_edge(edge.src, edge.src_conn, local_node, None, dcpy(edge.data))

                for other_edge in out_edges:
                    if other_edge is not edge:
                        graph.remove_edge(other_edge)
                        mem = Memlet(data=local_name, other_subset=other_edge.data.dst_subset)
                        graph.add_edge(local_node, src_connector, other_edge.dst, other_edge.dst_conn, mem)
            else:
                local_node = edge.src
                src_connector = edge.src_conn

            # If destination of edge leads to multiple destinations, redirect all through an access node.
            if other_edges:
                # NOTE: If a new local node was already created, reuse it.
                if local_node == edge.src:
                    local_node_out = graph.add_access(local_name)
                    connector_out = None
                else:
                    local_node_out = local_node
                    connector_out = src_connector
                graph.add_edge(local_node, src_connector, local_node_out, connector_out,
                               Memlet.from_array(local_name, sdfg.arrays[local_name]))
                graph.add_edge(local_node_out, connector_out, new_dst, new_dst_conn, dcpy(edge.data))
                for e in other_edges:
                    graph.add_edge(local_node_out, connector_out, e.dst, e.dst_conn, dcpy(edge.data))
            else:
                # Add edge that leads to the second node
                graph.add_edge(local_node, src_connector, new_dst, new_dst_conn, dcpy(edge.data))

        else:
            local_name, _ = sdfg.add_transient(local_name,
                                               symbolic.overapproximate(edge.data.subset.size()),
                                               dtype=access_node.desc(graph).dtype,
                                               find_new_name=True)
            old_edge = dcpy(edge)
            local_node = graph.add_access(local_name)
            src_connector = None
            edge.data.data = local_name
            edge.data.subset = ",".join(["0:" + str(s) for s in edge.data.subset.size()])
            # Add edge that leads to transient node
            graph.add_edge(
                edge.src,
                edge.src_conn,
                local_node,
                None,
                dcpy(edge.data),
            )

            # Add edge that leads to the second node
            graph.add_edge(local_node, src_connector, new_dst, new_dst_conn, dcpy(edge.data))

            for e in other_edges:
                graph.add_edge(local_node, src_connector, e.dst, e.dst_conn, dcpy(edge.data))

            # Modify data and memlets on all surrounding edges to match array
            for neighbor in graph.all_edges(local_node):
                for e in graph.memlet_tree(neighbor):
                    e.data.data = local_name
                    e.data.subset.offset(old_edge.data.subset, negative=True)
