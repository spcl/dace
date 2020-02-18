""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic
from dace.graph import nodes, nxutil
from dace.sdfg import replace
from dace.transformation import pattern_matching
from typing import List, Union
import networkx as nx


@registry.autoregister_params(singlestate=True)
class MapFusion(pattern_matching.Transformation):
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
    _first_map_exit = nodes.ExitNode()
    _some_array = nodes.AccessNode("_")
    _second_map_entry = nodes.EntryNode()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(
                MapFusion._first_map_exit,
                MapFusion._some_array,
                MapFusion._second_map_entry,
            )
        ]

    @staticmethod
    def find_permutation(first_map: nodes.Map,
                         second_map: nodes.Map) -> Union[List[int], None]:
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

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        first_map_entry = graph.entry_node(first_map_exit)
        second_map_entry = graph.nodes()[candidate[
            MapFusion._second_map_entry]]

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
                    n for n in graph.nodes()
                    if isinstance(n, nodes.AccessNode) and n.data == dst.data
                ])
                if num_occurrences > 1:
                    return False
            else:
                return False
        # Check map ranges
        perm = MapFusion.find_permutation(first_map_entry.map,
                                          second_map_entry.map)
        if perm is None:
            return False

        # Create a dict that maps parameters of the first map to those of the
        # second map.
        params_dict = {}
        for _index, _param in enumerate(first_map_entry.map.params):
            params_dict[_param] = second_map_entry.map.params[perm[_index]]

        out_memlets = [e.data for e in graph.in_edges(first_map_exit)]

        # Check that input set of second map is provided by the output set
        # of the first map, or other unrelated maps
        for _, _, _, _, second_memlet in graph.out_edges(second_map_entry):
            # Memlets that do not come from one of the intermediate arrays
            if second_memlet.data not in intermediate_data:
                # however, if intermediate_data eventually leads to
                # second_memlet.data, need to fail.
                for _n in intermediate_nodes:
                    source_node = _n  # graph.find_node(_n.data)
                    destination_node = graph.find_node(second_memlet.data)
                    # NOTE: Assumes graph has networkx version
                    if destination_node in nx.descendants(
                            graph._nx, source_node):
                        return False
                continue

            provided = False
            for first_memlet in out_memlets:
                if first_memlet.data != second_memlet.data:
                    continue
                # If there is an equivalent subset, it is provided
                expected_second_subset = []
                for _tup in first_memlet.subset:
                    new_tuple = []
                    if isinstance(_tup, symbolic.symbol):
                        new_tuple = symbolic.symbol(params_dict[str(_tup)])
                    elif isinstance(_tup, (list, tuple)):
                        for _sym in _tup:
                            if (isinstance(_sym, symbolic.symbol)
                                    and str(_sym) in params_dict):
                                new_tuple.append(
                                    symbolic.symbol(params_dict[str(_sym)]))
                            else:
                                new_tuple.append(_sym)
                        new_tuple = tuple(new_tuple)
                    else:
                        new_tuple = _tup
                    expected_second_subset.append(new_tuple)
                if expected_second_subset == list(second_memlet.subset):
                    provided = True
                    break

            # If none of the output memlets of the first map provide the info,
            # fail.
            if provided is False:
                return False

        # Success
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        second_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]

        return " -> ".join(entry.map.label + ": " + str(entry.map.params)
                           for entry in [first_exit, second_entry])

    def apply(self, sdfg):
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
        graph = sdfg.nodes()[self.state_id]
        first_exit = graph.nodes()[self.subgraph[MapFusion._first_map_exit]]
        first_entry = graph.entry_node(first_exit)
        second_entry = graph.nodes()[self.subgraph[
            MapFusion._second_map_entry]]
        second_exit = graph.exit_nodes(second_entry)[0]

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
        perm = MapFusion.find_permutation(first_entry.map, second_entry.map)
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
            memlet_path = graph.memlet_path(edge)
            edge_index = next(
                i for i, e in enumerate(memlet_path) if e == edge)
            access_node = memlet_path[-1].dst
            if access_node not in do_not_erase:
                out_edges = [
                    e for e in graph.out_edges(access_node)
                    if e.dst == second_entry
                ]
                # In this transformation, there can only be one edge to the
                # second map
                assert len(out_edges) == 1
                # Get source connector to the second map
                connector = out_edges[0].dst_conn[3:]

                new_dst = None
                new_dst_conn = None
                # Look at the second map entry out-edges to get the new
                # destination
                for _e in graph.out_edges(second_entry):
                    if _e.src_conn[4:] == connector:
                        new_dst = _e.dst
                        new_dst_conn = _e.dst_conn
                        break
                if new_dst is None:
                    # Access node is not used in the second map
                    nodes_to_remove.add(access_node)
                    continue
                # If the source is an access node, modify the memlet to point
                # to it
                if (isinstance(edge.src, nodes.AccessNode)
                        and edge.data.data != edge.src.data):
                    edge.data.data = edge.src.data
                    edge.data.subset = ("0" if edge.data.other_subset is None
                                        else edge.data.other_subset)
                    edge.data.other_subset = None

                else:
                    # Add a transient scalar/array
                    self.fuse_nodes(sdfg, graph, edge, new_dst, new_dst_conn)

                edges_to_remove.add(edge)

                # Remove transient node between the two maps
                nodes_to_remove.add(access_node)
            else:  # The case where intermediate array node cannot be removed
                # Node will become an output of the second map exit
                out_e = memlet_path[edge_index + 1]
                conn = second_exit.next_connector()
                graph.add_edge(
                    second_exit,
                    'OUT_' + conn,
                    out_e.dst,
                    out_e.dst_conn,
                    dcpy(out_e.data),
                )
                second_exit.add_out_connector('OUT_' + conn)

                graph.add_edge(edge.src, edge.src_conn, second_exit,
                               'IN_' + conn, dcpy(edge.data))
                second_exit.add_in_connector('IN_' + conn)

                edges_to_remove.add(out_e)

                # If the second map needs this node, link the connector
                # that generated this to the place where it is needed, with a
                # temp transient/scalar for memlet to be generated
                for out_e in graph.out_edges(second_entry):
                    second_memlet_path = graph.memlet_path(out_e)
                    source_node = second_memlet_path[0].src
                    if source_node == access_node:
                        self.fuse_nodes(sdfg, graph, edge, out_e.dst,
                                        out_e.dst_conn)

                edges_to_remove.add(edge)
        ###
        # First scope exit is isolated and can now be safely removed
        for e in edges_to_remove:
            graph.remove_edge(e)
        graph.remove_nodes_from(nodes_to_remove)
        graph.remove_node(first_exit)

        # Isolate second_entry node
        ###########################
        for edge in graph.in_edges(second_entry):
            memlet_path = graph.memlet_path(edge)
            edge_index = next(
                i for i, e in enumerate(memlet_path) if e == edge)
            access_node = memlet_path[0].src
            if access_node in intermediate_nodes:
                # Already handled above, can be safely removed
                graph.remove_edge(edge)
                continue

            # This is an external input to the second map which will now go
            # through the first map.
            conn = first_entry.next_connector()
            graph.add_edge(edge.src, edge.src_conn, first_entry, 'IN_' + conn,
                           dcpy(edge.data))
            first_entry.add_in_connector('IN_' + conn)
            graph.remove_edge(edge)
            out_e = memlet_path[edge_index + 1]
            graph.add_edge(
                first_entry,
                'OUT_' + conn,
                out_e.dst,
                out_e.dst_conn,
                dcpy(out_e.data),
            )
            first_entry.add_out_connector('OUT_' + conn)

            graph.remove_edge(out_e)
        ###
        # Second node is isolated and can now be safely removed
        graph.remove_node(second_entry)

        # Fix scope exit to point to the right map
        second_exit.map = first_entry.map

    def fuse_nodes(self, sdfg, graph, edge, new_dst, new_dst_conn):
        """ Fuses two nodes via memlets and possibly transient arrays. """
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
            sdfg.add_scalar(
                local_name,
                dtype=access_node.desc(graph).dtype,
                transient=True,
                storage=dtypes.StorageType.Register,
            )
            edge.data.data = local_name
            edge.data.subset = "0"
            local_node = edge.src
            src_connector = edge.src_conn
        else:
            sdfg.add_transient(
                local_name,
                edge.data.subset.size(),
                dtype=access_node.desc(graph).dtype)
            local_node = graph.add_access(local_name)
            src_connector = None
            edge.data.data = local_name
            edge.data.subset = ",".join(
                ["0:" + str(s) for s in edge.data.subset.size()])
            # Add edge that leads to transient node
            graph.add_edge(
                edge.src,
                edge.src_conn,
                local_node,
                None,
                dcpy(edge.data),
            )
        ########
        # Add edge that leads to the second node
        graph.add_edge(local_node, src_connector, new_dst, new_dst_conn,
                       dcpy(edge.data))
