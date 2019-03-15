""" This module contains classes that implement a map->for loop transformation.
"""

import dace
from copy import deepcopy as dcpy
from dace import data, symbolic, types, subsets
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching
from math import ceil
import sympy
import networkx as nx


class MapToForLoop(pattern_matching.Transformation):
    """ Implements the Map to for-loop transformation.

        Takes a map and enforces a sequential schedule by transforming it into
        a state-machine of a for-loop. Creates a nested SDFG, if necessary.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapToForLoop._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # Only uni-dimensional maps are accepted.
        map_entry = graph.nodes()[candidate[MapToForLoop._map_entry]]
        if len(map_entry.map.params) > 1:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MapToForLoop._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        # Retrieve map entry and exit nodes.
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[MapToForLoop._map_entry]]
        map_exits = graph.exit_nodes(map_entry)
        loop_idx = map_entry.map.params[0]
        loop_from, loop_to, loop_step = map_entry.map.range[0]

        nested_sdfg = dace.SDFG(graph.label + '_' + map_entry.map.label)

        # Construct nested SDFG
        begin = nested_sdfg.add_state('begin')
        guard = nested_sdfg.add_state('guard')
        body = nested_sdfg.add_state('body')
        end = nested_sdfg.add_state('end')

        nested_sdfg.add_edge(
            begin,
            guard,
            edges.InterstateEdge(assignments={str(loop_idx): str(loop_from)}))
        nested_sdfg.add_edge(
            guard,
            body,
            edges.InterstateEdge(condition = str(loop_idx) + ' <= ' + \
                                             str(loop_to))
        )
        nested_sdfg.add_edge(
            guard,
            end,
            edges.InterstateEdge(condition = str(loop_idx) + ' > ' + \
                                             str(loop_to))
        )
        nested_sdfg.add_edge(
            body,
            guard,
            edges.InterstateEdge(assignments = {str(loop_idx): str(loop_idx) + \
                                                ' + ' +str(loop_step)})
        )

        # Add map contents
        map_subgraph = graph.scope_subgraph(map_entry)
        for node in map_subgraph.nodes():
            if node is not map_entry and node not in map_exits:
                body.add_node(node)
        for src, src_conn, dst, dst_conn, memlet in map_subgraph.edges():
            if src is not map_entry and dst not in map_exits:
                body.add_edge(src, src_conn, dst, dst_conn, memlet)

        # Reconnect inputs
        nested_in_data_nodes = {}
        nested_in_connectors = {}
        nested_in_memlets = {}
        for i, edge in enumerate(graph.in_edges(map_entry)):
            src, src_conn, dst, dst_conn, memlet = edge
            data_label = '_in_' + memlet.data
            memdata = sdfg.arrays[memlet.data]
            if isinstance(memdata, data.Array):
                data_array = sdfg.add_array(data_label, memdata.dtype, [
                    symbolic.overapproximate(r)
                    for r in memlet.bounding_box_size()
                ])
            elif isinstance(memdata, data.Scalar):
                data_array = sdfg.add_scalar(data_label, memdata.dtype)
            else:
                raise NotImplementedError()
            data_node = nodes.AccessNode(data_label)
            body.add_node(data_node)
            nested_in_data_nodes.update({i: data_node})
            nested_in_connectors.update({i: data_label})
            nested_in_memlets.update({i: memlet})
            for _, _, _, _, old_memlet in body.edges():
                if old_memlet.data == memlet.data:
                    old_memlet.data = data_label
            #body.add_edge(data_node, None, dst, dst_conn, memlet)

        # Reconnect outputs
        nested_out_data_nodes = {}
        nested_out_connectors = {}
        nested_out_memlets = {}
        for map_exit in map_exits:
            for i, edge in enumerate(graph.out_edges(map_exit)):
                src, src_conn, dst, dst_conn, memlet = edge
                data_label = '_out_' + memlet.data
                memdata = sdfg.arrays[memlet.data]
                if isinstance(memdata, data.Array):
                    data_array = sdfg.add_array(data_label, memdata.dtype, [
                        symbolic.overapproximate(r)
                        for r in memlet.bounding_box_size()
                    ])
                elif isinstance(memdata, data.Scalar):
                    data_array = sdfg.add_scalar(data_label, memdata.dtype)
                else:
                    raise NotImplementedError()
                data_node = nodes.AccessNode(data_label)
                body.add_node(data_node)
                nested_out_data_nodes.update({i: data_node})
                nested_out_connectors.update({i: data_label})
                nested_out_memlets.update({i: memlet})
                for _, _, _, _, old_memlet in body.edges():
                    if old_memlet.data == memlet.data:
                        old_memlet.data = data_label
                #body.add_edge(src, src_conn, data_node, None, memlet)

        # Add nested SDFG and reconnect it
        nested_node = graph.add_nested_sdfg(
            nested_sdfg, sdfg, set(nested_in_connectors.values()),
            set(nested_out_connectors.values()))

        for i, edge in enumerate(graph.in_edges(map_entry)):
            src, src_conn, dst, dst_conn, memlet = edge
            graph.add_edge(src, src_conn, nested_node, nested_in_connectors[i],
                           nested_in_memlets[i])

        for map_exit in map_exits:
            for i, edge in enumerate(graph.out_edges(map_exit)):
                src, src_conn, dst, dst_conn, memlet = edge
                graph.add_edge(nested_node, nested_out_connectors[i], dst,
                               dst_conn, nested_out_memlets[i])

        for src, src_conn, dst, dst_conn, memlet in graph.out_edges(map_entry):
            i = int(src_conn[4:]) - 1
            new_memlet = dcpy(memlet)
            new_memlet.data = nested_in_data_nodes[i].data
            body.add_edge(nested_in_data_nodes[i], None, dst, dst_conn,
                          new_memlet)

        for map_exit in map_exits:
            for src, src_conn, dst, dst_conn, memlet in graph.in_edges(
                    map_exit):
                i = int(dst_conn[3:]) - 1
                new_memlet = dcpy(memlet)
                new_memlet.data = nested_out_data_nodes[i].data
                body.add_edge(src, src_conn, nested_out_data_nodes[i], None,
                              new_memlet)

        for node in map_subgraph:
            graph.remove_node(node)


pattern_matching.Transformation.register_pattern(MapToForLoop)
