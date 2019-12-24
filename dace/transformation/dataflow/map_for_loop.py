""" This module contains classes that implement a map->for loop transformation.
"""

import dace
from copy import deepcopy as dcpy
from dace import data, symbolic
from dace.sdfg import SDFG, SDFGState
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching
from dace.transformation.helpers import nest_state_subgraph


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
        map_exit = graph.exit_nodes(map_entry)[0]

        loop_idx = map_entry.map.params[0]
        loop_from, loop_to, loop_step = map_entry.map.range[0]

        # Turn the map scope into a nested SDFG
        node = nest_state_subgraph(sdfg, graph,
                                   graph.scope_subgraph(map_entry))

        # Create a loop inside the nested SDFG
        nsdfg: SDFG = node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]
        nsdfg.add_loop(None, nstate, None, loop_idx,
                       symbolic.symstr(loop_from),
                       '%s < %s' % (loop_idx, symbolic.symstr(loop_to + 1)),
                       '%s + %s' % (loop_idx, symbolic.symstr(loop_step)))

        # Skip map in input edges
        for edge in nstate.out_edges(map_entry):
            src_node = nstate.memlet_path(edge)[0].src
            nstate.add_edge(src_node, None, edge.dst, edge.dst_conn, edge.data)
            nstate.remove_edge(edge)

        # Skip map in output edges
        for edge in nstate.in_edges(map_exit):
            dst_node = nstate.memlet_path(edge)[-1].dst
            nstate.add_edge(edge.src, edge.src_conn, dst_node, None, edge.data)
            nstate.remove_edge(edge)

        # Remove nodes from dynamic map range
        nstate.remove_nodes_from([
            e.src for e in nstate.in_edges(map_entry)
            if not e.dst_conn.startswith('IN_')
        ])
        # Remove scope nodes
        nstate.remove_nodes_from([map_entry, map_exit])


pattern_matching.Transformation.register_pattern(MapToForLoop)
