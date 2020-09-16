# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial-map-elimination transformation. """

from dace import registry
from dace.symbolic import symlist
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pattern_matching
from dace.properties import make_properties
from typing import Tuple


@registry.autoregister_params(singlestate=True)
@make_properties
class TrivialMapElimination(pattern_matching.Transformation):
    """ Implements the Trivial-Map Elimination pattern.

        Trivial-Map Elimination takes a map with a range containing one element and removes the map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                TrivialMapElimination._map_entry
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[TrivialMapElimination._map_entry]]
        map_from, map_to, map_step = map_entry.map.range[0]

        return len(map_entry.map.range) == 1 and map_to == map_from

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[TrivialMapElimination._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[TrivialMapElimination._map_entry]]
        map_exit = graph.exit_node(map_entry)

        map_idx = map_entry.map.params[0]
        map_from, _, _ = map_entry.map.range[0]

        graph.replace(map_idx, map_from)

        # Redirect map entry's out edges.
        for edge in graph.out_edges(map_entry):
            path = graph.memlet_path(edge)
            ind = path.index(edge)

            # Add an edge directly from the previous source connector to the
            # destination
            graph.add_edge(path[ind - 1].src, path[ind - 1].src_conn,
                        edge.dst, edge.dst_conn, edge.data)

        # Redirect map exit's in edges.
        for edge in graph.in_edges(map_exit):
            path = graph.memlet_path(edge)
            ind = path.index(edge)

            # Add an edge directly from the source to the next destination
            # connector
            graph.add_edge(edge.src, edge.src_conn,
                        path[ind + 1].dst, path[ind + 1].dst_conn, edge.data)
        
        # Clean-up
        graph.remove_nodes_from([map_entry, map_exit])
