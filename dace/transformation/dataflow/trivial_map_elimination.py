# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial-map-elimination transformation. """

from dace import registry
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties


@registry.autoregister_params(singlestate=True)
@make_properties
class TrivialMapElimination(transformation.Transformation):
    """ Implements the Trivial-Map Elimination pattern.

        Trivial-Map Elimination removes all dimensions containing only one
        element from a map. If this applies to all ranges the map is removed.
        Example: Map[i=0:I,j=7] -> Map[i=0:I]
        Example: Map[i=0  ,j=7] -> nothing
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(TrivialMapElimination._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        map_entry = graph.nodes()[candidate[TrivialMapElimination._map_entry]]
        return any(r[0] == r[1] for r in map_entry.map.range)

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[TrivialMapElimination._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[
            TrivialMapElimination._map_entry]]
        map_exit = graph.exit_node(map_entry)

        remaining_ranges = []
        remaining_params = []
        for map_param, ranges in zip(map_entry.map.params,
                                     map_entry.map.range.ranges):
            map_from, map_to, _ = ranges
            if map_from == map_to:
                # Replace the map index variable with the value it obtained
                scope = graph.scope_subgraph(map_entry)
                scope.replace(map_param, map_from)
            else:
                remaining_ranges.append(ranges)
                remaining_params.append(map_param)

        map_entry.map.range.ranges = remaining_ranges
        map_entry.map.params = remaining_params

        if len(remaining_ranges) == 0:
            # Redirect map entry's out edges
            for edge in graph.out_edges(map_entry):
                path = graph.memlet_path(edge)
                index = path.index(edge)

                # Add an edge directly from the previous source connector to the destination
                graph.add_edge(path[index - 1].src, path[index - 1].src_conn,
                               edge.dst, edge.dst_conn, edge.data)

            # Redirect map exit's in edges.
            for edge in graph.in_edges(map_exit):
                path = graph.memlet_path(edge)
                index = path.index(edge)

                # Add an edge directly from the source to the next destination connector
                if len(path) > index + 1:
                    graph.add_edge(edge.src, edge.src_conn, path[index + 1].dst,
                                   path[index + 1].dst_conn, edge.data)

            # Remove map
            graph.remove_nodes_from([map_entry, map_exit])
