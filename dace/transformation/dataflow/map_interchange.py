""" Implements the map interchange transformation. """

from copy import deepcopy as dcpy
import dace
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import make_properties


@make_properties
class MapInterchange(pattern_matching.Transformation):
    """ Implements the map-interchange transformation.
    
        Map-interchange takes two nested maps and interchanges their position.
    """

    _outer_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _inner_map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(MapInterchange._outer_map_entry,
                                   MapInterchange._inner_map_entry)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # TODO: Add matching condition that the map variables are independent
        # of each other.
        # TODO: Assuming that the subsets on the edges between the two map
        # entries/exits are the union of separate inner subsets, is it possible
        # that inverting these edges breaks the continuity of union? What about
        # the opposite?

        # Check the edges between the entries of the two maps.
        outer_map_entry = graph.nodes()[candidate[
            MapInterchange._outer_map_entry]]
        inner_map_entry = graph.nodes()[candidate[
            MapInterchange._inner_map_entry]]
        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for e in graph.out_edges(outer_map_entry):
            if e.dst != inner_map_entry:
                return False
        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for e in graph.in_edges(inner_map_entry):
            if e.src != outer_map_entry:
                return False

        # Check the edges between the exits of the two maps.
        inner_map_exits = graph.exit_nodes(inner_map_entry)
        outer_map_exits = graph.exit_nodes(outer_map_entry)
        inner_map_exit = inner_map_exits[0]
        outer_map_exit = outer_map_exits[0]

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for e in graph.out_edges(inner_map_exit):
            if e.dst != outer_map_exit:
                return False
        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for e in graph.in_edges(outer_map_exit):
            if e.src != inner_map_exit:
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        outer_map_entry = graph.nodes()[candidate[
            MapInterchange._outer_map_entry]]
        inner_map_entry = graph.nodes()[candidate[
            MapInterchange._inner_map_entry]]

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [outer_map_entry, inner_map_entry])

    def apply(self, sdfg):
        # Extract the parameters and ranges of the inner/outer maps.
        graph = sdfg.nodes()[self.state_id]
        outer_map_entry = graph.nodes()[self.subgraph[
            MapInterchange._outer_map_entry]]
        inner_map_entry = graph.nodes()[self.subgraph[
            MapInterchange._inner_map_entry]]
        inner_map_exits = graph.exit_nodes(inner_map_entry)
        outer_map_exits = graph.exit_nodes(outer_map_entry)
        if len(inner_map_exits) > 1 or len(outer_map_exits) > 1:
            raise NotImplementedError('Map interchange does not work with ' +
                                      'multiple map exits')
        inner_map_exit = inner_map_exits[0]
        outer_map_exit = outer_map_exits[0]

        # Switch connectors
        outer_map_entry.in_connectors, inner_map_entry.in_connectors = \
            inner_map_entry.in_connectors, outer_map_entry.in_connectors
        outer_map_entry.out_connectors, inner_map_entry.out_connectors = \
            inner_map_entry.out_connectors, outer_map_entry.out_connectors
        outer_map_exit.in_connectors, inner_map_exit.in_connectors = \
            inner_map_exit.in_connectors, outer_map_exit.in_connectors
        outer_map_exit.out_connectors, inner_map_exit.out_connectors = \
            inner_map_exit.out_connectors, outer_map_exit.out_connectors

        # Get edges between the map entries and exits.
        entry_edges = graph.edges_between(outer_map_entry, inner_map_entry)
        exit_edges = graph.edges_between(inner_map_exit, outer_map_exit)
        for e in entry_edges + exit_edges:
            graph.remove_edge(e)

        # Change source and destination of edges.
        dace.graph.nxutil.change_edge_dest(graph, outer_map_entry,
                                           inner_map_entry)
        dace.graph.nxutil.change_edge_src(graph, inner_map_entry,
                                          outer_map_entry)
        dace.graph.nxutil.change_edge_dest(graph, inner_map_exit,
                                           outer_map_exit)
        dace.graph.nxutil.change_edge_src(graph, outer_map_exit,
                                          inner_map_exit)

        # Add edges between the map entries and exits.
        for e in entry_edges + exit_edges:
            graph.add_edge(e.dst, e.src_conn, e.src, e.dst_conn, e.data)


pattern_matching.Transformation.register_pattern(MapInterchange)
