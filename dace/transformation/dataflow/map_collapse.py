""" Contains classes that implement the map-collapse transformation. """

from copy import deepcopy as dcpy
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import make_properties


@make_properties
class MapCollapse(pattern_matching.Transformation):
    """ Implements the Map Collapse pattern.

        Map-collapse takes two nested maps with M and N dimensions respectively,
        and collapses them to a single M+N dimensional map.
    """

    _outer_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _inner_map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(
                MapCollapse._outer_map_entry,
                MapCollapse._inner_map_entry,
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # Check the edges between the entries of the two maps.
        outer_map_entry = graph.nodes()[candidate[
            MapCollapse._outer_map_entry]]
        inner_map_entry = graph.nodes()[candidate[
            MapCollapse._inner_map_entry]]

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for _src, _, dest, _, _ in graph.out_edges(outer_map_entry):
            if dest != inner_map_entry:
                return False

        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for src, _, _dest, _, _ in graph.in_edges(inner_map_entry):
            if src != outer_map_entry:
                return False

        # Check the edges between the exits of the two maps.
        inner_map_exits = graph.exit_nodes(inner_map_entry)
        outer_map_exits = graph.exit_nodes(outer_map_entry)
        if len(inner_map_exits) > 1 or len(outer_map_exits) > 1:
            return False

        inner_map_exit = inner_map_exits[0]
        outer_map_exit = outer_map_exits[0]

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for _src, _, dest, _, _ in graph.out_edges(inner_map_exit):
            if dest != outer_map_exit:
                return False

        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for src, _, _dest, _, _ in graph.in_edges(outer_map_exit):
            if src != inner_map_exit:
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        outer_map_entry = graph.nodes()[candidate[
            MapCollapse._outer_map_entry]]
        inner_map_entry = graph.nodes()[candidate[
            MapCollapse._inner_map_entry]]

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [outer_map_entry, inner_map_entry])

    def apply(self, sdfg):
        # Extract the parameters and ranges of the inner/outer maps.
        graph = sdfg.nodes()[self.state_id]
        outer_map_entry = graph.nodes()[self.subgraph[
            MapCollapse._outer_map_entry]]
        inner_map_entry = graph.nodes()[self.subgraph[
            MapCollapse._inner_map_entry]]
        inner_map_exit = graph.exit_nodes(inner_map_entry)[0]
        outer_map_exit = graph.exit_nodes(outer_map_entry)[0]

        nxutil.merge_maps(graph, outer_map_entry, outer_map_exit,
                          inner_map_entry, inner_map_exit)

        return


pattern_matching.Transformation.register_pattern(MapCollapse)
