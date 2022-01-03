# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial map range elimination transformation. """

from dace import registry
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties


@registry.autoregister_params(singlestate=True)
@make_properties
class TrivialMapRangeElimination(transformation.Transformation):
    """ Implements the Trivial Map Range Elimination pattern.

        Trivial Map Range Elimination takes a multi-dimensional map with 
        a range containing one element and removes the corresponding dimension.
        Example: Map[i=0:I,j=0] -> Map[i=0:I]
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(TrivialMapRangeElimination._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        map_entry = graph.nodes()[candidate[
            TrivialMapRangeElimination._map_entry]]
        if len(map_entry.map.range) <= 1:
            return False  # only acts on multi-dimensional maps
        return any(frm == to for frm, to, _ in map_entry.map.range)

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[
            TrivialMapRangeElimination._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[
            TrivialMapRangeElimination._map_entry]]

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
