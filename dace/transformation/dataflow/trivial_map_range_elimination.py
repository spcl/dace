# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial map range elimination transformation. """

from dace import registry
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class TrivialMapRangeElimination(transformation.SingleStateTransformation):
    """ Implements the Trivial Map Range Elimination pattern.

        Trivial Map Range Elimination takes a multi-dimensional map with 
        a range containing one element and removes the corresponding dimension.
        Example: Map[i=0:I,j=0] -> Map[i=0:I]
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        if len(map_entry.map.range) <= 1:
            return False  # only acts on multi-dimensional maps
        return any(frm == to for frm, to, _ in map_entry.map.range)

    def apply(self, graph, sdfg):
        map_entry = self.map_entry

        remaining_ranges = []
        remaining_params = []
        for map_param, ranges in zip(map_entry.map.params, map_entry.map.range.ranges):
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
