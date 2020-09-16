# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial map range elimination transformation. """

from dace import registry
from dace.symbolic import symlist
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pattern_matching
from dace.properties import make_properties
from typing import Tuple


@registry.autoregister_params(singlestate=True)
@make_properties
class TrivialMapRangeElimination(pattern_matching.Transformation):
    """ Implements the Trivial Map Range Elimination pattern.

        Trivial Map Range Elimination takes a multi-dimensional map with 
        a range containing one element and removes the corresponding dimension.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                TrivialMapRangeElimination._map_entry
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[TrivialMapRangeElimination._map_entry]]
        if len(map_entry.map.range) <= 1:
            return False # only acts on multi-dimensional maps
        return any(frm == to for frm, to, _ in map_entry.map.range)

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[TrivialMapRangeElimination._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[TrivialMapRangeElimination._map_entry]]
        map_exit = graph.exit_node(map_entry)

        trivials = [i for i, range in enumerate(map_entry.map.range.ranges) if range[0] == range[1]] # where "from == to".

        for i in trivials:
            map_idx = map_entry.map.params[i]
            map_from, _, _ = map_entry.map.range[i]
            graph.replace(map_idx, map_from)

        map_entry.map.range.ranges = [range for i, range in enumerate(map_entry.map.range.ranges) if i not in trivials]
        map_entry.map.params = [p for i, p in enumerate(map_entry.map.params) if i not in trivials]
