# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the map-dim shuffle transformation. """

from dace import registry
from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, ShapeProperty


@registry.autoregister_params(singlestate=True)
@make_properties
class MapDimShuffle(transformation.Transformation):
    """ Implements the map-dim shuffle transformation.
    
        MapDimShuffle takes a map and a list of params.
        It reorders the dimensions in the map such that it matches the list.
    """

    _map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    parameters = ShapeProperty(dtype=list,
                            default=None,
                            desc="Desired order of map parameters")

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(MapDimShuffle._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MapDimShuffle._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: SDFG):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[self._map_entry]]

        if set(self.parameters) != set(map_entry.map.params):
            return
        
        map_entry.range.ranges = [r
            for list_param in self.parameters
            for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param]
        map_entry.map.params = self.parameters
