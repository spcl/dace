# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the map-dim shuffle transformation. """

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import ListProperty, make_properties


@make_properties
class MapDimShuffle(transformation.SingleStateTransformation):
    """ Implements the map-dim shuffle transformation.
    
        MapDimShuffle takes a map and a list of params.
        It reorders the dimensions in the map such that it matches the list.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    parameters = ListProperty(element_type=str, default=None, desc="Desired order of map parameters")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if self.parameters is not None:
            return (set(self.parameters) == set([str(param) for param in self.map_entry.map.params]))
        else:
            return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry

        if set(self.parameters) != set(map_entry.map.params):
            raise ValueError

        map_entry.range.ranges = [
            r for list_param in self.parameters for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param
        ]
        map_entry.map.params = self.parameters
