""" Contains an implementation of the map-dimension-interchange transformation.
"""

from dace.graph import nodes, nxutil
from dace.properties import ShapeProperty
from dace.transformation import pattern_matching as pm


@make_properties
class MapDimInterchange(pm.Transformation):
    """ Implements the map-dimension-interchange pattern.

        Map-dimension-interchange re-orders the dimensions of a map.
    """

    _map_entry = nodes.MapEntry(None)

    order = ShapeProperty()

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapDimInterchange._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        """ A candidate subgraph matches the map-dimension-interchange 
            transformation when a map has at least two dimensions.
        """
        map_entry = graph.nodes()[candidate[MapDimInterchange._map_entry]]
        return map_entry.map.get_param_num() > 1

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = candidate[MapDimInterchange._map_entry]

        return str(map_entry)

    def apply(self, sdfg):
        """ Reorders the dimensions of the map by reordering the
            parameters and the range of the map as specified through the 
            properties.
        """

        # Extract the map and its entry node.
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[MapDimInterchange._map_entry]]
        current_map = map_entry.map

        order = self.order
        if len(self.order) != current_map.get_param_num():
            # 'order' must be of the same length as the number of map
            # dimensions.
            return

        # Re-order the map dimensions
        current_map.params = [current_map.params[idx] for idx in order]
        current_map.range.reorder(order)

        return

    def __init__(self, *args, **kwargs):
        self.entry = nodes.EntryNode()
        self.tasklet = nodes.Tasklet('_')
        self.exit = nodes.ExitNode()
        self.pairs = None
        super().__init__(*args, **kwargs)

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(MapDimInterchange)
