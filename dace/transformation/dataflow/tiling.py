""" This module contains classes and functions that implement the orthogonal
    tiling transformation. """

from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


@registry.autoregister_params(singlestate=True)
@make_properties
class MapTiling(pattern_matching.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    prefix = Property(dtype=str,
                      default="tile",
                      desc="Prefix for new range symbols")
    tile_sizes = ShapeProperty(dtype=tuple,
                               default=(128, 128, 128),
                               desc="Tile size per dimension")
    strides = ShapeProperty(
        dtype=tuple,
        default=tuple(),
        desc="Tile stride (enables overlapping tiles). If empty, matches tile")
    divides_evenly = Property(dtype=bool,
                              default=False,
                              desc="Tile size divides dimension length evenly")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapTiling._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MapTiling._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        tile_strides = self.tile_sizes
        if self.strides is not None and len(self.strides) == len(tile_strides):
            tile_strides = self.strides

        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[self.subgraph[MapTiling._map_entry]]
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining
        stripmine_subgraph = {
            StripMining._map_entry: self.subgraph[MapTiling._map_entry]
        }
        sdfg_id = sdfg.sdfg_list.index(sdfg)
        last_map_entry = None
        removed_maps = 0
        for dim_idx in range(len(map_entry.map.params)):
            if dim_idx >= len(self.tile_sizes):
                tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
            else:
                tile_size = symbolic.pystr_to_symbolic(
                    self.tile_sizes[dim_idx])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

            dim_idx -= removed_maps
            # If tile size is trivial, skip strip-mining map dimension
            if tile_size == map_entry.map.range.size()[dim_idx]:
                continue

            stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                    self.expr_index)

            # Special case: Tile size of 1 should be omitted from inner map
            if tile_size == 1 and tile_stride == 1:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = ''
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = True
                stripmine.apply(sdfg)
                removed_maps += 1
            else:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = self.divides_evenly
                stripmine.apply(sdfg)

            if last_map_entry:
                new_map_entry = graph.in_edges(map_entry)[0].src
                mapcollapse_subgraph = {
                    MapCollapse._outer_map_entry:
                    graph.node_id(last_map_entry),
                    MapCollapse._inner_map_entry: graph.node_id(new_map_entry)
                }
                mapcollapse = MapCollapse(sdfg_id, self.state_id,
                                          mapcollapse_subgraph, 0)
                mapcollapse.apply(sdfg)
            last_map_entry = graph.in_edges(map_entry)[0].src
