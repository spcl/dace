# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the orthogonal
    tiling transformation. """

from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@registry.autoregister_params(singlestate=True)
@make_properties
class MapTiling(transformation.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

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

    tile_offset = ShapeProperty(dtype=tuple,
                                default=None,
                                desc="Negative Stride offset per dimension",
                                allow_none=True)

    divides_evenly = Property(dtype=bool,
                              default=False,
                              desc="Tile size divides dimension length evenly")
    tile_trivial = Property(dtype=bool,
                              default=False,
                              desc="Tiles even if tile_size is 1")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(MapTiling.map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MapTiling.map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        tile_strides = self.tile_sizes
        if self.strides is not None and len(self.strides) == len(tile_strides):
            tile_strides = self.strides

        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[self.subgraph[MapTiling.map_entry]]
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining
        stripmine_subgraph = {
            StripMining._map_entry: self.subgraph[MapTiling.map_entry]
        }
        sdfg_id = sdfg.sdfg_id
        last_map_entry = None
        removed_maps = 0

        original_schedule = map_entry.schedule

        for dim_idx in range(len(map_entry.map.params)):
            if dim_idx >= len(self.tile_sizes):
                tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
            else:
                tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[dim_idx])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

            # handle offsets
            if self.tile_offset and dim_idx >= len(self.tile_offset):
                offset = self.tile_offset[-1]
            elif self.tile_offset:
                offset = self.tile_offset[dim_idx]
            else:
                offset = 0

            dim_idx -= removed_maps
            # If tile size is trivial, skip strip-mining map dimension
            if tile_size == map_entry.map.range.size()[dim_idx]:
                continue

            stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                    self.expr_index)

            # Special case: Tile size of 1 should be omitted from inner map
            if tile_size == 1 and tile_stride == 1 and self.tile_trivial == False:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = ''
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = True
                stripmine.tile_offset = str(offset)
                stripmine.apply(sdfg)
                removed_maps += 1
            else:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = self.divides_evenly
                stripmine.tile_offset = str(offset)
                stripmine.apply(sdfg)

            # apply to the new map the schedule of the original one
            map_entry.schedule = original_schedule

            if last_map_entry:
                new_map_entry = graph.in_edges(map_entry)[0].src
                mapcollapse_subgraph = {
                    MapCollapse._outer_map_entry: graph.node_id(last_map_entry),
                    MapCollapse._inner_map_entry: graph.node_id(new_map_entry)
                }
                mapcollapse = MapCollapse(sdfg_id, self.state_id,
                                          mapcollapse_subgraph, 0)
                mapcollapse.apply(sdfg)
            last_map_entry = graph.in_edges(map_entry)[0].src
        return last_map_entry
