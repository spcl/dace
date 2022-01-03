# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the orthogonal
    tiling with overlap transformation. """

from dace import registry
from dace.properties import make_properties, ShapeProperty
from dace.transformation.dataflow import MapTiling
from dace.sdfg import nodes
from dace.symbolic import pystr_to_symbolic


@make_properties
class MapTilingWithOverlap(MapTiling):
    """ Implements the orthogonal tiling transformation with overlap.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map. The overlap can vary in each
        dimension and direction. It is added to each tile and the starting
        and end points of the outer map are adjusted to account for the overlap.
    """

    # Properties
    lower_overlap = ShapeProperty(dtype=tuple, default=None, desc="Lower overlap per dimension")
    upper_overlap = ShapeProperty(dtype=tuple, default=None, desc="Upper overlap per dimension")

    def apply(self, sdfg):
        if len(self.lower_overlap) == 0:
            return
        if len(self.upper_overlap) == 0:
            return

        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[self.map_entry]]

        # Tile the map
        self.tile_trivial = True
        super().apply(sdfg)
        tile_map_entry = graph.in_edges(map_entry)[0].src
        tile_map_exit = graph.exit_node(tile_map_entry)

        # Introduce overlap
        for lower_overlap, upper_overlap, param in zip(self.lower_overlap, self.upper_overlap, tile_map_entry.params):
            pystr = pystr_to_symbolic(param)
            lower_replace_dict = {pystr: pystr - lower_overlap}
            upper_replace_dict = {pystr: pystr + upper_overlap}

            # Extend the range of the inner map
            map_entry.range.ranges = [(r[0].subs(lower_replace_dict), r[1].subs(upper_replace_dict), r[2])
                                      for r in map_entry.range.ranges]

            # Fix the memlets
            for edge in graph.out_edges(tile_map_entry) + graph.in_edges(tile_map_exit):
                edge.data.subset.ranges = [(r[0].subs(lower_replace_dict), r[1].subs(upper_replace_dict), r[2])
                                           for r in edge.data.subset.ranges]

        # Reduce the range of the tile_map
        tile_map_entry.range.ranges = [
            (r[0] + lo, r[1] - uo, r[2])
            for r, lo, uo in zip(tile_map_entry.range.ranges, self.lower_overlap, self.upper_overlap)
        ]
