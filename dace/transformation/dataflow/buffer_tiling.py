# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the BufferTiling transformation. """

from dace import registry
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.properties import ShapeProperty, make_properties
from dace.transformation import transformation
from dace.transformation.dataflow import MapTiling, MapTilingWithOverlap, MapFusion, TrivialMapElimination


@registry.autoregister_params(singlestate=True)
@make_properties
class BufferTiling(transformation.Transformation):
    """ Implements the buffer tiling transformation.

        BufferTiling tiles a buffer that is in between two maps, where the preceding map
        writes to the buffer and the succeeding map reads from it.
        It introduces additional computations in exchange for reduced memory footprint.
        Commonly used to make use of shared memory on GPUs.
    """

    _map1_exit = nodes.MapExit(nodes.Map('', [], []))
    _array = nodes.AccessNode('')
    _map2_entry = nodes.MapEntry(nodes.Map('', [], []))

    tile_sizes = ShapeProperty(dtype=tuple, default=(128, 128, 128), desc="Tile size per dimension")

    # Returns a list of graphs that represent the pattern
    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(
            BufferTiling._map1_exit,
            BufferTiling._array,
            BufferTiling._map2_entry,
        )]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        map1_exit = graph.nodes()[candidate[BufferTiling._map1_exit]]
        map2_entry = graph.nodes()[candidate[BufferTiling._map2_entry]]

        for buf in graph.all_nodes_between(map1_exit, map2_entry):
            # Check that buffers are AccessNodes.
            if not isinstance(buf, nodes.AccessNode):
                return False

            # Check that buffers are transient.
            if not sdfg.arrays[buf.data].transient:
                return False

            # Check that buffers have exactly 1 input and 1 output edge.
            if graph.in_degree(buf) != 1:
                return False
            if graph.out_degree(buf) != 1:
                return False

            # Check that buffers are next to the maps.
            if graph.in_edges(buf)[0].src != map1_exit:
                return False
            if graph.out_edges(buf)[0].dst != map2_entry:
                return False

            # Check that the data consumed is provided.
            provided = graph.in_edges(buf)[0].data.subset
            consumed = graph.out_edges(buf)[0].data.subset
            if not provided.covers(consumed):
                return False

            # Check that buffers occur only once in this state.
            num_occurrences = len([n for n in graph.nodes() if isinstance(n, nodes.AccessNode) and n.data == buf])
            if num_occurrences > 1:
                return False
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map1_exit = graph.nodes()[candidate[BufferTiling._map1_exit]]
        map2_entry = graph.nodes()[candidate[BufferTiling._map2_entry]]

        return " -> ".join(entry.map.label + ": " + str(entry.map.params) for entry in [map1_exit, map2_entry])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map1_exit = graph.nodes()[self.subgraph[self._map1_exit]]
        map1_entry = graph.entry_node(map1_exit)
        map2_entry = graph.nodes()[self.subgraph[self._map2_entry]]
        buffers = graph.all_nodes_between(map1_exit, map2_entry)
        # Situation:
        # -> map1_entry -> ... -> map1_exit -> buffers -> map2_entry -> ...

        lower_extents = tuple(b - a for a, b in zip(map1_entry.range.min_element(), map2_entry.range.min_element()))
        upper_extents = tuple(a - b for a, b in zip(map1_entry.range.max_element(), map2_entry.range.max_element()))

        # Tile the first map with overlap
        MapTilingWithOverlap.apply_to(sdfg,
                                      map_entry=map1_entry,
                                      options={
                                          'tile_sizes': self.tile_sizes,
                                          'lower_overlap': lower_extents,
                                          'upper_overlap': upper_extents
                                      })
        tile_map1_exit = graph.out_edges(map1_exit)[0].dst
        tile_map1_entry = graph.entry_node(tile_map1_exit)
        tile_map1_entry.label = 'BufferTiling'

        # Tile the second map
        MapTiling.apply_to(sdfg, map_entry=map2_entry, options={'tile_sizes': self.tile_sizes, 'tile_trivial': True})
        tile_map2_entry = graph.in_edges(map2_entry)[0].src

        # Fuse maps
        some_buffer = next(iter(buffers))  # some dummy to pass to MapFusion.apply_to()
        MapFusion.apply_to(sdfg, first_map_exit=tile_map1_exit, array=some_buffer, second_map_entry=tile_map2_entry)

        # Optimize the simple cases
        map1_entry.range.ranges = [
            (r[0], r[0], r[2]) if l_ext == 0 and u_ext == 0 and ts == 1 else r
            for r, l_ext, u_ext, ts in zip(map1_entry.range.ranges, lower_extents, upper_extents, self.tile_sizes)
        ]

        map2_entry.range.ranges = [(r[0], r[0], r[2]) if ts == 1 else r
                                   for r, ts in zip(map2_entry.range.ranges, self.tile_sizes)]

        if any(ts == 1 for ts in self.tile_sizes):
            if any(r[0] == r[1] for r in map1_entry.map.range):
                TrivialMapElimination.apply_to(sdfg, _map_entry=map1_entry)
            if any(r[0] == r[1] for r in map2_entry.map.range):
                TrivialMapElimination.apply_to(sdfg, _map_entry=map2_entry)
