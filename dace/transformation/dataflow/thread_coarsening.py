# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
from dace.transformation.dataflow.change_thread_block_map import ChangeThreadBlockMap
from dace import subsets
from typing import List

@make_properties
class ThreadCoarsening(transformation.SingleStateTransformation):
    """
    Thread coarsening means for GPU code-gen one thread does not comute 1 cell of the output, but
    a tile_size_x * tile_size_y * tile_size_t sub domain of the output.
    """

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    tile_size_x = SymbolicProperty(dtype=int, default=4, desc="Number threads in the threadBlock X Dim")
    tile_size_y = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Y Dim")
    tile_size_z = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Z Dim")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.device_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        thread_block_entry = self.thread_block_map_entry

        # Applicable if the map is a GPU_ThreadBlock Scheduled Map
        if thread_block_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False

        return MapTiling.can_be_applied(self, graph, expr_index=expr_index, sdfg=sdfg, permissive=permissive)

    def update_names():
        pass

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # When the ThreadBlock scheduled loop is tiled, then beg:end:1 becomes beg:end:tile_size
        # For GPU scheduling the thread block scheduled map needs to be scaled according to the tile_sizes
        # Furthermore the step of the device scheduled map needs to be increase too.
        # This can be handled by changing the range and the step of the thread block scheduled loop and increasing the step size of the parent

        dev_entry = self.device_map_entry
        thread_block_entry = self.thread_block_map_entry

        tx = self.tile_size_x
        ty = self.tile_size_y
        tz = self.tile_size_z
        possible_tile_sizes = [tz, ty, tx]
        # Depending on the sizes of the params use: (tz,ty,tx), (ty,tx) or (tx)
        if len(thread_block_entry.map.params) <= 3:
            tile_sizes = possible_tile_sizes[-len(thread_block_entry.map.params):]
        else:
            tile_sizes = [1] * (len(thread_block_entry.map.params) - 3) + possible_tile_sizes

        MapTiling.apply_to(sdfg=sdfg, options=dict(prefix="block", tile_sizes=tile_sizes, tile_trivial=True),  map_entry=thread_block_entry)

        thread_entry = thread_block_entry
        thread_entry.map.schedule = dtypes.ScheduleType.Sequential
        thread_entry.map.unroll = True

        # Find the thread block map encapsulating the sequential map
        # Entry node of a map should be the map scope avoce
        thread_block_entry = graph.entry_node(thread_block_entry)

        # Update the range if the inner sequential map
        thread_map : nodes.Map = thread_entry.map

        # Create the new dimension sizes for the ThreadBlock Map.
        # The dimensions of the thread block map to the step sizes of the device scheduled map.
        # They need to be scaled according to the tilesize, and the order of how they are mapped
        params = {"dim_size_z":1, "dim_size_y":1, "dim_size_x":1}
        dimension_names = ["dim_size_z", "dim_size_y", "dim_size_x"]
        for i in range(min(3, len(thread_block_entry.map.params)), 0, -1):
            params[dimension_names[-i]] = dev_entry.map.range[-i][2] * possible_tile_sizes[-i]

        # Last param of inner map is always the last iteration variable the map above
        range_str = ""
        # Thread block map is i0,i1,i2,i3
        # Which maps to           z, y, x
        # If more the 3 parameters they are linearized
        for i in range(len(thread_map.params), 0, -1):
            (beg, _, step) = thread_map.range[-i]
            (_, _, tstep) = thread_block_entry.map.range[-i]
            (_, dev_end, _) = dev_entry.map.range[-i]
            range_str += f"{beg}:Min({dev_end}+1, {beg}+{tstep}):{step}, "
        thread_map.range = subsets.Range.from_string(range_str[:-2])

        ChangeThreadBlockMap.apply_to(sdfg=sdfg, verify=False, device_scheduled_map_entry = dev_entry, 
                                           thread_block_scheduled_map_entry = thread_block_entry, 
                                           options=params)


    @staticmethod
    def annotates_memlets():
        return True
