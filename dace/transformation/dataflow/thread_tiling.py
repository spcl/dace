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
from dace.transformation.dataflow.thread_block_map_range_change import ThreadBlockMapRangeChange
from dace import subsets
from typing import List

@make_properties
class ThreadTiling(transformation.SingleStateTransformation):
    """
    Adds a thread block schedule to a device map scope
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
        dev_entry = self.device_map_entry
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

        tile_sizes = None
        if len(thread_block_entry.map.params) >= 3:
            tile_sizes = (tz, ty, tx)
        elif len(thread_block_entry.map.params) == 2:
            tile_sizes = (ty, tx, 1)
        else: #1, 0 is impossible
            tile_sizes = (tx, 1, 1)

        MapTiling.apply_to(sdfg=sdfg, options=dict(prefix="block_", tile_sizes=tile_sizes),  map_entry=thread_block_entry)

        thread_entry = thread_block_entry
        thread_entry.map.schedule = dtypes.ScheduleType.Sequential
        thread_entry.map.unroll = True

        # Find the thread block map encapsulating the sequential map
        # Entry node of a map should be the map scope avoce
        thread_block_entry = graph.entry_node(thread_block_entry)

        # Update the range if the inner sequential map
        thread_map : nodes.Map = thread_entry.map
        thread_range_str = ""
        assert(min(len(thread_block_entry.map.range), len(thread_map.range), len(dev_entry.map.range)) >= 1)
        assert(len(thread_block_entry.map.range) == len(dev_entry.map.range))
        #assert(len(thread_block_entry.map.range) == len(thread_map.range))

        #for i in range(3):
        #    if i < min(len(thread_block_entry.map.range), len(thread_map.range), len(dev_entry.map.range)):
        #        (beg, end, step) = thread_map.range[i]
        #        (_, block_end, block_step) = thread_block_entry.map.range[i]
        #        (_, dev_end, _) = dev_entry.map.range[i]
        #        thread_range_str += f"{beg}:Min({dev_end}, {block_end}, {beg}+{block_step}-1)+1:{step}, "
        #thread_map.range = subsets.Range.from_string(thread_range_str[:-2])

        # Last param of inner map is always the last iteration variable the map above
        num_iter = min(len(thread_block_entry.map.range), len(thread_map.range))
        ranges_from_last = []
        # Access from last
        for i in range(-1, -(num_iter+1), -1):
            (beg, end, step) = thread_map.range[i]
            (_, block_end, block_step) = thread_block_entry.map.range[i]
            (_, dev_end, _) = dev_entry.map.range[i]
            ranges_from_last.append(f"{beg}:Min({dev_end}, {block_end}, {beg}+{block_step}-1)+1:{step}")
        range_str = ", ".join(list(reversed(ranges_from_last)))
        thread_map.range = subsets.Range.from_string(range_str)

        # Create the new dimension sizes for the ThreadBlock Map.
        # The dimensions of the thread block map to the step sizes of the device scheduled map.
        # They need to be scaled according to the tilesize
        params = dict()
        if len(thread_block_entry.map.params) >= 3:
            #tile_sizes = (tz, ty, tx)
            params[f"dim_size_z"] = thread_block_entry.map.range[0][2]*tz
            params[f"dim_size_y"] = thread_block_entry.map.range[1][2]*ty
            params[f"dim_size_x"] = thread_block_entry.map.range[2][2]*tx
        elif len(thread_block_entry.map.params) == 2:
            #tile_sizes = (ty, tx, 1)
            params[f"dim_size_z"] = 1
            params[f"dim_size_y"] = thread_block_entry.map.range[0][2]*ty
            params[f"dim_size_x"] = thread_block_entry.map.range[1][2]*tx
        else: #1, 0 is impossible
            #tile_sizes = (tx, 1, 1)
            params[f"dim_size_z"] = 1
            params[f"dim_size_y"] = 1
            params[f"dim_size_x"] = thread_block_entry.map.range[0][2]*tx

        ThreadBlockMapRangeChange.apply_to(sdfg=sdfg, verify=False, device_scheduled_map_entry = dev_entry, 
                                           thread_block_scheduled_map_entry = thread_block_entry, 
                                           options=params)


    @staticmethod
    def annotates_memlets():
        return True
