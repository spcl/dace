# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import dace
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
from dace.transformation.auto_tile.change_thread_block_map import ChangeThreadBlockMap
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

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        # Applicable if the map is a GPU_ThreadBlock Scheduled Map
        if self.thread_block_map_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False

        return MapTiling.can_be_applied(self, state, expr_index=expr_index, sdfg=sdfg, permissive=permissive)

    def update_names():
        pass

    def apply(self, state: SDFGState, sdfg: SDFG):
        # When the ThreadBlock scheduled loop is tiled, then beg:end:1 becomes beg:end:tile_size
        # For GPU scheduling the thread block scheduled map needs to be scaled according to the tile_sizes
        # Furthermore the step of the device scheduled map needs to be increase too.
        # This can be handled by changing the range and the step of the thread block scheduled loop and increasing the step size of the parent

        dev_entry : nodes.MapEntry = self.device_map_entry
        thread_block_entry : nodes.MapEntry = self.thread_block_map_entry

        # If there are transient access nodes leading to the first inner sequential map the sizes of the arrays and the
        # subsets / volumes of the corresponding memlets need to be adapted
        inner_sequential_map_entry = None
        for out_edge in state.out_edges(thread_block_entry):
            u, _, v, _, _ = out_edge
            if isinstance(v, nodes.MapEntry):
                assert(v.map.schedule == dtypes.ScheduleType.Sequential)
                inner_sequential_map_entry = v

        tx = self.tile_size_x
        ty = self.tile_size_y
        tz = self.tile_size_z
        possible_tile_sizes = [tz, ty, tx]
        used_dimensions = min(3, len(thread_block_entry.map.params))
        tile_sizes = [1] * len(thread_block_entry.map.params)
        # Depending on the sizes of the params use: (tz,ty,tx), (ty,tx) or (tx)
        tile_sizes[-used_dimensions:] = possible_tile_sizes[-used_dimensions:]

        MapTiling.apply_to(sdfg=sdfg,
                           options=dict(prefix="d",
                                        skew=True,
                                        tile_sizes=tile_sizes,
                                        divides_evenly=True,
                                        tile_trivial=True),
                           map_entry=thread_block_entry)



        sequential_map_entry = thread_block_entry
        sequential_map_entry.map.schedule = dtypes.ScheduleType.Sequential
        sequential_map_entry.map.unroll = True
        sequential_map_entry.map.label = "ThreadCoarsenedMap"

        # Find the thread block map encapsulating the sequential map
        # Entry node of a map should be the map scope avoce
        thread_block_entry = state.entry_node(thread_block_entry)

        # Update the range if the inner sequential map
        sequential_map : nodes.Map = sequential_map_entry.map

        # Rm unnecessary copy-over edges
        """
        edges_to_remove = []
        for edge in state.out_edges(thread_block_entry):
            u, u_conn, v, v_conn, memlet = edge
            if u_conn == None and v_conn == None and memlet.data == None:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            state.remove_edge(edge)
        """

        # Move the access above the outer sequential map and update memlets for the map entry
        if inner_sequential_map_entry != None:
            updated_arr_names = set()
            inner_offset_memlet_list = [(dace.symbol(param) - beg, dace.symbol(param) - beg, 1) for \
                                        param, (beg, _, _) in zip(sequential_map.params, sequential_map.range)]
            for in_edge in state.in_edges(inner_sequential_map_entry):
                u, _, _, _, _ = in_edge
                if isinstance(u, nodes.AccessNode):
                    access_node = u
                    # Current assumption outer seq map -> access node -> inner seq map
                    assert(len(state.in_edges(access_node)) == 1)
                    assert(len(state.out_edges(access_node)) == 1)
                    seq_to_access_node_edge = state.in_edges(access_node)[0]
                    in_u, in_u_conn, in_v, in_v_conn, in_memlet = seq_to_access_node_edge
                    access_node_to_inner_seq = in_edge
                    out_u, out_u_conn, out_v, out_v_conn, out_memlet = access_node_to_inner_seq
                    assert(in_u == sequential_map_entry)
                    assert(out_v == inner_sequential_map_entry)

                    state.remove_edge(seq_to_access_node_edge)
                    state.remove_edge(access_node_to_inner_seq)

                    new_memlet_list = [(0, end-1, 1) for end in tile_sizes[-used_dimensions:]]
                    if in_memlet.data == None:
                        new_in_memlet = Memlet(subset=None, data=in_memlet.data)
                    else:
                        new_in_memlet = Memlet(subset=subsets.Range(new_memlet_list), data=in_memlet.data, wcr=in_memlet.wcr, wcr_nonatomic=in_memlet.wcr_nonatomic, allow_oob=in_memlet.allow_oob, debuginfo=in_memlet.debuginfo)
                    new_out_memlet = Memlet(subset=subsets.Range(new_memlet_list), data=out_memlet.data, wcr=out_memlet.wcr, wcr_nonatomic=out_memlet.wcr_nonatomic, allow_oob=out_memlet.allow_oob, debuginfo=out_memlet.debuginfo)
                    offseted_memlet = Memlet(subset=subsets.Range(inner_offset_memlet_list), data=out_memlet.data, wcr=out_memlet.wcr, wcr_nonatomic=out_memlet.wcr_nonatomic, allow_oob=out_memlet.allow_oob, debuginfo=out_memlet.debuginfo)

                    state.add_edge(thread_block_entry, in_u_conn, access_node, in_v_conn, new_in_memlet)
                    state.add_edge(access_node, out_u_conn, sequential_map_entry, out_v_conn, new_out_memlet)

                    # If data was first create it will be be (outconn) None -> access node (inconn) smth.
                    # In that case we copy as it is to the map above, but the new map needs both connectors
                    if in_u_conn != None:
                        thread_block_entry.add_out_connector(in_u_conn)
                    assert(out_v_conn != None)
                    sequential_map_entry.add_in_connector(out_v_conn)

                    in_conn_for_inner_seq = out_v_conn[:]
                    out_conn_for_seq = "OUT_" + out_v_conn[3:]

                    sequential_map_entry.add_out_connector(out_conn_for_seq)
                    inner_sequential_map_entry.add_in_connector(in_conn_for_inner_seq)

                    state.add_edge(sequential_map_entry, out_conn_for_seq, inner_sequential_map_entry, in_conn_for_inner_seq, offseted_memlet)
                    updated_arr_names.add(out_memlet.data)

                    # It was scalar before convert to array
                    data_type = sdfg.arrays[out_memlet.data].dtype
                    assert(sdfg.arrays[out_memlet.data].storage == dtypes.StorageType.Register or sdfg.arrays[out_memlet.data].storage == dtypes.StorageType.Default)
                    sdfg.remove_data(out_memlet.data, validate=False)
                    sdfg.add_array(name=out_memlet.data, shape=tile_sizes[-used_dimensions:], storage=dtypes.StorageType.Register,
                                   dtype=data_type, transient=True, alignment=16)

            # Now update remaining memlets, accessing temporary scalars
            data_to_check = set(updated_arr_names)
            edges_to_check = set(state.out_edges(inner_sequential_map_entry))
            while (len(edges_to_check) > 0):
                edge = edges_to_check.pop()
                u, u_conn, v, v_conn, memlet = edge
                if memlet.data != None and memlet.data in data_to_check:
                    offseted_memlet = Memlet(subset=subsets.Range(inner_offset_memlet_list), data=memlet.data, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
                    state.remove_edge(edge)
                    state.add_edge(u, u_conn, v, v_conn, offseted_memlet)
                if not (isinstance(v, nodes.MapExit) and v == state.exit_node(sequential_map_entry)):
                    edges_to_check = edges_to_check.union(state.out_edges(v))

        # Map Tiling does not update the range as we need them
        # Update the threadblock and device ranges
        dev_updated_range_list = [(beg,end,step*tstep) for (beg,end,step), (_,_,tstep) in zip(dev_entry.map.range, thread_block_entry.map.range)]
        dev_entry.map.range = subsets.Range(dev_updated_range_list)
        thread_block_updated_range_list = [(beg,(end+1)*step-1,step) for (beg,end,step) in thread_block_entry.map.range]
        thread_block_entry.map.range = subsets.Range(thread_block_updated_range_list)

        for m in [dev_entry.map, thread_block_entry.map, sequential_map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d


    @staticmethod
    def annotates_memlets():
        return False