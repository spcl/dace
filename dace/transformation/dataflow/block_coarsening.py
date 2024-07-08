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
from dace import subsets

@make_properties
class BlockCoarsening(transformation.SingleStateTransformation):
    """
    Implements block tiling. The thread block over a sub domain of the output, 
    essentially a thread computes a block_iter_x * block_iter_y * block_iter_z subdomain instead of
    1 cell, where the iteration has the stride blockDim.x, blockDim.y and blockDim.z for each dimension
    """

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)
    sequential_map_entry = transformation.PatternNode(nodes.MapEntry)
    
    # Properties
    block_iter_x = SymbolicProperty(dtype=int, default=4, desc="Number threads in the threadBlock X Dim")
    block_iter_y = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Y Dim")
    block_iter_z = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Z Dim")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.device_map_entry, cls.sequential_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        dev_entry = self.device_map_entry
        thread_block_entry = self.thread_block_map_entry
        sequential_entry = self.sequential_map_entry

        # Applicable if the map is a GPU_ThreadBlock Scheduled Map and the we have 
        # A device -> thread block -> sequential (thread tiling, could be a loop of lenth 1) map hierarchy
        if thread_block_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False
        if graph.entry_node(thread_block_entry) != dev_entry or \
            graph.entry_node(sequential_entry) != sequential_entry:
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
        sequential_entry = self.sequential_map_entry

        bx = self.block_iter_x
        by = self.block_iter_y
        bz = self.block_iter_z
        requested_block_iters = [bz, by, bx]

        # Map the block iterations (bx, by, bz) to the mapranges
        # In code-gen i0=...,i1=...,i2=...,i3=... is mapped to block dimensions as
        # i0xi1 -> z, i2 -> y, i3 -> x. If less than 2 or 1 then to x,y or x respectively
        assert(len(thread_block_entry.map.params) == len(dev_entry.map.params))
        used_dimension_count = min(3, len(thread_block_entry.map.params))
        block_iterations = [1] * len(thread_block_entry.map.params)
        saved_thread_block_steps = [1] * len(thread_block_entry.map.params)
        saved_dev_steps = [1] * len(thread_block_entry.map.params)
        saved_dev_gpu_block_size = dev_entry.map.gpu_block_size

        block_iterations[-used_dimension_count:] = requested_block_iters[-used_dimension_count:]

        # Save the current step sizes of the thread block as map tiling changes is incorrectly as we rely on different strides
        # Which is not implemented
        for i in range(used_dimension_count, 0, -1):
            (_, _, dev_step) = dev_entry.map.range[-i]
            (_, _, thread_block_step) = thread_block_entry.map.range[-i]
            saved_dev_steps[-i] = dev_step # Get step size of Dev map, it is same as the block stride
            saved_thread_block_steps[-i] = thread_block_step

        # We need to change the step sized and ranges according to the following formula, here only for first dimension X,
        # Implementation is same for every dimensions, before the trasnformation the map ranges are as follows:
        # Before:
        # GPU_Device: [d=0:N:dX*blockDim.x] # Set GPU_block_size to [blockDim.x, ...]
        # GPU_ThreadBlock: [b=d:N:dX]
        # Sequential(ThreadCoarsening): [t=b:Min(N, t+dX):1]
        # After:
        # GPU_Device: [d=0:N:dX*blockDim.x*bX] # Keep GPU_block_size to [blockDim.x, ...]
        # Sequential(BlockCoarsening): [g=d:Min(N, g+blockDim.x*dX*bX):blockDim.x*dX]
        # GPU_ThreadBlock: [b=g:Min(N, b+dX):dX]
        # Sequential(ThreadCoarsening): [t=b:Min(N, t+dX):1]
        # bX = block_iteration
        # blockDim.x = block_stride
        block_iterations_mapped_to_tile_sizes = [num_iter * blockdim for num_iter, blockdim in zip(block_iterations, saved_dev_steps)]
        # block_strides != tile_sizes is not implemented, we will update them in this transformation
        MapTiling.apply_to(sdfg=sdfg, options=dict(prefix="g", tile_sizes=block_iterations_mapped_to_tile_sizes, tile_trivial=True),  map_entry=dev_entry)

        grid_strided_entry = dev_entry
        grid_strided_entry.map.schedule = dtypes.ScheduleType.Sequential
        grid_strided_entry.map.unroll = False

        dev_entry = graph.entry_node(dev_entry)

        for i in range(len(dev_entry.map.params), 0, -1):
            (dev_beg, dev_end, _) = dev_entry.map.range[-i]
            dev_entry.map.range[-i] = (dev_beg, dev_end, saved_dev_steps[-i] * block_iterations[-i])
        dev_entry.map.gpu_block_size = saved_dev_gpu_block_size 

        # Now update the memlets (approximation of map tiling is not the best, can do better)
        # Out edges from GPU_Device now how a different volume and subset due to step change
        dev_exit = graph.exit_node(dev_entry)
        for edge in graph.out_edges(dev_entry) + graph.in_edges(dev_exit):
            u, u_conn, v, v_conn, memlet = edge
            new_volume = 1
            range_str = ""
            for i in range(len(dev_entry.map.range), 0, -1):
                (dev_beg, dev_end, dev_step) = dev_entry.map.range[-i]
                (grid_beg, _, _) = grid_strided_entry.map.range[-i]
                new_volume *= dev_step
                range_str += f"{grid_beg}:{grid_beg}+{dev_step}, "
            memlet.volume = new_volume
            memlet._subset = subsets.Range.from_string(range_str[:-2])
            memlet._dynamic = False
            edge = (u, u_conn, v, v_conn, memlet)

    @staticmethod
    def annotates_memlets():
        return True
