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

        # Map the block iterations (bx, by, bz) to the mapranges
        # In code-gen i0=...,i1=...,i2=...,i3=... is mapped to block dimensions as
        # i0xi1 -> z, i2 -> y, i3 -> x. If less than 2 or 1 then to x,y or x respectively
        block_iterations = [1, 1, 1]
        possible_block_iters = [bz, by, bx]
        for i in range(min(3, len(thread_block_entry.map.params)), 0, -1):
            ri = min(3, len(thread_block_entry.map.params)) - i
            block_iterations[ri] = possible_block_iters[-i]
        # Pad such that it has form (dX, dY, 1) if len <3
        block_iterations += [1] * (3 - len(block_iterations))

        # Update inner tile sizes, starting from the highest possible tile size (up to 3 for GPU dimensions)
        # The non-one tile sizes are set by the ThreadTiling transformation
        inner_tile_sizes = [1, 1, 1]
        for i in range(min(3, len(thread_block_entry.map.range)), 0, -1):
            ri = min(3, len(thread_block_entry.map.range)) - i
            (_, _, step) = thread_block_entry.map.range[-i]
            inner_tile_sizes[ri] = step

        # We need to change the step sized and ranges according to the following formula, here only for first dimension X,
        # Implementation is same for every dimensions, before the trasnformation the map ranges are as follows:
        # Before:
        # GPU_Device: [d=0:N:dX*blockDim.x] # Set GPU_block_size to [blockDim.x, ...]
        # GPU_ThreadBlock: [b=d:N:dX]
        # Sequential(ThreadCoarsening): [t=b:Min(N, t+dX):1]
        # After:
        # GPU_Device: [d=0:N:dX*blockDim.x*bX] # Keep GPU_block_size to [blockDim.x, ...]
        # GPU_ThreadBlock: [g=d:N:dX]
        # Sequential(BlockCoarsening): [b=g:N:blockDim.x*dX]
        # Sequential(ThreadCoarsening): [t=b:Min(N, t+dX):1]

        # The strides for the block iterations can be computed by the tile size (steps sizes of thread block schedule)
        # And the step size of the device schedule map devStep.x // threadBlockStep.x gives the exact number of threads
        block_strides = [1, 1, 1]
        saved_thread_block_steps = [1, 1, 1]
        assert(len(dev_entry.map.range) >= len(thread_block_entry.map.range))
        for i in range(min(3, len(thread_block_entry.map.range)), 0, -1):
            ri = min(3, len(thread_block_entry.map.range)) - i
            (_, _, block_step) = sequential_entry.map.range[-i]
            (_, _, dev_step) = dev_entry.map.range[-i]
            (_, _, thread_block_step) = thread_block_entry.map.range[-i]
            saved_thread_block_steps[ri] = thread_block_step
            block_strides[ri] = dev_step
            assert(dev_step % block_step == 0)

        # block_strides != tile_sizes is not implemented, we will update them in this transformation
        MapTiling.apply_to(sdfg=sdfg, options=dict(prefix="s", tile_sizes=block_iterations, tile_trivial=True),  map_entry=thread_block_entry)

        grid_strided_entry = thread_block_entry
        grid_strided_entry.map.schedule = dtypes.ScheduleType.Sequential
        grid_strided_entry.map.unroll = False

        thread_block_entry = graph.entry_node(thread_block_entry)

        # The hierarchy now is: dev_entry -> thread_block_entry -> grid_strided_entry -> sequential_entry
        # Update step size of dev_entry, need to update volumes and subsets accordingly 
        for i in range(min(3, len(dev_entry.map.range)), 0, -1):
            ri = min(3, len(dev_entry.map.range)) - i
            (dev_beg, dev_end, dev_step) = dev_entry.map.range[-i]
            iter_count = block_iterations[ri]
            dev_entry.map.range[-i] = (dev_beg, dev_end, dev_step * iter_count)

        # Assign the step size of block scheduled map as the saved sizes, map tiling changes this falsely
        for i in range(min(3, len(thread_block_entry.map.range)), 0, -1):
            (thread_block_beg, thread_block_end, _) = thread_block_entry.map.range[-i]
            thread_block_entry.map.range[-i] = (thread_block_beg, thread_block_end, saved_thread_block_steps[-i])

        # Update step size of the grided strided (block-coarsened) map
        # Only the innermost map needs the minimum computation, and the current range needs to change as well, update it
        range_str = ""
        for i in range(min(3, len(grid_strided_entry.map.range)), 0, -1):
            ri = min(3, len(dev_entry.map.range)) - i
            blockDims = dev_entry.map.gpu_block_size
            (_, dev_end, _) = dev_entry.map.range[-i]
            (grid_strided_beg, _, _) = grid_strided_entry.map.range[-i]
            iter_count = block_iterations[ri]
            range_str += f"{grid_strided_beg}:Min({dev_end},{grid_strided_beg}+({iter_count}*{saved_thread_block_steps[-i] * blockDims[ri]})-1)+1:{saved_thread_block_steps[-i] * blockDims[ri]}, "
        grid_strided_entry.map.range = subsets.Range.from_string(range_str[:-2])

        # TODO: Extend this to multiple edges
        # Now update the memlets
        # Out edges from GPU_Device now how a different volume and subset due to step change
        dev_exit = graph.exit_node(dev_entry)
        for edge in graph.out_edges(dev_entry) + graph.in_edges(dev_exit):
            u, u_conn, v, v_conn, memlet = edge
            new_volume = memlet.volume
            range_str = ""
            for i, (dev_range, block_range) in enumerate(zip(dev_entry.map.range, thread_block_entry.map.range)):
                (dev_beg, dev_end, dev_step) = dev_range
                (block_beg, _, _) = block_range
                new_volume *= block_iterations[i]
                range_str += f"{block_beg}:{block_beg}+{dev_step}, "
            memlet.volume = new_volume
            memlet._subset = subsets.Range.from_string(range_str[:-2])
            memlet._dynamic = False
            edge = (u, u_conn, v, v_conn, memlet)

        # The volume of incoming edges should be equal to the volume of outgoing edges times the number of iterations
        # Subset now involves a step that is not 1.
        # For every in_edge of form IN_* find the matching Out_*, and copy the volume
        # For every in_edge Update the subset from i:i+dx:1 to i:i+dx*blockDim.x*bx:blockDim.x*bx
        grid_strided_exit = graph.exit_node(grid_strided_entry)
        volume_factor = self.block_iter_x * self.block_iter_y * self.block_iter_z
        for ie, oe in [(graph.in_edges(grid_strided_entry),graph.out_edges(grid_strided_entry)),
                      (graph.out_edges(grid_strided_exit), graph.in_edges(grid_strided_exit))]:
            for edge in ie:
                u, u_conn, v, v_conn, memlet = edge
                new_volume = -1
                for out_edge in oe:
                    _, _u_conn, _, _, _memlet = out_edge
                    if _u_conn == u_conn:
                        new_volume = _memlet.volume * volume_factor
                        break
                assert(new_volume > 0)

                range_str = ""
                for i, (grid_strided_range, thread_block_range) in enumerate(zip(grid_strided_entry.map.range, thread_block_entry.map.range)):
                    (_, _, thread_block_step) = thread_block_range
                    (grid_strided_beg, _, grid_strided_step) = grid_strided_range
                    range_str += f"{grid_strided_beg}:{grid_strided_beg}+{thread_block_step}*{grid_strided_step}:{grid_strided_step}, "
                memlet.volume = new_volume
                memlet._subset = subsets.Range.from_string(range_str[:-2])
                memlet._dynamic = False
                edge = (u, u_conn, v, v_conn, memlet)


    @staticmethod
    def annotates_memlets():
        return True
