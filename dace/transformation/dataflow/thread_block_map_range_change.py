# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from typing import List
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import dtypes
from dace import subsets

@make_properties
class ThreadBlockMapRangeChange(transformation.SingleStateTransformation):
    """
    Changes the range and step size of a thread block scheduled map
    """

    device_scheduled_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_scheduled_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    dim_size_x = SymbolicProperty(dtype=int, default=32, desc="Number threads in the threadBlock X Dim")
    dim_size_y = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Y Dim")
    dim_size_z = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Z Dim")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_scheduled_map_entry, cls.thread_block_scheduled_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Applicable if the map is a GPU_ThreadBlock Scheduled Map
        if self.thread_block_scheduled_map_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False
        if self.device_scheduled_map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # The device map is the form gm=0:M:Ms, gk=0:K:Ks, gn=0:N:Ns
        # The thread block mas is the form bm=gm:Min(M-1, gm + Ms):1, bk=gk:Min(K-1, gk + Ks):1, bn=gn:Min(N-1, gn + Ns):1
        # Where Ms = #threadsX, K = #threadsY, N = #threadsZ for CUDA kernels

        # We want to change thread block scheduled inner map with another map with different number of threads
        # It means we need to update the following:
        # 1. Ms, Ks, Ns of the device schuduled map tp #X,#Y,#Z (num_parameters_(x|y|z)).
        # 2. Update the step sized and ranges of the thread block schedule map.
        #    It will look as bm=gm:Min(M-1, gm + #X):1, bk=gk:Min(K-1, gk + #Y):1, bn=gn:Min(N-1, gn + #Z):1
        # 3. Update all memlets, meaning their subsets and volumes accordingly to changes in the eges between the two maps
        #    This needs to be done both for Map entry and Map exit
        # 4. Update the gpu_block_size and gpu_launch_bounds according to the new number of threads

        dev_entry : nodes.MapEntry = self.device_scheduled_map_entry
        block_entry : nodes.MapEntry = self.thread_block_scheduled_map_entry

        dev_map : nodes.Map = dev_entry.map
        block_map : nodes.Map = block_entry.map

        new_block_dimensions = (self.dim_size_x, self.dim_size_y, self.dim_size_z)
        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        new_block_dimensions = None
        if len(dev_entry.map.params) >= 3:
            new_block_dimensions = (self.dim_size_z, self.dim_size_y, self.dim_size_x)
        elif len(dev_entry.map.params) == 2:
            new_block_dimensions = (self.dim_size_y, self.dim_size_x, 1)
        else: #1, 0 is impossible
            new_block_dimensions = (self.dim_size_x, 1, 1)

        # Step 1. Update step sizes of device map
        dev_old_step_sizes = []

        dev_ranges : List[subsets.Range] = dev_map.range
        for i, dev_range in enumerate(dev_ranges):
            (beg, end, step) = dev_range
            dev_old_step_sizes.append(step)
            dev_map.range[i] = (beg, end, new_block_dimensions[i])

        # Step 2. Update the range of the thread block map (end)
        block_ranges : List[subsets.Range] = block_map.range
        block_steps = []
        new_thread_block_map_range_str = ""
        for i, (dev_range, block_range) in enumerate(zip(dev_ranges, block_ranges)):
            (_, dev_end, _) = dev_range
            (block_beg, _, block_step) = block_range
            new_block_dim = new_block_dimensions[i]
            block_steps.append(block_step)
            new_thread_block_map_range_str += f"{block_beg}:{dev_end}+1:{block_step}"
            new_thread_block_map_range_str += ", "
        block_map.range = subsets.Range.from_string(new_thread_block_map_range_str[:-2])

        # Step 3. Propagate memlets
        # In the device Map the step size has changed.
        # 1. The input  volume of the memlets of the device map are not changed
        #    Yet due to the teiling size input dimensions are approximated as stepD * ceil(lenD / stepD),
        #    and it needs to be updated as well (due to the Min this approximation can be improved)
        # 2. The output volume of the memlets of the device map are changed because of the change in step size
        # 3. The input  volume of the memlets of the thread block map are changed according to the device step size change
        # 4. The output volume of the memlets of the thread block map are not changed
        # Note: currently it is assuemd that all edges are of the from Device Map -> Thread Block Map
        # Note: it needs to be repeated both for entry and exit
        # Changes for 1:
        dev_exit = graph.exit_node(dev_entry)
        for edge in graph.in_edges(dev_entry) + graph.out_edges(dev_exit):
            u, u_conn, v, v_conn, memlet = edge
            #m = copy.deepcopy(memlet)
            new_volume = 1
            range_str = ""
            for i, (dev_range, block_range) in enumerate(zip(dev_ranges, block_ranges)):
                (dev_beg, dev_end, dev_step) = dev_range
                (block_beg, _, _) = block_range
                new_volume *= (dev_end - dev_beg) + 1
                range_str += f"{dev_beg}:{dev_end}+1"
                range_str += ", "
            memlet.volume = new_volume
            memlet._subset = subsets.Range.from_string(range_str[:-2])
            memlet._dynamic = False
            edge = (u, u_conn, v, v_conn, memlet)

        # Changes for 2: (Assumption means changes for 2 = changes for 3)
        for edge in graph.out_edges(dev_entry) + graph.in_edges(dev_exit):
            u, u_conn, v, v_conn, memlet = edge
            new_volume = 1
            range_str = ""
            for i, (dev_range, block_range) in enumerate(zip(dev_ranges, block_ranges)):
                (_, _, dev_step) = dev_range
                (block_beg, _, _) = block_range
                new_volume *= dev_step
                range_str += f"{block_beg}:{block_beg}+{dev_step}"
                range_str += ", "
            memlet.volume = new_volume
            memlet._subset = subsets.Range.from_string(range_str[:-2])
            edge = (u, u_conn, v, v_conn, memlet)
        
        # Changes for 4: change gpu_block_size and gpu_launch_bounds
        # Apparently gpu_block_size only necessary if there is no gpu_thread_block schedule
        # Therefore the code-gen now choose the values in gpu_block_size if it conflicts with the
        # detected blocksizes, and these transformatiosn need to update them
        # Block steps are returned in the form of z, y, x
        if len(block_steps) == 3:
            dev_map.gpu_block_size = (self.dim_size_x // block_steps[2], self.dim_size_y // block_steps[1], self.dim_size_z // block_steps[0])
            block_map.gpu_block_size = dev_map.gpu_block_size
        elif len(block_steps) == 2:
            dev_map.gpu_block_size = (self.dim_size_x // block_steps[1], self.dim_size_y // block_steps[0], self.dim_size_z)
            block_map.gpu_block_size = dev_map.gpu_block_size
        elif len(block_steps) == 1:
            dev_map.gpu_block_size = (self.dim_size_x // block_steps[0], self.dim_size_y, self.dim_size_z)
            block_map.gpu_block_size = dev_map.gpu_block_size

        # TODO:
        # Implement launch bounds

    @staticmethod
    def annotates_memlets():
        return True
