# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

from typing import List
from dace.sdfg import SDFG, SDFGState, propagation
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import dtypes
from dace import subsets


@make_properties
class ChangeThreadBlockMap(transformation.SingleStateTransformation):
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


        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        new_block_dimensions = [1, 1, 1]
        dims = [self.dim_size_z, self.dim_size_y, self.dim_size_x]
        new_block_dimensions[-min(3, len(dev_map.range)):] = dims[-min(3, len(dev_map.range)):]

        if new_block_dimensions == dims:
            # Same type requested, do no need the range recalculation, and only update the memlets
            # Step 1. Update step sizes of device map
            dev_old_step_sizes = []

            dev_ranges : List[subsets.Range] = dev_map.range
            for i, dev_range in enumerate(dev_ranges):
                (beg, end, step) = dev_range
                dev_old_step_sizes.append(step)

            for i in range(min(3, len(dev_map.range)), 0, -1):
                (beg, end, _) = dev_map.range[-i]
                (_, _, step) = block_map.range[-i]
                dev_map.range[-i] = (beg, end, new_block_dimensions[-i] * step)


            # Step 2. Update the range of the thread block map
            block_ranges : List[subsets.Range] = block_map.range
            block_steps = []
            new_thread_block_map_range_str = ""
            for i, (dev_range, block_range) in enumerate(zip(dev_ranges, block_ranges)):
                (_, dev_end, dev_step) = dev_range
                (block_beg, _, block_step) = block_range
                block_steps.append(block_step)
                new_thread_block_map_range_str += f"{block_beg}:{dev_step}:{block_step}, "
            block_map.range = subsets.Range.from_string(new_thread_block_map_range_str[:-2])

        # Step 3. Propagate memlets
        # Overapproximate the memory moved such that the terms do not involve any min:
        # propagation.propagate_memlets_state(sdfg=sdfg, state=graph)

        # Apparently gpu_block_size only necessary if there is no gpu_thread_block schedule
        # Therefore the code-gen now choose the values in gpu_block_size if it conflicts with the
        # detected blocksizes, and these transformatiosn need to update them
        # Block steps are returned in the form of z, y, x
        # dev_map.gpu_block_size =  list(reversed(new_block_dimensions))
        for m in [dev_map, block_map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d
    @staticmethod
    def annotates_memlets():
        return False
