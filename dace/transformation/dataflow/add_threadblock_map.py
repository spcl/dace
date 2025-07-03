# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy

import dace
from dace.sdfg import SDFG, ControlFlowRegion, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
import warnings


@make_properties
class AddThreadBlockMap(transformation.SingleStateTransformation):
    """
    Adds a thread block schedule to a device map scope
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    thread_block_size_x = SymbolicProperty(dtype=int,
                                           default=None,
                                           allow_none=True,
                                           desc="Number threads in the threadBlock X Dim")
    thread_block_size_y = SymbolicProperty(dtype=int,
                                           default=None,
                                           allow_none=True,
                                           desc="Number threads in the threadBlock Y Dim")
    thread_block_size_z = SymbolicProperty(dtype=int,
                                           default=None,
                                           allow_none=True,
                                           desc="Number threads in the threadBlock Z Dim")
    tiles_evenly = SymbolicProperty(dtype=bool,
                                    default=False,
                                    desc="Whether the map should be tiled evenly or not. If False, the "
                                    "transformation will try to tile the map as evenly as possible.")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def preprocess_default_dims(self):
        # If None is passed for the pass we will get the default configs
        # 1. If arguments are passed:
        #    1.1 Is the arguments passed
        # 2. If no arguments are passed (at least one arg is None):
        #   2.1. First check if the device map has gpu_block_size set
        #   2.2. Otherwise check the global default
        if self.thread_block_size_x is None or self.thread_block_size_y is None or self.thread_block_size_z is None:
            if self.map_entry.gpu_block_size is not None:
                # If gpu_block_size ap_entry.gpu_block_sizeis set, use it
                self.thread_block_size_x = self.map_entry.gpu_block_size[0]
                self.thread_block_size_y = self.map_entry.gpu_block_size[1]
                self.thread_block_size_z = self.map_entry.gpu_block_size[2]
            else:
                x, y, z = dace.config.Config.get('compiler', 'cuda', 'default_block_size').split(',')
                try:
                    self.thread_block_size_x = int(x)
                    self.thread_block_size_y = int(y)
                    self.thread_block_size_z = int(z)
                except ValueError:
                    raise ValueError("Invalid default block size format. Expected 'x,y,z' where x, y, z are integers.")

            num_dims_in_map = len(self.map_entry.map.range)
            # Collapse missing thread block dimensions into y if 2 dimensions in the map, to x if 1 dimension in the map
            if num_dims_in_map < 3:
                print_warning = False
                old_block = (self.thread_block_size_x, self.thread_block_size_y, self.thread_block_size_z)
                if num_dims_in_map == 2:
                    self.thread_block_size_y *= self.thread_block_size_z
                    if self.thread_block_size_z > 1:
                        print_warning = True
                    self.thread_block_size_z = 1
                elif num_dims_in_map == 1:
                    self.thread_block_size_x *= self.thread_block_size_y * self.thread_block_size_z
                    if self.thread_block_size_y > 1 or self.thread_block_size_z > 1:
                        print_warning = True
                    self.thread_block_size_y = 1
                    self.thread_block_size_z = 1
                new_block = (self.thread_block_size_x, self.thread_block_size_y, self.thread_block_size_z)
                if print_warning:
                    warnings.warn(
                        UserWarning, f'Default block size has more dimensions ({old_block}) than kernel dimensions '
                        f'({num_dims_in_map}) in map "{self.map_entry.map.label}". Linearizing block '
                        f'size to {new_block}. Consider setting the ``gpu_block_size`` property.')
            
            

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        self.preprocess_default_dims()

        # Reject if thread block size exceeds GPU hardware limits
        total_block_size = self.thread_block_size_x * self.thread_block_size_y * self.thread_block_size_z

        if total_block_size > 1024:
            return False

        # Only applicable to GPU_Device maps
        if self.map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False

        # Traverse inner scopes (ordered outer -> inner)
        for _, inner_entry in helpers.get_internal_scopes(graph, self.map_entry):
            schedule = inner_entry.map.schedule

            if schedule in {dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,}:
                # Already scheduled with thread block — cannot apply
                return False

            if schedule == dtypes.ScheduleType.GPU_Device:
                # Found another kernel launch — safe to apply
                return True


        # No thread block schedule found - do apply
        return True


    def apply(self, state: SDFGState, sdfg: SDFG):
        self.preprocess_default_dims()

        map_entry = self.map_entry

        tx = self.thread_block_size_x
        ty = self.thread_block_size_y
        tz = self.thread_block_size_z
        block_dims = [tz, ty, tx]

        # Set the gpu_block_size which the GPU_ThreadBlock map will use. This is important, because the CUDACodeGen
        # will otherwise try to deduce it, leading to issues
        self.map_entry.gpu_block_size = [self.thread_block_size_x, self.thread_block_size_y, self.thread_block_size_z]

        # TODO: Adapt this code once MapTiling transformation also considers existing stride.
        # The below tile size works around this by including the existing stride into the tile size
        num_dims = len(map_entry.map.params)
        existing_strides = map_entry.range.strides()

        len_diff = num_dims - len(block_dims) # Note
        if len_diff > 0: # num_dims > block_dims
            block_dims = [1] * len_diff + block_dims
        else:
            block_dims = block_dims[-num_dims:]

        tile_sizes = [stride * block for stride, block in zip(existing_strides, block_dims)]
        
        # Tile trivial simplifies come checks for the BlockCoarsening and ThreadCoarsening transformations
        MapTiling.apply_to(
            sdfg=sdfg,
            options=dict(
                prefix="b",
                tile_sizes=tile_sizes,
                divides_evenly=self.tiles_evenly,  # Todo improve this
                tile_trivial=True,
                skew=False),
            map_entry=map_entry)

        # The old dev_entry is the new tblock_map_entry
        map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock



    def update_names():
        pass

    @staticmethod
    def annotates_memlets():
        return False
