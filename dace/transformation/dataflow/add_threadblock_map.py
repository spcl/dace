# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
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
                # If gpu_block_size is set, use it
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

        if self.thread_block_size_x * self.thread_block_size_y * self.thread_block_size_z > 1024:
            return False

        if self.map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False

        kernel_nodes = graph.all_nodes_between(self.map_entry, graph.exit_node(self.map_entry))
        for node in kernel_nodes:
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_ThreadBlock:
                # If the map already has a thread block schedule, do not apply
                return False

        return True

    def update_names():
        pass

    def apply(self, state: SDFGState, sdfg: SDFG):
        self.preprocess_default_dims()

        map_entry = self.map_entry

        tx = self.thread_block_size_x
        ty = self.thread_block_size_y
        tz = self.thread_block_size_z
        block_dims = [tz, ty, tx]

        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        # If more tile sizes are given than the available number of parameters cull the list and ignore
        # the additional parameters
        tile_sizes = [1] * len(map_entry.map.params)
        used_dimensions = min(3, len(map_entry.map.params))
        tile_sizes[-used_dimensions:] = block_dims[-used_dimensions:]
        applied_gpu_block_dims = [1, 1, 1]
        applied_gpu_block_dims[-used_dimensions:] = block_dims[-used_dimensions:]

        # Tile trivial simplifies come checks for the BlockCoarsening and ThreadCoarsening transformations
        MapTiling.apply_to(sdfg=sdfg,
                           options=dict(prefix="b",
                                        tile_sizes=tile_sizes,
                                        divides_evenly=True,
                                        tile_trivial=True,
                                        skew=True),
                           map_entry=map_entry)

        # The old dev_entry is the new tblock_map_entry
        map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock

    @staticmethod
    def annotates_memlets():
        return False
