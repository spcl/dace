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
        gpu_block_dims_ordered = list(reversed(applied_gpu_block_dims))

        # Tile trivial simplifies come checks for the BlockCoarsening and ThreadCoarsening transformations
        MapTiling.apply_to(
            sdfg=sdfg,
            options=dict(
                prefix="b",
                #tile_offset=map_begins,
                tile_sizes=tile_sizes,
                divides_evenly=True,
                tile_trivial=True,
                skew=True),
            map_entry=map_entry)

        map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock
        map_entry.map.label = "ThreadBlockMap"

        for m in [map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d

        # The dev map is a new map where the gpu_block_size param is not transferred over
        dev_entry = state.entry_node(map_entry)
        dev_entry.map.label = "KernelEntryMap"

        # Clear the copied-over edges that are not between any connectors (happens if such an edge exist to ensure
        # proper allocation of a constnat in after the device map)
        """
        edges_to_remove = []
        for edge in state.out_edges(dev_entry):
            u, u_conn, v, v_conn, memlet = edge
            if u_conn == None and v_conn == None and memlet.data == None:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            state.remove_edge(edge)
        """
        """
        # In assignment maps it can be that the new map is not connected.
        out_edges = state.out_edges(dev_entry)
        if len(out_edges) == 0:
            state.add_edge(dev_entry, None, map_entry, None, None)
        """

    @staticmethod
    def annotates_memlets():
        return Fals
