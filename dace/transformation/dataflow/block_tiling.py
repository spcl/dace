# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from typing import List
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes

@make_properties
class BlockTiling(transformation.SingleStateTransformation):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)
    sequential_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    block_tiling_sizes = SymbolicProperty(dtype=List, default=[4], desc="")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry, cls.thread_block_map_entry, cls.sequential_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if graph.entry_node(self.sequential_map_entry) != self.thread_block_map_entry:
            return False
        if graph.entry_node(self.thread_block_map_entry) != self.device_map_entry:
            return False

        return True

    def update_names():
        pass

    def apply(self, graph: SDFGState, sdfg: SDFG):
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
        MapTiling.apply_to(sdfg=sdfg, 
                           options=dict(prefix="b", 
                                        tile_sizes=tile_sizes, 
                                        tile_trivial=True),
                            map_entry=map_entry)

        map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock
        map_entry.map.gpu_block_size = gpu_block_dims_ordered

        # The dev map is a new map where the gpu_block_size param is not transferred over
        dev_entry = graph.entry_node(map_entry)
        dev_entry.map.gpu_block_size = gpu_block_dims_ordered

        # Clear the copied-over edges that are not between any connectors (happens if such an edge exist to ensure
        # proper allocation of a constnat in after the device map)
        edges_to_remove = []
        for edge in graph.out_edges(dev_entry):
            u, u_conn, v, v_conn, memlet = edge
            if u_conn == None and v_conn == None and memlet.data == None:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            graph.remove_edge(edge)

    @staticmethod
    def annotates_memlets():
        return False
