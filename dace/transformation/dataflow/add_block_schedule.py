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

@make_properties
class AddBlockSchedule(transformation.SingleStateTransformation):
    """
    Adds a thread block schedule to a device map scope
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    thread_block_size_x = SymbolicProperty(dtype=int, default=32, desc="Number threads in the threadBlock X Dim")
    thread_block_size_y = SymbolicProperty(dtype=int, default=8, desc="Number threads in the threadBlock Y Dim")
    thread_block_size_z = SymbolicProperty(dtype=int, default=1, desc="Number threads in the threadBlock Z Dim")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        map_entry = self.map_entry

        # Applicable if the map is a GPU_Device scheduled map
        # And there is no other maps inside the map schedule
        for e in graph.out_edges(map_entry):
            if isinstance(e.dst, nodes.MapEntry):
                return False
        for e in graph.in_edges(map_entry):
            if isinstance(e.src, nodes.MapEntry):
                return False
        if self.thread_block_size_x * self.thread_block_size_y * self.thread_block_size_z > 1024:
            return False

        return MapTiling.can_be_applied(self, graph, expr_index=expr_index, sdfg=sdfg, permissive=permissive)

    def update_names():
        pass

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry

        tx = self.thread_block_size_x
        ty = self.thread_block_size_y
        tz = self.thread_block_size_z

        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        tile_sizes = None
        if len(map_entry.map.params) >= 3:
            tile_sizes = (tz, ty, tx)
        elif len(map_entry.map.params) == 2:
            tile_sizes = (ty, tx, 1)
        else: #1, 0 is impossible
            tile_sizes = (tx, 1, 1)

        MapTiling.apply_to(sdfg=sdfg, options=dict(prefix="grid_", tile_sizes=tile_sizes),  map_entry=map_entry)

        map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock

    @staticmethod
    def annotates_memlets():
        return False
