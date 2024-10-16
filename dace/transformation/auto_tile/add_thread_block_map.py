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
class AddThreadBlockMap(transformation.SingleStateTransformation):
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
        if self.thread_block_size_x * self.thread_block_size_y * self.thread_block_size_z > 1024:
            return False

        #return MapTiling.can_be_applied(self, graph, expr_index=expr_index, sdfg=sdfg, permissive=permissive)
        return True

    def update_names():
        pass

    def apply(self, state: SDFGState, sdfg: SDFG):
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
        return False