# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty, ListProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes, subsets
from functools import reduce

@make_properties
class AddWarpMap(transformation.SingleStateTransformation):
    """
    Adds a warp schedule to a ThreadBlock map scope
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    warp_dims =  ListProperty(element_type=int, default=[32], desc="Dimensions of Arrangement")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if self.map_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False

        requested_warp_size = reduce(lambda x, y: x * y, self.warp_dims)
        if requested_warp_size != 32:
            return False

        return True

    def update_names():
        pass

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry

        warp_dims = self.warp_dims + [1] * (3 - len(self.warp_dims)) if len(self.warp_dims) < 3 else self.warp_dims
        warp_dims.reverse()

        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        # If more tile sizes are given than the available number of parameters cull the list and ignore
        # the additional parameters
        tile_sizes = [1] * len(map_entry.map.params)
        used_dimensions = min(3, len(map_entry.map.params))
        tile_sizes[-used_dimensions:] = warp_dims[-used_dimensions:]

        MapTiling.apply_to(sdfg=sdfg,
                           options=dict(prefix="w",
                                        tile_sizes=tile_sizes,
                                        divides_evenly=True,
                                        tile_trivial=True,
                                        skew=True),
                            map_entry=map_entry)

        map_entry.map.schedule = dtypes.ScheduleType.GPU_Warp
        map_entry.map.label = "WarpMap"

    @staticmethod
    def annotates_memlets():
        return False