# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import ListProperty, Property, make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
from functools import reduce


@make_properties
class AddComputeElementBlockMap(transformation.SingleStateTransformation):
    """
    Adds a specific schedule map below a another map scope
    """

    compute_element_group_dims = ListProperty(
        element_type=int, default=[32, 1, 1], desc="Dimensions of the thread group"
    )
    map_schedule = Property(
        dtype=dace.dtypes.ScheduleType,
        default=None,
        allow_none=True,
        desc="Parent scope",
    )
    map_entry = transformation.PatternNode(dace.nodes.MapEntry)
    schedule_to_add = Property(
        dtype=dace.dtypes.ScheduleType,
        default=None,
        allow_none=True,
        desc="Schedule type to add",
    )

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Can be applied to any map, whether it makes sense is user's responsibility
        # Some GPU checks
        if (
            self.map_entry is None
            or self.schedule_to_add is None
            or self.map_schedule is None
        ):
            return False

        if self.schedule_to_add in dace.dtypes.ScheduleType.GPU_ThreadBlock:
            if len(self.compute_element_group_dims) > 3:
                return False
            if reduce(self.compute_element_group_dims) > 1024:
                return False
        # Passing it double ensures the user know what they are doing
        if self.map_entry.map.schedule != self.map_schedule:
            return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry: dace.nodes.MapEntry = self.map_entry

        block_dims = self.compute_element_group_dims

        # The thread block sizes depend on the number of dimensions we have
        # GPU code gen maps the params i0:...,i1:...,i2:... respectively to blockDim.z,.y,.x
        # If more tile sizes are given than the available number of parameters cull the list and ignore
        # the additional parameters

        # tiles: 1, 2, 3, map: A, B, C, D, E
        # becomes A, B, C3, D2, E1
        # tiles: 1, 2, 3, map: A, B
        # becomes A2, B1
        tile_sizes = []

        map_entry_len = len(map_entry.map.range)
        block_dim_len = len(block_dims)
        if map_entry_len > block_dim_len:
            # tiles: 1, 2, 3, map: A, B, C, D, E
            # becomes A, B, C3, D2, E1
            tile_sizes = [1] * (map_entry_len - block_dim_len) + list(reversed(block_dims))
        elif map_entry_len == block_dim_len:
            tile_sizes = list(reversed(block_dims))
        else:
            # tiles: 1, 2, 3, map: A, B
            # becomes A2, B1
            tile_sizes  = list(reversed(block_dims[:map_entry_len]))

        # Tile trivial simplifies come checks for the BlockCoarsening and ThreadCoarsening transformations
        MapTiling.apply_to(
            sdfg=sdfg,
            options=dict(
                prefix="b",
                tile_sizes=tile_sizes,
                divides_evenly=True,
                tile_trivial=True,
                skew=True,
            ),
            map_entry=map_entry,
        )

        map_entry.map.schedule = self.schedule_to_add
        map_entry.map.label = self.schedule_to_add.name + "Map"

        """
        for m in [map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d
        """

        # The dev map is a new map where the gpu_block_size param is not transferred over
        prev_entry = state.entry_node(map_entry)
        prev_entry.map.label = self.map_schedule.name + "Map"

    @staticmethod
    def annotates_memlets():
        return False