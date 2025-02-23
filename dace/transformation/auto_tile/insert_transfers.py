# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
from typing import Callable
import dace
from dace.properties import DictProperty, Property, SetProperty, make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

import operator


@make_properties
class InsertTransfers(transformation.SingleStateTransformation):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    location_requirements = DictProperty(
        key_type=str,
        value_type=str,
        default={"MMU": [("A2_L1", 0), ("B2_L1", 0), ("A1_L2", 1), ("B1_L2", 1)], "VECTOR": [("VECIN", 0)]},
    )

    def __init__(self):
        super().__init__()

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        all_nodes = state.all_nodes_between(self.device_map_entry, state.exit_node(self.device_map_entry))
        for node in all_nodes:
            if isinstance(node, dace.nodes.MapEntry):
                map_entry = node
                print(map_entry, map_entry.computational_units)
                # If we have memlets with data types that are higher than the suggested distance, we need to insert transfers
                for e in state.in_edges(map_entry):
                    if e.data.data is not None:
                        in_arr = sdfg.arrays[e.data.data]
                        print(in_arr.storage)
                        print(map_entry.computational_units)
                        if

        raise Exception("Not implemented")

    @staticmethod
    def annotates_memlets():
        return True
