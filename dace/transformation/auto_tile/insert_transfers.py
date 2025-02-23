# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
from typing import Callable
import typing
import dace
from dace.properties import DictProperty, Property, SetProperty, make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

import operator

from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove


@make_properties
class InsertTransfers(transformation.SingleStateTransformation):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    location_requirements = DictProperty(
        key_type=str,
        value_type=tuple,
        default={
            "MMU": [
                (str(dace.dtypes.StorageType.Ascend_L2), 0),
                (str(dace.dtypes.StorageType.Ascend_L1), 0),
                (str(dace.dtypes.StorageType.Ascend_L2), 1),
                (str(dace.dtypes.StorageType.Ascend_L2), 1),
            ],
            "VECTOR": [
                (str(dace.dtypes.StorageType.Ascend_VECIN), 0),
                (str(dace.dtypes.StorageType.Ascend_L2), 1),
                (str(dace.dtypes.StorageType.Ascend_Global), 1),
                (str(dace.dtypes.StorageType.Register), 1),
            ],
        },
    )

    def __init__(self):
        super().__init__()

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        all_nodes = state.all_nodes_between(
            self.device_map_entry, state.exit_node(self.device_map_entry)
        )
        edges_to_add = set()
        nodes_to_add = set()
        edges_to_rm = set()
        for node in all_nodes:
            if isinstance(node, dace.nodes.MapEntry):
                map_entry = node
                print(map_entry, map_entry.computational_units)
                # If we have memlets with data types that are higher than the suggested distance, we need to insert transfers
                for compute_unit, distance in map_entry.computational_units.items():
                    if compute_unit == "VECTOR":
                        for e in state.in_edges(map_entry):
                            if e.data.data is not None:
                                in_arr = sdfg.arrays[e.data.data]
                                print(in_arr.storage)
                                print(map_entry.computational_units)
                                for lr in self.location_requirements[compute_unit]:
                                    if str(in_arr.storage) == lr[0]:
                                        if lr[1] > distance:
                                            lc = [(loc,dist) for (loc,dist) in self.location_requirements[compute_unit] if dist == 0]
                                            print("Must move:", in_arr, ", from: ", in_arr.storage, ", to:", lc)
                                            print(e.data)
                                            shape = [(end+1-beg)//step for beg, end, step in e.data.subset]
                                            nname = "VECIN_" + e.data.data
                                            sst = lc[0][0].split(".")[-1]
                                            recons_storage = dace.dtypes.StorageType._member_map_[sst]

                                            arrname, arr = sdfg.add_array(name=nname, shape=shape, dtype=in_arr.dtype, storage=recons_storage,
                                                                          transient=True)
                                            access_node = nodes.AccessNode(nname)
                                            nodes_to_add.add(access_node)
                                            edges_to_rm.add(e)
                                            edges_to_add.add((access_node, None, e.dst, e.dst_conn, dace.memlet.Memlet(expr=arrname)))
                                            if not isinstance(e.src, nodes.AccessNode):
                                                #access_node2 = nodes.AccessNode(e.data.data)
                                                #nodes_to_add.add(access_node2)
                                                #edges_to_add.add((e.src, None, access_node2, None, copy.deepcopy(e.data)))
                                                #edges_to_add.add((access_node2, None, access_node, None, copy.deepcopy(e.data)))
                                                edges_to_add.add((e.src, None, access_node, None, copy.deepcopy(e.data)))
                                            else:
                                                edges_to_add.add((e.src, None, access_node, None, copy.deepcopy(e.data)))

                                #self.location_requirements[str(in_arr.storage)]
                                #print(str(in_arr.storage) in )
        for e in edges_to_rm:
            state.remove_edge(e)
        for n in nodes_to_add:
            state.add_node(n)
        for e in edges_to_add:
            state.add_edge(*e)

        #raise Exception("Not implemented")

    @staticmethod
    def annotates_memlets():
        return True
