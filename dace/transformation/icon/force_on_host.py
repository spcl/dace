# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import copy
from dace import subsets
import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import ListProperty, make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@make_properties
class ForceOnHost(transformation.SingleStateTransformation):
    i = 0
    map_entry = transformation.PatternNode(nodes.MapEntry)
    access_names = ListProperty(element_type=str)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state : SDFGState, expr_index, sdfg, permissive=False):
        if self.map_entry.map.schedule != dace.ScheduleType.Default:
            return False

        map_exit = state.exit_node(self.map_entry)
        in_edges = state.in_edges(self.map_entry)
        out_edges = state.out_edges(map_exit)

        for node in [u for u,_,_,_,_ in in_edges] + [v for _,_,v,_,_ in out_edges]:
            if isinstance(node, nodes.AccessNode) and node.data in self.access_names:
                return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        self.map_entry.map.schedule = dace.ScheduleType.Sequential
        self.map_entry.map.host_map = True
        map_exit = state.exit_node(self.map_entry)

        ans = [u for u,_,_,_,_ in state.in_edges(self.map_entry)] + [v for _,_,v,_,_ in state.out_edges(map_exit)]
        data_desc_s = [node.desc(sdfg) for node in ans]
        for data_desc in data_desc_s:
            data_desc.host_data = True

        in_edges = state.in_edges(self.map_entry)
        out_edges = state.out_edges(map_exit)
        for node in [u for u,_,_,_,_ in in_edges] + [v for _,_,v,_,_ in out_edges]:
            if isinstance(node, nodes.AccessNode):
                sdfg.arrays[node.data].storage = dace.StorageType.Default


    def annotates_memlets():
        return False
