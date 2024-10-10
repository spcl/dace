# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import dace
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import dtypes

@make_properties
class ReuseSharedMemory(transformation.SingleStateTransformation):

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    grid_strided_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.grid_strided_map_entry, cls.device_map_entry)]

    def can_be_applied(self, graph : SDFGState, expr_index, sdfg, permissive=False):
        thread_block_entry = self.thread_block_map_entry

        if thread_block_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return False
        if self.device_map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False
        if self.grid_strided_map_entry.map.schedule != dtypes.ScheduleType.Sequential:
            return False
        if graph.entry_node(self.grid_strided_map_entry) != self.device_map_entry:
            return False
        if graph.entry_node(self.thread_block_map_entry) != self.grid_strided_map_entry:
            return False
        for edge in graph.out_edges(self.device_map_entry):
            u, u_conn, v, v_conn, memlet = edge
            if v != self.grid_strided_map_entry:
                return False
        for edge in graph.out_edges(self.grid_strided_map_entry):
            u, u_conn, v, v_conn, memlet = edge
            if v != self.thread_block_map_entry:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
          state: SDFGState = graph
          nested_sdfg = dace.SDFG(state.label + "_compute")
          input_data = set()
          output_data = set()
          for conn in self.thread_block_map_entry.in_connectors:
            input_data.add(conn)
          for conn in graph.exit_node(self.device_map_entry).out_connectors:
            output_data.add(conn)
          added_nested_sdfg = state.add_nested_sdfg(sdfg=nested_sdfg, parent=state, inputs=input_data, outputs=output_data)

          for edge in graph.in_edges(self.thread_block_map_entry):
            u, u_conn, _, v_conn, memlet = edge
            graph.add_edge(u, u_conn, added_nested_sdfg, v_conn, memlet)
          for edge in graph.out_edges(graph.exit_node(self.thread_block_map_entry)):
            _, u_conn, v, v_conn, memlet = edge
            graph.add_edge(added_nested_sdfg, u_conn, v, v_conn, memlet)

    @staticmethod
    def annotates_memlets():
        return True


