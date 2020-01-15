""" Contains the FPGA Transform Map transformation. """

import copy
import itertools

from dace import data, dtypes, sdfg as sd, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


class FPGATransformMap(pattern_matching.Transformation):
    """ Implements the FPGATransformMap transformation.

        Converts a single map to an FPGA-scheduled map and creates FPGA arrays
        outside it, generating CPU<->FPGA memory copies automatically.
  """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(FPGATransformMap._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[FPGATransformMap._map_entry]]
        candidate_map = map_entry.map

        # No more than 3 dimensions
        if candidate_map.range.dims() > 3: return False

        # Map schedules that are disallowed to transform to FPGAs
        if (candidate_map.schedule in [
                dtypes.ScheduleType.MPI, dtypes.ScheduleType.GPU_Device,
                dtypes.ScheduleType.FPGA_Device,
                dtypes.ScheduleType.GPU_ThreadBlock
        ]):
            return False

        # Recursively check parent for FPGA schedules
        sdict = graph.scope_dict()
        current_node = map_entry
        while current_node is not None:
            if (current_node.map.schedule in [
                    dtypes.ScheduleType.GPU_Device,
                    dtypes.ScheduleType.FPGA_Device,
                    dtypes.ScheduleType.GPU_ThreadBlock
            ]):
                return False
            current_node = sdict[current_node]

        # Ensure that map does not include internal arrays that are allocated
        # on non-default space
        subgraph = graph.scope_subgraph(map_entry)
        for node in subgraph.nodes():
            if (isinstance(node, nodes.AccessNode)
                    and node.desc(sdfg).storage != dtypes.StorageType.Default):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[FPGATransformMap._map_entry]]

        return str(map_entry)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[FPGATransformMap._map_entry]]
        map_entry.map._schedule = dtypes.ScheduleType.FPGA_Device

        # Find map exit nodes
        exit_nodes = graph.exit_nodes(map_entry)

        fpga_storage_types = [
            dtypes.StorageType.FPGA_Global, dtypes.StorageType.FPGA_Local,
            dtypes.StorageType.CPU_Pinned
        ]

        #######################################################
        # Add FPGA copies of CPU arrays (i.e., not already on FPGA)

        # First, understand which arrays to clone
        all_out_edges = []
        for enode in exit_nodes:
            all_out_edges.extend(list(graph.out_edges(enode)))
        in_arrays_to_clone = set()
        out_arrays_to_clone = set()
        for e in graph.in_edges(map_entry):
            data_node = sd.find_input_arraynode(graph, e)
            if data_node.desc(sdfg).storage not in fpga_storage_types:
                in_arrays_to_clone.add(data_node)
        for e in all_out_edges:
            data_node = sd.find_output_arraynode(graph, e)
            if data_node.desc(sdfg).storage not in fpga_storage_types:
                out_arrays_to_clone.add(data_node)

        # Second, create a FPGA clone of each array
        cloned_arrays = {}
        in_cloned_arraynodes = {}
        out_cloned_arraynodes = {}
        for array_node in in_arrays_to_clone:
            array = array_node.desc(sdfg)
            if array_node.data in cloned_arrays:
                pass
            elif 'fpga_' + array_node.data in sdfg.arrays:
                pass
            else:
                sdfg.add_array(
                    'fpga_' + array_node.data,
                    dtype=array.dtype,
                    shape=array.shape,
                    materialize_func=array.materialize_func,
                    transient=True,
                    storage=dtypes.StorageType.FPGA_Global,
                    allow_conflicts=array.allow_conflicts,
                    strides=array.strides,
                    offset=array.offset)
                cloned_arrays[array_node.data] = 'fpga_' + array_node.data
            cloned_node = nodes.AccessNode('fpga_' + array_node.data)

            in_cloned_arraynodes[array_node.data] = cloned_node
        for array_node in out_arrays_to_clone:
            array = array_node.desc(sdfg)
            if array_node.data in cloned_arrays:
                pass
            elif 'fpga_' + array_node.data in sdfg.arrays:
                pass
            else:
                sdfg.add_array(
                    'fpga_' + array_node.data,
                    dtype=array.dtype,
                    shape=array.shape,
                    materialize_func=array.materialize_func,
                    transient=True,
                    storage=dtypes.StorageType.FPGA_Global,
                    allow_conflicts=array.allow_conflicts,
                    strides=array.strides,
                    offset=array.offset)
                cloned_arrays[array_node.data] = 'fpga_' + array_node.data
            cloned_node = nodes.AccessNode('fpga_' + array_node.data)

            out_cloned_arraynodes[array_node.data] = cloned_node

        # Third, connect the cloned arrays to the originals
        # TODO(later): Shift indices and create only the necessary sub-arrays
        for array_name, node in in_cloned_arraynodes.items():
            graph.add_node(node)
            for edge in graph.in_edges(map_entry):
                if edge.data.data == array_name:
                    graph.remove_edge(edge)
                    graph.add_edge(edge.src, None, node, None, edge.data)
                    newmemlet = copy.copy(edge.data)
                    newmemlet.data = node.data
                    graph.add_edge(node, edge.src_conn, edge.dst,
                                   edge.dst_conn, newmemlet)
        for array_name, node in out_cloned_arraynodes.items():
            graph.add_node(node)
            for edge in all_out_edges:
                if edge.data.data == array_name:
                    graph.remove_edge(edge)
                    graph.add_edge(node, None, edge.dst, None, edge.data)
                    newmemlet = copy.copy(edge.data)
                    newmemlet.data = node.data
                    graph.add_edge(edge.src, edge.src_conn, node,
                                   edge.dst_conn, newmemlet)

        # Fourth, replace memlet arrays as necessary
        scope_subgraph = graph.scope_subgraph(map_entry)
        for edge in scope_subgraph.edges():
            if (edge.data.data is not None
                    and edge.data.data in cloned_arrays):
                edge.data.data = cloned_arrays[edge.data.data]

    def modifies_graph(self):
        return True


# pattern_matching.Transformation.register_pattern(FPGATransformMap)
