# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from sympy import Integer
from dace import subsets
from dace.data import Array
from dace.properties import Property, make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
import typing
from typing import Dict
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import transformation
from dace import dtypes

@make_properties
class MemoryMovementNode(CodeLibraryNode):
    src_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Global, desc="Src location")
    dst_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Global, desc="Dst location")
    input_name = Property(dtype=str, default=None, desc="")
    output_name = Property(dtype=str, default=None, desc="")
    offset = Property(dtype=str, default="0", desc="")
    load_len = Property(dtype=typing.Union[Integer, int], default=Integer(32), desc="")
    num_threads = Property(dtype=typing.Union[Integer, int], default=Integer(32), desc="")

    def __init__(self, name, input_name, output_name, src_location, dst_location, offset, load_len, num_threads, *args, **kwargs):
        self.src_location = src_location
        self.dst_location = dst_location
        self.offset = offset
        self.load_len = load_len
        self.num_threads = num_threads
        self.input_name = input_name
        self.output_name = output_name
        super().__init__(name=name, input_names=[self.input_name], output_names=[self.output_name])

    def generate_code(self, inputs: Dict[str, Array], outputs: Dict[str, Array]):
        assert len(inputs) == 1
        assert len(outputs) == 1

        # Construct for loops
        code = f"""//Code Library Node to Load 1D Memory
{{
    constexpr int num_threads = {self.num_threads};
    constexpr int load_len = {self.load_len};
    constexpr int load_iter = (num_threads + load_len - 1) / load_len;
    constexpr int load_remainder = num_threads % load_len;
    const int offset = {self.offset};
    const int tid = threadIdx.x;

    #pragma unroll
    for (int _mem_move_i = 0; _mem_move_i < load_iter; ++_mem_move_i) {{
        {self.output_name}[_mem_move_i*num_threads + tid] = {self.input_name}[_mem_move_i*num_threads + tid];
    }}

    if (tid < load_remainder){{
        {self.output_name}[load_iter*num_threads + tid] = {self.input_name}[load_iter*num_threads + tid];
    }}

    __syncthreads();
}}
"""
        return code


@make_properties
class ExplicitMemoryMove(transformation.SingleStateTransformation):
    """
    For all memlets that connect the outer map to inner map,
    The src location is inferred and the memory is moved explicitly using a library not the given memory location
    """
    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    memory_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Shared, desc="Destination memory location")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def update_names():
        pass

    def infer_source(self, graph: SDFGState, sdfg: SDFG, edge : MultiConnectorEdge[Memlet]):
        u, u_conn, _, _, _ = edge
        # Recurse back up to the first access node memlet comes from the derive the storage type
        # If a map, then find the corresponding in connector to the out connector and continue
        # If an access node, return the storage type of it
        # Other continue upwards, assert the number of in and out edges are one (for the other case
        # might need to find a fix)
        while(True):
          if isinstance(u, nodes.AccessNode):
              return (u.data, sdfg.arrays[u.data].storage)
          elif isinstance(u, nodes.MapEntry):
              in_conn_name = "IN" + u_conn[3:]
              # Check the in_conn exists
              found = False
              for in_conn in u.in_connectors:
                  if in_conn_name == in_conn:
                      found = True
                      break
              if not found:
                  raise Exception("In connector not found, check implementation of explicit memory move")
              found = False
              for in_edge in graph.in_edges(u):
                  iu, iu_conn, iv, iv_conn, imemlet = in_edge
                  if iv_conn == in_conn_name:
                    u, u_conn, _, _, _ = iu, iu_conn, iv, iv_conn, imemlet
                    found = True
                    break
              assert(found)
          elif graph.in_degree(u) == 0: # Final node
              # Could be const memory or something like that, return None
              return (None, None)
          else:
              assert(graph.in_degree(u) == 1 and graph.out_degree(u) == 1)
              u, u_conn, _, _, _ = graph.in_edges(u)[0]

    def apply(self, graph: SDFGState, sdfg: SDFG):
        for edge in graph.out_edges(self.outer_map_entry):
          src_arr_name, src_storage_type_of_memlet = self.infer_source(graph=graph, sdfg=sdfg, edge=edge)
          dst_arr_name = f"shr{src_arr_name}"
          # The current state (and assumptions), the other edges connect outer map to the inner map,
          # With a volume and subset, this needs to go through a library node now that writes to an access node of the new type
          u, u_conn, v, v_conn, memlet = edge
          
          parsed_storage_type = f"{src_storage_type_of_memlet}".split(".")[-1]
          parsed_memory_location = f"{self.memory_location}".split(".")[-1]
          lib_node = MemoryMovementNode(name=f"move_{memlet.data}_from_{parsed_storage_type}_to_{parsed_memory_location}",
                                        input_name="IN" + u_conn[3:],
                                        output_name=u_conn,
                                        src_location=src_storage_type_of_memlet,
                                        dst_location=self.memory_location,
                                        offset=str(self.outer_map_entry.map.params[0]),
                                        load_len=self.outer_map_entry.map.range[0][2],
                                        num_threads=self.outer_map_entry.map.range[0][2])

          graph.add_node(lib_node)
          graph.remove_edge(edge)

          # We have a new access node after the code library node, new data too
          src_arr : Array = sdfg.arrays[src_arr_name]
          # The memlet subset of from i -> i + 128 is used to derive the shape (remove i)
          shape = []
          access_str = ""
          for r in self.outer_map_entry.map.range:
              (dev_begin, _, dev_step) = r
              shape.append(dev_step)
              access_str = f"{dev_begin}:({dev_begin}+{dev_step}), "
          (dst_arr_name, dst_arr) = sdfg.add_array(name=dst_arr_name, dtype=src_arr.dtype, shape=shape, transient=True, storage=self.memory_location)
          dst_access_node = nodes.AccessNode(data=dst_arr_name)
          to_dst_memlet = Memlet(f"{dst_arr_name}[{access_str[:-2]}]")
          to_map_memlet = Memlet(f"{dst_arr_name}[{access_str[:-2]}]")

          # Outer Map -> Lib Node
          graph.add_edge(u, u_conn, lib_node, "IN" + u_conn[3:], memlet)

          # Lib Node -> Access Node
          graph.add_edge(lib_node, u_conn, dst_access_node, None, to_dst_memlet)

          # Acces Node -> Inner Map
          graph.add_edge(dst_access_node, None, v, v_conn, to_map_memlet)

          # Now we have to update every other memlet until we reach the map exit
          # Start from the inner map
          itnodes = set()
          itnodes.add(self.inner_map_entry)
          while len(itnodes) != 0:
            node = itnodes.pop()
            for out_edge in graph.out_edges(node):
                ou, ou_conn, ov, ov_conn, omemlet = out_edge
                if isinstance(ov, nodes.MapExit):
                  continue
                assert(not isinstance(ov, nodes.MapExit))
                itnodes.add(ov)
                if omemlet.data == src_arr_name:
                  subset_range : subsets.Range = omemlet.subset
                  new_access_str = f"{dst_arr_name}["
                  for map_param, sr in zip(self.outer_map_entry.params, subset_range):
                      (beg, end, step) = sr
                      map_len = end - beg
                      new_access_str += f"{beg}-{map_param}:1+{beg}+{(map_len+step-1)//step}-{map_param}:1"
                  new_access_str = new_access_str[:-2] + "]"
                  nm = Memlet(expr=new_access_str, data=dst_arr_name)
                  graph.remove_edge(out_edge)
                  graph.add_edge(ou, ou_conn, ov, ov_conn, nm)

    @staticmethod
    def annotates_memlets():
        return True
