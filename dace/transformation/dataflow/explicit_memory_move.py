# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import sympy
from dace import subsets
import dace
from dace.data import Array
from dace.properties import Property, make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
from typing import Dict, List
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import transformation
from dace import dtypes
import functools
import operator
from operator import mul
import warnings

@make_properties
class MemoryMovementNode(CodeLibraryNode):
    src_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Global, desc="Src location")
    dst_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Global, desc="Dst location")
    input_name = Property(dtype=str, default="", desc="")
    output_name = Property(dtype=str, default="", desc="")
    offsets = Property(dtype=List, default=[], desc="")
    load_lengths = Property(dtype=List, default=[], desc="")
    thread_dims = Property(dtype=List, default=[], desc="")
    in_arr = Property(dtype=str, default="", desc="")
    out_arr = Property(dtype=str, default="", desc="")
    global_tensor_dims = Property(dtype=List, default=[], desc="")

    def __init__(self, name, input_name, output_name, src_location, dst_location, offsets, load_lengths, thread_dims, global_tensor_dims, in_arr, out_arr, *args, **kwargs):
        self.src_location = src_location
        self.dst_location = dst_location
        self.offsets = offsets
        self.load_lengths = load_lengths
        self.thread_dims = thread_dims
        self.global_tensor_dims = global_tensor_dims
        self.input_name = input_name
        self.output_name = output_name
        self.in_arr = in_arr
        self.out_arr = out_arr
        super().__init__(name=name, input_names=[self.input_name], output_names=[self.output_name])

    def generate_code(self, inputs: Dict[str, Array], outputs: Dict[str, Array]):
        assert len(inputs) == 1
        assert len(outputs) == 1
        assert len(self.load_lengths)  >= 1
        assert len(self.offsets) == len(self.load_lengths)
        dims_to_load = len(self.load_lengths)
        line_length = self.load_lengths[0]
        total_memory_to_load = functools.reduce(operator.mul, self.load_lengths)
        assert total_memory_to_load % line_length == 0
        num_lines = total_memory_to_load // line_length
        self.thread_dims = [1 if x == 0 else x for x in self.thread_dims]
        num_threads = functools.reduce(operator.mul, self.thread_dims)
        lines_fitting_to_num_threads = num_threads // line_length
        if len(self.load_lengths) > 1 and lines_fitting_to_num_threads > self.load_lengths[1]:
          lines_fitting_to_num_threads = self.load_lengths[1]
        if num_threads % line_length != 0 and line_length % num_threads != 0:
            warnings.warn(f"Memory move requested involves moving contiguous memory blocks ({num_lines}) of length ({line_length}) with"
                            f" {num_threads} threads. Attempt to arrange the parameters such that a line of contiguous memory ({line_length})"
                            f" is a multiple of (or divides evenly) the number of threads ({num_threads}) if possible")

        # The code is a loop of N dimensions (equal to the given length of offsets and load lengths)
        # If the lines fitting is >= 1 then X line at a time is loaded, where X is the number of lines fit to the
        # number of threads available.
        # If the lines fitting is == 0 then all threads collectively load a line with a possible remainder at time
        # 
        # Depending on the number of lines and fitting lines the innermost loop is unrolled

        formatted_load_lengths = [f"constexpr int load_len_d{i} = {load_len};" for i, load_len in enumerate(self.load_lengths)]
        # Load first dimension in contiguous lines, lest to loop
        formatted_for_loops_open_fitting_to_num_threads_is_0 = list(reversed([f"#pragma unroll\nfor (int i_d{i+1} = 0; i_d{i+1} < {load_len}; ++i_d{i+1}){{" \
                                                                               for i, load_len in enumerate(self.load_lengths[1:])]))
        formatted_for_loops_close_fitting_to_num_threads_is_0  = ["}" for _ in self.load_lengths[1:]]


        formatted_for_loops_open_fitting_to_num_threads_is_geq_2 = [] if len(self.load_lengths) <= 1 else \
          [f"#pragma unroll\nfor (int i_d1 = 0; i_d1 < {self.load_lengths[1]}; i_d1 += {lines_fitting_to_num_threads}){{"]
        if len(self.load_lengths) > 2:
          formatted_for_loops_open_fitting_to_num_threads_is_geq_2 +=  [f"#pragma unroll\nfor (int i_d{i+2} = 0; i_d{i+2} < {load_len}; ++i_d{i+2}){{" for i, load_len in enumerate(self.load_lengths[2:])]
        formatted_for_loops_open_fitting_to_num_threads_is_geq_2 = list(reversed(formatted_for_loops_open_fitting_to_num_threads_is_geq_2))
        formatted_for_loops_close_fitting_to_num_threads_is_geq_2  = ["}" for _ in self.load_lengths[1:]]

        # To access an n dimensional tensor (contigously stored in 1d memory in row major fashion),
        # The offset can be computed for for the dimensions 1 -> n-1 (0 excluded)
        # 1d_offset_d0 = load_offset_d0 + i_d0 * num_threads
        # 1d_offset_d1 = global_dim_d0 * (load_offset_d1 + i_d1)
        # 1d_offset_d2 = (global_dim_d0 * global_dim_d1) * (load_offset_d2 + i_d2)
        # and so on...
        # This means we can use a partially accumulated array
        global_offsets = [0] + [" * ".join(self.global_tensor_dims[:i]) for i in range(1, len(self.global_tensor_dims))]
        shared_offsets = [0] + [functools.reduce(operator.mul, self.load_lengths[:i]) for i in range(1, len(self.load_lengths))]

        global_access_offsets = [f"const int glb_at_d{i+1} = {global_offsets[i+1]} * (i_d{i+1});" for i in range(len((self.offsets[1:])))]
        ats = ["0"] + [f"glb_at_d{i+1}" for i in range(len(self.offsets[1:]))]
        global_access_offset = f"const int glb_access_offset = {' + '.join(ats)}"

        d1_global_offset = f"{self.global_tensor_dims[0]} * line_num"

        shared_access_offsets = [f"const int shr_at_d{i+1} = {shared_offsets[i+1]} * (i_d{i+1});" for i in range(len((self.offsets[1:])))]
        ats = ["0"] +  [f"shr_at_d{i+1}" for i in range(len(self.offsets[1:]))]
        local_access_offset = f"const int shr_access_offset = {' + '.join(ats)}"

        d1_shared_offset = "0" if len(shared_offsets) <= 1 else f"{shared_offsets[1]} * line_num"

        # Construct for loops
        code = f"""//Code Library Node to Load Memory from GPU Global to GPU Shared
{{
    constexpr int num_threads = {num_threads};
    // Load lengths for each dimension
    {"\n".join(formatted_load_lengths)}
    // Load lengths for each dimension end
    constexpr int lines_fitting_to_num_threads = {lines_fitting_to_num_threads};
    constexpr int load_iter_d0 = {line_length // num_threads if lines_fitting_to_num_threads == 0 else 1};
    constexpr int load_remainder_d0 = {num_threads % self.load_lengths[0]};
    constexpr int active_load_threads = {line_length * lines_fitting_to_num_threads if lines_fitting_to_num_threads > 0 else num_threads};
    constexpr int dimensions_to_load = {dims_to_load}; 

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int line_num = tid / {line_length};
    const int line_offset = tid % {line_length};

    if constexpr (lines_fitting_to_num_threads == 0 || dimensions_to_load == 1){{
        {"\n".join(formatted_for_loops_open_fitting_to_num_threads_is_0)}
        // Global access offsets
        {"\n".join(global_access_offsets)}
        // Shared access offsets
        {"\n".join(shared_access_offsets)}
        // Joined access offset
        {global_access_offset};
        {local_access_offset};
        #pragma unroll
        for (int i_d0 = 0; i_d0 < load_iter_d0; ++i_d0){{
            {self.output_name}[i_d0*num_threads + tid] = {self.input_name}[i_d0*num_threads + tid];
        }}
        if constexpr (load_remainder_d0 != 0){{
            if (tid < load_remainder_d0){{
                if (load_iter_d0*num_threads + tid < {self.global_tensor_dims[0]}){{
                  {self.output_name}[load_iter_d0*num_threads + tid] = {self.input_name}[load_iter_d0*num_threads + tid];
                }}
            }}
        }}
        {"\n".join(formatted_for_loops_close_fitting_to_num_threads_is_0)}
    }} else {{
        {"\n".join(formatted_for_loops_open_fitting_to_num_threads_is_geq_2)}
        // Global access offsets
        {"\n".join(global_access_offsets)}
        // Shared access offsets
        {"\n".join(shared_access_offsets)}
        // Joined global access offset
        {global_access_offset} + ({d1_global_offset});
        {local_access_offset} + ({d1_shared_offset});
        if (tid < active_load_threads){{
            {self.output_name}[shr_access_offset + line_offset] = {self.input_name}[glb_access_offset + line_offset];
        }}
        {"\n".join(formatted_for_loops_close_fitting_to_num_threads_is_geq_2)}
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
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    grid_strided_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    memory_location = Property(dtype=dtypes.StorageType, default=dtypes.StorageType.GPU_Shared, desc="Destination memory location")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry, cls.grid_strided_map_entry, cls.thread_block_map_entry)]

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

    location_to_prefix = {
        dtypes.StorageType.GPU_Global: "glb",
        dtypes.StorageType.GPU_Shared: "shr",
    }

    def has_minus_sign(self, term):
        if term.is_Mul:
            coeff, _ = term.as_coeff_mul()
            return coeff < 0
        elif term.is_Number:
            return term < 0
        return False

    def filter_map_params_from_subset(self, subset_range : subsets.Range, dev_map_params : set,
                                      grid_map_params : set, thread_block_map_params : set, src_arr_shape : tuple, 
                                      dst_arr_shape : tuple, map_mode : int):
      assert(map_mode == 1 or map_mode == 0)
      exprs = []
      for sid, sr in enumerate(subset_range):
        (beg, end, step) = sr
        for i, expr in enumerate([beg, end, step]):
          param_signs = dict()
          symbols_to_remove = dev_map_params + grid_map_params
          expanded_expr = expr.expand()
          terms = expanded_expr.as_ordered_terms()
          filtered_terms = []
          for term in terms:
              has = False
              for sub_term in term.as_ordered_factors():
                for sym in symbols_to_remove:
                  if sub_term.has(sym) or str(sub_term) == str(sym):
                    has = True
                for sym_to_check_sign in thread_block_map_params:
                  param_signs[str(sym_to_check_sign)] = not self.has_minus_sign(term)
              if not has:
                filtered_terms.append(term)
          filtered_expr = sympy.Add(*filtered_terms)
          # If the inner loop begins from the variable of the loop above
          # Then we need to subtract the subset,
          # If the variable is negative we need to add
          if map_mode == 1 and (i == 1 or i == 0):
            if next(iter(param_signs.values())) == True:
              filtered_expr = filtered_expr - dace.symbolic.symbol(grid_map_params[sid])
            else:
              filtered_expr = filtered_expr + dace.symbolic.symbol(grid_map_params[sid])
          for old_dim, new_dim in zip(src_arr_shape, dst_arr_shape):
            filtered_expr = filtered_expr.subs(old_dim, new_dim)
          exprs.append(filtered_expr)
      return exprs

    def gen_access_str_from_exprs(self, exprs: List, dst_arr_name: str):
      split_expr = [exprs[i:i + 3] for i in range(0, len(exprs), 3)]
      new_access_str = f"{dst_arr_name}["
      # Sympy is end range that is included, +1 for correct range computation
      for range_expr in split_expr:
        new_access_str += f"{range_expr[0]}:{range_expr[1]+1}:{range_expr[2]}, "
      new_access_str = new_access_str[:-2] + "]"
      return new_access_str

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_mode = 1 # If the map mode is 1 (same as used by map tiling transformations)
        # Inner maps range depends on the outer map varaible, this requries different offset calculations
        # Of mode is 0, then it means all maps start from range 0, and the subset calcualtions need to consider
        # every type of access, then, subset computations need to adapt differently

        for edge in graph.out_edges(self.grid_strided_map_entry):
          src_arr_name, src_storage_type_of_memlet = self.infer_source(graph=graph, sdfg=sdfg, edge=edge)
          dst_arr_name = self.location_to_prefix.get(self.memory_location, "") + f"{src_arr_name}"
          # The current state (and assumptions), the other edges connect outer map to the inner map,
          # With a volume and subset, this needs to go through a library node now that writes to an access node of the new type
          u, u_conn, v, v_conn, memlet = edge

          parsed_storage_type = f"{src_storage_type_of_memlet}".split(".")[-1]
          parsed_memory_location = f"{self.memory_location}".split(".")[-1]
          # Build the offsets, load_lengths, num_threads 
          # DaCe maps of form i0, i1, i2, i3 are mapped to the GPU block dimensions
          # (i0 x i1) -> z, i2 -> y, i3 -> x, therefore we iterate from back up to three
          offsets = ["0"] * len(self.grid_strided_map_entry.map.range)
          load_lengths = [0] * len(self.grid_strided_map_entry.map.range)
          num_threads = [0, 0, 0]
          for i in range(len(self.grid_strided_map_entry.map.range), 0, -1):
              (_, grid_end, grid_step) = self.grid_strided_map_entry.map.range[-i]
              (tblock_beg, tblock_end, tblock_step) = self.thread_block_map_entry.map.range[-i]
              # Both statements need to be simplifiable to integers
              num_thread = sympy.simplify(((tblock_end+1) - tblock_beg) // tblock_step)
              load_len = sympy.simplify((tblock_end+1) - tblock_beg)
              # Should not stay in sympy type Zero
              if isinstance(num_thread, sympy.Integer):
                  num_thread = int(num_thread)
              else:
                  raise Exception("Number of threads expression needs to be simplifiable to an integer for the memory movement trasnformation")
              if isinstance(load_len, sympy.Integer):
                  load_len = int(load_len)
              else:
                  raise Exception("The length of data need to be loaded needs to be simplifiable to an integer for the memory movement trasnformation")
              if (i < 3):
                num_threads[-i] = num_thread
              load_lengths[-i] = load_len
              offsets[-i] = str(self.device_map_entry.map.params[-i])

          mem_loc_a = parsed_storage_type
          mem_loc_b = parsed_memory_location
          lib_node_name = f"move_{memlet.data}_from_{mem_loc_a}_to_{mem_loc_b}"
          lib_node = MemoryMovementNode(name=lib_node_name,
                                        input_name="IN" + u_conn[3:],
                                        output_name=u_conn,
                                        src_location=src_storage_type_of_memlet,
                                        dst_location=self.memory_location,
                                        offsets=list(reversed(offsets)),
                                        load_lengths=list(reversed(load_lengths)),
                                        thread_dims=list(reversed(num_threads)),
                                        global_tensor_dims=list(reversed([str(d) for d in sdfg.arrays[src_arr_name].shape])),
                                        in_arr=src_arr_name,
                                        out_arr=dst_arr_name)
          if map_mode == 0 and len(load_lengths) > 3:
            raise Exception("Inner maps (other than the outer-most device scheduled map) starting from 0 is not support above for rank > 3 tensors")

          graph.add_node(lib_node)
          graph.remove_edge(edge)

          src_arr : Array = sdfg.arrays[src_arr_name]
          shape = []
          for dev_range, grid_range, thread_block_range in zip(self.device_map_entry.map.range, self.grid_strided_map_entry.map.range, self.thread_block_map_entry.map.range):
              (_, _, dev_step) = dev_range
              (grid_begin, grid_end, grid_step) = grid_range
              (tblock_begin, tblock_end, tblock_step) = thread_block_range
              num_thread = int(sympy.simplify((tblock_end+1) - tblock_begin))
              grid_len = int(sympy.simplify((grid_end+1) - grid_begin))
              assert(dev_step % grid_step == 0)
              assert(dev_step % grid_len == 0)
              # It menas the grid strided loop is for example g=i:i+256:128 and dev step 256
              if (dev_step // grid_len == 1):
                shape.append(grid_step)
                map_mode = 1
              else: # Meaning that every map starts from 0
                shape.append(dev_step // grid_len)
                map_mode = 0

          memlet_to_lib_node : Memlet = memlet
          exprs = self.filter_map_params_from_subset(memlet_to_lib_node.subset, 
                                                     self.device_map_entry.map.params,
                                                     self.grid_strided_map_entry.map.params,
                                                     self.thread_block_map_entry.map.params,
                                                     src_arr.shape,
                                                     shape,
                                                     0)
          new_access_str = self.gen_access_str_from_exprs(exprs, dst_arr_name)
          (dst_arr_name, dst_arr) = sdfg.add_array(name=dst_arr_name, dtype=src_arr.dtype, shape=shape, transient=True, storage=self.memory_location)
          dst_access_node = nodes.AccessNode(data=dst_arr_name)
          to_dst_memlet = Memlet(expr=new_access_str, data=dst_arr_name)
          to_map_memlet = Memlet(expr=new_access_str, data=dst_arr_name)

          # Outer Map -> Lib Node
          graph.add_edge(u, u_conn, lib_node, "IN" + u_conn[3:], memlet)

          # Lib Node -> Access Node
          graph.add_edge(lib_node, u_conn, dst_access_node, None, to_dst_memlet)

          # Acces Node -> Inner Map
          graph.add_edge(dst_access_node, None, v, v_conn, to_map_memlet)

          # Now we have to update every other memlet until we reach the map exit
          # Start from the inner map
          itnodes = set()
          itnodes.add(self.thread_block_map_entry)
          while len(itnodes) != 0:
            node = itnodes.pop()
            for out_edge in graph.out_edges(node):
                out_u, out_u_conn, out_v, out_v_conn, out_memlet = out_edge
                if isinstance(out_v, nodes.MapExit):
                  continue
                assert(not isinstance(out_v, nodes.MapExit))
                itnodes.add(out_v)
                if out_memlet.data == src_arr_name:
                  subset_range : subsets.Range = out_memlet.subset
                  # Filter anything from the subset string of the memlet that involves terms that include
                  # Patameters from the grid-strided or device maps, this way effectively one computes the offset mapping
                  # From the global tensor to shared memory tensor
                  exprs = self.filter_map_params_from_subset(subset_range, 
                                                             self.device_map_entry.map.params,
                                                             self.grid_strided_map_entry.map.params,
                                                             self.thread_block_map_entry.map.params,
                                                             src_arr.shape,
                                                             shape,
                                                             map_mode)
                  # For each dimension we get (beg:end:step) expressions
                  new_access_str = self.gen_access_str_from_exprs(exprs, dst_arr_name)
                  # Update the state with the new memlet annotation
                  updated_memlet = Memlet(expr=new_access_str, data=dst_arr_name)
                  graph.remove_edge(out_edge)
                  graph.add_edge(out_u, out_u_conn, out_v, out_v_conn, updated_memlet)

    @staticmethod
    def annotates_memlets():
        return True
