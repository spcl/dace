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
    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    {f"""
        constexpr int load_iter_d0 = {line_length // num_threads if lines_fitting_to_num_threads == 0 else 1};
        constexpr int load_remainder_d0 = {num_threads % self.load_lengths[0]};
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
      """ if lines_fitting_to_num_threads == 0 or dims_to_load == 1 else 
      f"""
        constexpr int lines_fitting_to_num_threads = {lines_fitting_to_num_threads};
        constexpr int active_load_threads = {line_length * lines_fitting_to_num_threads if lines_fitting_to_num_threads > 0 else num_threads};
        const int line_num = tid / {line_length};
        const int line_offset = tid % {line_length};
        {"\n".join(formatted_for_loops_open_fitting_to_num_threads_is_geq_2)}
        // Global access offsets
        {"\n".join(global_access_offsets)}
        // Shared access offsets
        {"\n".join(shared_access_offsets)}
        // Joined global access offset
        {global_access_offset} + ({d1_global_offset});
        {local_access_offset} + ({d1_shared_offset});
        if (tid < active_load_threads && (glb_access_offset + line_offset) < {self.global_tensor_dims[0]}){{
            {self.output_name}[shr_access_offset + line_offset] = {self.input_name}[glb_access_offset + line_offset];
        }}
        {"\n".join(formatted_for_loops_close_fitting_to_num_threads_is_geq_2)}
      """
    }

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
        # Recurse back up to the first access node memlet comes from the derive the storage type
        # If a map, then find the corresponding in connector to the out connector and continue
        # If an access node, return the storage type of it
        # Other continue upwards, assert the number of in and out edges are one (for the other case
        # might need to find a fix)
        u, u_conn, _, _, _ = edge
        while(True):
          print(u, u_conn)
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

    def split_min_expression(self, expr):
        if expr.func == sympy.Min:
            return expr.args
        else:
            return expr

    def incr_expression(self, term):
      return tuple(x + 1 for x in term)

    def filter(self, expressions, symbol):
      found_expressions = set()
      for expr in expressions:
        free_symbols = expr.free_symbols
        free_symbol_strings = [str(x) for x in free_symbols]
        if expr.has(symbol) or str(symbol) in free_symbol_strings:
          found_expressions.add(expr)

      if len(found_expressions) != 1:
        raise Exception(f"After filter len should be 1, the length is {len(found_expressions)}: filtered: {expressions}, for the symbol: {symbol}")

      return found_expressions.pop()

    def filter_map_params_from_subset(self, subset_range : subsets.Range, dev_map_params : set,
                                      grid_map_params : set, thread_block_map_params : set, 
                                      thread_block_map : nodes.Map,  src_arr_shape : tuple, 
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
          # Since the order of parameters do not need to be the same (e.g., transposed access)
          # We need to find the order of it, this can be done by check the initial value of the inner param
          if map_mode == 1 and (i == 1 or i == 0):
            params_used_in_access = list(expr.free_symbols)
            # Assume thread block map j0=g0...,j1=g1...
            # But we access shrA[j1,j0] (instead of j0, j1)
            # Then we need to subset g1 from j1, first index.
            # For this we find the ordering of param j1, and then
            # get the param g1 with the same id and substract that.
            # Filter the params that appear in the thread block map, to avoid
            # issues with for example M - i0, where we iterate from reverse
            tblock_param_strs = [str(param) for param in thread_block_map_params]
            used_param_strs = [str(param) for param in params_used_in_access]
            params_used_from_tblock_params = []
            for i, used_param_str in enumerate(used_param_strs):
              if used_param_str in tblock_param_strs:
                params_used_from_tblock_params.append(params_used_in_access[i])

            for used_param in params_used_from_tblock_params:
              corresponding_grid_param = None
              corresponding_sign = None
              for tb_id, tblock_param in enumerate(thread_block_map.params):
                if used_param == tblock_param or str(used_param) == str(tblock_param):
                  corresponding_grid_param = thread_block_map.range[tb_id][0]
                  corresponding_sign = param_signs[str(used_param)]
                  break
              assert(corresponding_grid_param != None and corresponding_sign != None)
              if corresponding_sign == True:
                filtered_expr = filtered_expr - corresponding_grid_param
              else:
                filtered_expr = filtered_expr + corresponding_grid_param
          for old_dim, new_dim in zip(src_arr_shape, dst_arr_shape):
            filtered_expr = filtered_expr.subs(old_dim, new_dim)
          print(filtered_expr)
          exprs.append(filtered_expr)
      return exprs

    def filter_map_params_from_subset_2(self, subset_range : subsets.Range, 
                                        params_to_remove : set, 
                                        current_map : nodes.Map,  src_arr_shape : tuple, 
                                        dst_arr_shape : tuple, map_mode : int):
      assert(map_mode == 1 or map_mode == 0)
      exprs = []
      for sid, sr in enumerate(subset_range):
        (beg, end, step) = sr
        print("SE", params_to_remove)
        print("tes", beg, end, step)
        for i, expr in enumerate([beg, end, step]):
          param_signs = dict()
          symbols_to_remove = params_to_remove
          expanded_expr = expr.expand()
          terms = expanded_expr.as_ordered_terms()
          filtered_terms = []
          for term in terms:
              has = False
              for sub_term in term.as_ordered_factors():
                for sym in symbols_to_remove:
                  if sub_term.has(sym) or str(sub_term) == str(sym):
                    has = True
                for sym_to_check_sign in current_map.params:
                  param_signs[str(sym_to_check_sign)] = not self.has_minus_sign(term)
              if not has:
                filtered_terms.append(term)
          filtered_expr = sympy.Add(*filtered_terms)
          print("FE", filtered_expr)
          # If the inner loop begins from the variable of the loop above
          # Then we need to subtract the subset,
          # If the variable is negative we need to add
          # Since the order of parameters do not need to be the same (e.g., transposed access)
          # We need to find the order of it, this can be done by check the initial value of the inner param
          if map_mode == 1 and (i == 1 or i == 0):
            params_used_in_access = list(expr.free_symbols)
            # Assume thread block map j0=g0...,j1=g1...
            # But we access shrA[j1,j0] (instead of j0, j1)
            # Then we need to subset g1 from j1, first index.
            # For this we find the ordering of param j1, and then
            # get the param g1 with the same id and substract that.
            # Filter the params that appear in the thread block map, to avoid
            # issues with for example M - i0, where we iterate from reverse
            param_strs = [str(param) for param in current_map.params]
            used_param_strs = [str(param) for param in params_used_in_access]
            params_used_from_map_params = []
            print("1 ",param_strs)
            print("11", used_param_strs)
            for i, used_param_str in enumerate(used_param_strs):
              if used_param_str in param_strs:
                params_used_from_map_params.append(params_used_in_access[i])

            print("2 ",params_used_from_map_params)
            print("3 ",current_map.params)
            for used_param in params_used_from_map_params:
              corresponding_outer_param = None
              corresponding_outer_sign = None
              for tb_id, tblock_param in enumerate(current_map.params):
                if used_param == tblock_param or str(used_param) == str(tblock_param):
                  corresponding_outer_param = current_map.range[tb_id][0]
                  corresponding_outer_sign = param_signs[str(used_param)]
                  break
              assert(corresponding_outer_param != None and corresponding_outer_sign != None)
              if corresponding_outer_sign == True:
                filtered_expr = filtered_expr - corresponding_outer_param
              else:
                filtered_expr = filtered_expr + corresponding_outer_param
          for old_dim, new_dim in zip(src_arr_shape, dst_arr_shape):
            filtered_expr = filtered_expr.subs(old_dim, new_dim)
          print(filtered_expr)
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

    def filter_integer_part(self, expr):
      terms = expr.args
      int_part = None
      
      for term in terms:
          if term.is_Integer:
              int_part = term
              break
      return int_part

    a = 0
    def apply(self, graph: SDFGState, sdfg: SDFG):
        print("APPLC CALL NUM ", ExplicitMemoryMove.a)
        ExplicitMemoryMove.a += 1
        map_mode: int = 1 # If the map mode is 1 (same as used by map tiling transformations)
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
          num_threads = [int(t) for t in self.device_map_entry.map.gpu_block_size]
          shape = [0] * len(memlet.subset)
          for i, (beg, end, step) in enumerate(memlet.subset):
            assert(step == 1)
            assert(isinstance(sympy.simplify(end + 1- beg), sympy.Integer))
            offsets[i] = str(beg)
            load_lengths[i] = int(sympy.simplify(end + 1 - beg))
            shape[i] = int(sympy.simplify(end + 1 - beg))

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
                                        global_tensor_dims=list(reversed([str(d) if isinstance(d, sympy.Symbol) else int(d) for d in sdfg.arrays[src_arr_name].shape])),
                                        in_arr=src_arr_name,
                                        out_arr=dst_arr_name)

          graph.add_node(lib_node)
          graph.remove_edge(edge)


          src_arr : Array = sdfg.arrays[src_arr_name]

          memlet_to_lib_node : Memlet = memlet
          exprs = self.filter_map_params_from_subset_2(memlet_to_lib_node.subset, 
                                                     set.union(set(self.device_map_entry.map.params),
                                                      set(self.grid_strided_map_entry.map.params)),
                                                     self.thread_block_map_entry.map,
                                                     src_arr.shape,
                                                     shape,
                                                     0)
          print(exprs)
          new_access_str = self.gen_access_str_from_exprs(exprs, dst_arr_name)
          if not dst_arr_name in sdfg.arrays.keys():
            print(shape)
            (dst_arr_name, dst_arr) = sdfg.add_array(name=dst_arr_name, dtype=src_arr.dtype, shape=shape, transient=True, storage=self.memory_location)
          else:
            (dst_arr_name, dst_arr) = (dst_arr_name, sdfg.arrays[dst_arr_name])
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
          innermost_map = self.thread_block_map_entry.map
          outer_map = self.grid_strided_map_entry.map
          while len(itnodes) != 0:
            node = itnodes.pop()
            for out_edge in graph.out_edges(node):
                out_u, out_u_conn, out_v, out_v_conn, out_memlet = out_edge
                if isinstance(out_u, nodes.MapEntry):
                  outer_map = innermost_map
                  innermost_map = out_u.map
                if isinstance(out_v, nodes.MapExit):
                  continue
                assert(not isinstance(out_v, nodes.MapExit))
                itnodes.add(out_v)
                if out_memlet.data == src_arr_name:
                  subset_range : subsets.Range = out_memlet.subset
                  # Filter anything from the subset string of the memlet that involves terms that include
                  # Patameters from the grid-strided or device maps, this way effectively one computes the offset mapping
                  # From the global tensor to shared memory tensor
                  exprs = self.filter_map_params_from_subset_2(subset_range=subset_range, 
                                                                params_to_remove=set(),
                                                                current_map=innermost_map,
                                                                src_arr_shape=src_arr.shape,
                                                                dst_arr_shape=shape,
                                                                map_mode=1)
                  print(exprs)
                  # For each dimension we get (beg:end:step) expressions
                  new_access_str = self.gen_access_str_from_exprs(exprs, dst_arr_name)
                  # Update the state with the new memlet annotation
                  updated_memlet = Memlet(expr=new_access_str, data=dst_arr_name)
                  graph.remove_edge(out_edge)
                  graph.add_edge(out_u, out_u_conn, out_v, out_v_conn, updated_memlet)

    @staticmethod
    def annotates_memlets():
        return True
