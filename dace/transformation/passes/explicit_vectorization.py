# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import dace
import ast
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from dace import SDFG, Memlet, SDFGState, properties, transformation
from dace import typeclass
from dace.sdfg.graph import Edge
from dace.sdfg.nodes import CodeNode
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.passes import analysis as ap, pattern_matching as pmp
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import RemoveFPTypeCasts, RemoveIntTypeCasts, PowerOperatorExpansion
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes import InlineSDFGs
from dace.sdfg.fp_utils import change_fp_types
from dace.transformation.passes.explicit_vectorization_utils import *
import dace.sdfg.construction_utils as cutil


@properties.make_properties
@transformation.explicit_cf_compatible
class ExplicitVectorization(ppl.Pass):
    templates = properties.DictProperty(
        key_type=str,
        value_type=str,
    )
    vector_width = properties.Property(dtype=int, default=4)
    vector_input_storage = properties.Property(dtype=dace.dtypes.StorageType, default=dace.dtypes.StorageType.Register)
    vector_output_storage = properties.Property(dtype=dace.dtypes.StorageType, default=dace.dtypes.StorageType.Register)
    global_code = properties.Property(dtype=str, default="")
    global_code_location = properties.Property(dtype=str, default="")
    vector_op_numeric_type = properties.Property(dtype=typeclass, default=dace.float64)

    def __init__(self, templates, vector_width, vector_input_storage, vector_output_storage, vector_op_numeric_type,
                 global_code, global_code_location):
        super().__init__()
        self.templates = templates
        self.vector_width = vector_width
        self.vector_input_storage = vector_input_storage
        self.vector_output_storage = vector_output_storage
        self.global_code = global_code
        self.global_code_location = global_code_location
        self._tasklet_vectorizable_map = dict()
        self.vector_op_numeric_type = vector_op_numeric_type
        self._used_names = set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {
            PowerOperatorExpansion, SplitTasklets, RemoveFPTypeCasts, RemoveIntTypeCasts,
            CleanDataToScalarSliceToTaskletPattern
        }

    def _get_all_parent_scopes(self, state: SDFGState, node: dace.nodes.NestedSDFG):
        parents = list()
        parent = state.scope_dict()[node]
        while parent is not None:
            parents.append(parent)
            parent = state.scope_dict()[parent]
        return parents

    def _vectorize_map(self, state: SDFGState, inner_map_entry: dace.nodes.MapEntry, vectorization_number: int):
        # Get the innermost maps
        assert isinstance(inner_map_entry, dace.nodes.MapEntry)

        tile_sizes = [1 for _ in inner_map_entry.map.range]
        tile_sizes[-1] = self.vector_width
        MapTiling.apply_to(
            sdfg=state.parent_graph.sdfg,
            map_entry=inner_map_entry,
            options={
                "tile_sizes": tile_sizes,
                "skew": False,
                "divides_evenly": False,
            },
        )
        new_inner_map = inner_map_entry
        new_inner_map.schedule = dace.dtypes.ScheduleType.Sequential
        old_inner_map = state.entry_node(new_inner_map)

        (b, e, s) = new_inner_map.map.range[0]
        assert len(new_inner_map.map.range) == 1
        lane_base = f"({new_inner_map.map.params[0]} - {b})"
        vector_map_param = new_inner_map.map.params[0]
        try:
            int_size = int(e + 1 - b)
        except:
            int_size = None
        assert (int_size is not None and int_size == self.vector_width) or (
            e - b + 1
        ).approx == self.vector_width, f"MapTiling should have created a map with range of size {self.vector_width}, found {(e - b + 1)}"
        assert s == 1, f"MapTiling should have created a map with stride 1, found {s}"
        # Vector the range by for example making [0:4:1] to [0:4:4]
        #assert e == b + self.vector_width - 1 or e.approx == b + self.vector_width - 1, f"(b,e,s): ({b}, {e}, {s}), b + vector = {b + self.vector_width - 1}"
        new_inner_map.map.range = dace.subsets.Range([(b, e, self.vector_width)])

        # Need to check that all tasklets within the map are vectorizable
        nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
        assert all(
            {self._is_vectorizable(state, node)
             for node in nodes if isinstance(node, dace.nodes.Tasklet)}
        ), f"All tasklets within maps need to be vectorizable. This means all inputs / outputs of the maps need to be arrays"

        has_single_nested_sdfg = len(nodes) == 1 and isinstance(next(iter(nodes)), dace.nodes.NestedSDFG)

        # Updates memlets from [k, i] to [k, i:i+4]
        self._extend_memlets(state, new_inner_map)
        self._extend_temporary_scalars(state, new_inner_map)
        if not has_single_nested_sdfg:
            self._copy_in_and_copy_out(state, new_inner_map, vectorization_number)
        # Replactes tasklets of form A op B to vectorized_op(A, B)
        self._replace_tasklets(state, new_inner_map, vector_map_param)
        # Copies in data to the storage needed by the vector unit
        # Copies out data from the storage needed by the vector unit
        # Copy-in out needs to skip scalars and arrays that pass complete dimensions (over approximation must be due to a reason)

        # If tasklet -> sclar -> tasklet, now we have,
        # vector_tasklet -> scalar -> vector_tasklet
        # makes the scalar into vector

        # If the inner node is a NestedSDFG we need to vectorize that too
        if has_single_nested_sdfg:
            nsdfg_node = next(iter(nodes))
            state.sdfg.save("x0.sdfg")
            fix_nsdfg_connector_array_shapes_mismatch(state, nsdfg_node)
            check_nsdfg_connector_array_shapes_match(state, nsdfg_node)
            self._vectorize_nested_sdfg(state, nsdfg_node, vector_map_param)
            add_copies_before_and_after_nsdfg(state, nsdfg_node, (self.vector_width, ), self.vector_input_storage)
            state.sdfg.save("x6.sdfg")

        vector_tasklets = {
            n
            for n in state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
            if isinstance(n, dace.nodes.Tasklet)
        }
        # TODO: fix it in add array
        #for n in vector_tasklets:
        #    for e in state.in_edges(n) + state.out_edges(n):
        #        if e.data.data is not None:
        #            if state.sdfg.arrays[e.data.data].dtype != self.vector_op_numeric_type:
        #                state.sdfg.arrays[e.data.data].dtype = self.vector_op_numeric_type

    def _vectorize_nested_sdfg(self, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, vector_map_param: str):
        inner_sdfg: dace.SDFG = nsdfg.sdfg
        # Imagine the case where
        # On vectorization of interstate edges (Step 3):
        # We do A[idx[i]]
        # DaCe generates this as | State 1 | -(sym = idx[i])-> | State 2 code = A[sym] ... |
        # This means when we vectorize access to 8 elements then we need to access:
        # idx[i:i+8]
        # And access A accordingly: A[idx[i], idx[i+1], ..., idx[i+8]]
        # Since we can't improve this we need to load the individual sym1, sym2, ..., sym8 individually

        # First we track which inputs have the shape of the vector unit
        # We can copy all of them in the state e.g. idx[i:i+8]
        # Scalars need to be skipped (input subset is still a scalar)
        # Views that have different dimensions have indirect accesses
        # We need to load them individually, for that we need to add "A_packed"
        # and anytime we detect a load from A we need load the 8 elements accordingly

        # Any time we find a load from A[sym] we need to expand load to sym1, sym2, sym3, sym4 (need to find the sym assignment before)
        # and extend that state

        # Step 1. Analyze
        # 1.1. Detect input and output shapes
        # 1.1.1 Make all non-transient data within the nested SDFG match the connector shapes.
        # 1.1.2 All transient arrays should match the vector unit shape
        # 1.2 Detect sink and source scalars (non-transient scalar access nodes without out_edges or without in_edges)
        # 1.3 Scalar sources are not supported (because duplicating input scalar data results in a different program), raise Error
        # 1.3.1 Unless has flops in that case we can move it, if it involved in the last floating point operation
        # 1.4 Scalar sinks are supported as they can be de-duplicated when writing, track them
        # 1.5 Detect indirect accesses that need a packed intermediate storage
        # 1.6 Each access parameter needs to becomes is own array make its own array

        # After replacing all arrays to match, vectorize:
        # Step 2. Duplicate all interstate symbols to respect lane-ids
        # 2.1 Generate packed loads
        # Step 3. For all scalar sink nodes de-duplicate writes
        # Step 4. Replace all memlet subsets and names
        # Step 5. Replace all tasklets to use vectorized types
        # Step 6. Collect all data used, make sure their type matches the vector op type

        # 1.1.1
        fix_nsdfg_connector_array_shapes_mismatch(state, nsdfg)
        # 1.1.2
        transient_arrays = {arr_name for arr_name, arr in inner_sdfg.arrays.items() if arr.transient}
        print("Transient arrays:", transient_arrays)
        vector_width_transient_arrays = {
            arr_name
            for arr_name in transient_arrays if inner_sdfg.arrays[arr_name].shape == (self.vector_width, )
        }
        non_vector_width_transient_arrays = transient_arrays - vector_width_transient_arrays
        print("Non vector width transient arrays:", non_vector_width_transient_arrays)
        replace_arrays_with_new_shape(inner_sdfg, vector_width_transient_arrays, (self.vector_width, ),
                                      self.vector_op_numeric_type)
        replace_arrays_with_new_shape(inner_sdfg, non_vector_width_transient_arrays, (self.vector_width, ), None)
        inner_sdfg.reset_cfg_list()

        vector_width_arrays = {
            arr_name
            for arr_name, arr in inner_sdfg.arrays.items()
            if isinstance(arr, dace.data.Array) and arr.shape == (self.vector_width, )
        }
        scalars = {
            arr_name
            for arr_name, arr in inner_sdfg.arrays.items()
            if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, ))
        }
        print(f"Inner SDFG now has following vector-width arrays: {vector_width_arrays}")
        print(f"Inner SDFG now has following scalars: {scalars}")
        state.sdfg.validate()
        state.sdfg.save("y1.sdfg")

        # 1.2
        scalar_source_nodes: List[Tuple[dace.SDFGState,
                                        dace.nodes.AccessNode]] = get_scalar_source_nodes(inner_sdfg, True)
        array_source_nodes: List[Tuple[dace.SDFGState,
                                        dace.nodes.AccessNode]] = get_array_source_nodes(inner_sdfg, True)
        scalar_sink_nodes: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]] = get_scalar_sink_nodes(inner_sdfg, True)
        array_sink_nodes: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]] = get_array_sink_nodes(inner_sdfg, True)
        print("CXC", scalar_source_nodes, scalar_sink_nodes)

        # 1.3 and 1.3.1
        if len(scalar_source_nodes) > 1 and len(scalar_sink_nodes) > 1:
            raise Exception("Pass can't handle more than one scalar sink and source nodes in the NestedSDFG"
                            f"Scalar source nodes: {scalar_source_nodes}"
                            f"Scalar sink nodes: {scalar_sink_nodes}")
        elif len(scalar_source_nodes) == 1 and len(scalar_sink_nodes) == 1:
            move_out_reduction(scalar_source_nodes, state, nsdfg, inner_sdfg, self.vector_width)
            # This changes scalar sink and source nodes
            inner_sdfg.reset_cfg_list()
            scalar_source_nodes: List[Tuple[dace.SDFGState,
                                            dace.nodes.AccessNode]] = get_scalar_source_nodes(inner_sdfg, True)
            scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                          dace.nodes.AccessNode]] = get_scalar_sink_nodes(inner_sdfg, True)

        state.sdfg.save("y2.sdfg")

        if len(scalar_source_nodes) > 0 and len(scalar_sink_nodes) > 0:
            raise Exception(
                f"Pass tried to lift a reduction within the nested SDFG to enable auto-vectorization but failed. remainign sink nodes: {scalar_sink_nodes}, remaining scalar source nodes: {scalar_source_nodes}"
            )
        # No scalar sink nodes should be left

        # 1.5
        # Generate subset to packed array name map
        # This analysis needs to be more detailed
        # Consider x = A[0, 0, _for_it_52]
        # This can be vectorized but the input shape will not be the (1,) or (vector_width,)
        # use the utility function that returns the accesses that are vectorizable:
        # vectorizable access means that all subets to an array depends purely on constants or loop parameters
        vectorizable_arrays_dict = collect_vectorizable_arrays(inner_sdfg, nsdfg, state)
        vectorizable_arrays = {k for k, v in vectorizable_arrays_dict.items() if v is True}
        non_vectorizable_arrays = {k for k, v in vectorizable_arrays_dict.items() if v is False}

        print("VECTORIZABLE ARRAYS", vectorizable_arrays)
        print("NON-VECTORIZABLE ARRAYS", non_vectorizable_arrays)

        #non_scalar_non_vector_width_arrays = {
        #    (arr_name + "_packed", (self.vector_width, ), self.vector_input_storage, arr.dtype)
        #    for arr_name, arr in inner_sdfg.arrays.items()
        #    if isinstance(arr, dace.data.Array) and (arr.shape != (1, ) and arr.shape != (self.vector_width, ))
        #}
        #array_accessed_to_be_packed = {
        #    arr_name[:-len("_packed")]
        #    for arr_name, _, _, _ in non_scalar_non_vector_width_arrays
        #}
        #add_transient_arrays_from_list(inner_sdfg, non_scalar_non_vector_width_arrays)
        add_transient_arrays_from_list(inner_sdfg, non_vectorizable_arrays)

        modified_nodes: Set[dace.nodes.Node] = set()
        modified_edges: Set[Edge[Memlet]] = set()

        # 2 and 2.1
        inner_sdfg.save("x1.sdfg")
        #new_mn, new_me = self._generate_loads_to_packed_storage(inner_sdfg, array_accessed_to_be_packed,
        #                                                        vector_width_arrays)
        new_mn, new_me = self._generate_loads_to_packed_storage(inner_sdfg, non_vectorizable_arrays,
                                                                vector_width_arrays)
        modified_nodes = modified_nodes.union(new_mn)
        modified_edges = modified_edges.union(new_me)

        inner_sdfg.save("x2.sdfg")

        # 3
        print("Scalar sink nodes:", scalar_sink_nodes)
        check_writes_to_scalar_sinks_happen_through_assign_tasklets(inner_sdfg, scalar_sink_nodes)
        new_mn, new_me = self._duplicate_unstructured_writes(inner_sdfg, non_vectorizable_arrays)
        modified_nodes = modified_nodes.union(new_mn)
        modified_edges = modified_edges.union(new_me)

        inner_sdfg.save("x3.sdfg")

        # 4
        for inner_state in inner_sdfg.all_states():
            # Skip the data data that are still scalar and source nodes
            scalar_source_data = {n.data for s, n in scalar_source_nodes}
            edges_to_replace = {
                e
                for e in inner_state.edges()
                if e not in modified_edges and e.data is not None and e.data.data not in scalar_source_data
            }
            old_subset = dace.subsets.Range([(0, 0, 1)])
            new_subset = dace.subsets.Range([(0, self.vector_width - 1, 1)])
            replace_memlet_expression(inner_state, edges_to_replace, old_subset, new_subset, True, modified_edges,
                                      self.vector_op_numeric_type)

        # Do it for arrays
        # TODO: ALSO NEED TO EXPAND MEMLET EXPRESSIONS of arrays
        for inner_state in inner_sdfg.all_states():
            # Skip the data data that are still scalar and source nodes
            source_data = {n.data for s, n in array_source_nodes}
            edges_to_replace = {
                e
                for e in inner_state.edges()
                if e not in modified_edges and e.data is not None and e.data.data in source_data
            }
            old_subset = dace.subsets.Range([(0, 0, 1)])
            new_subset = dace.subsets.Range([(0, self.vector_width - 1, 1)])
            #replace_memlet_expression(inner_state, edges_to_replace, old_subset, new_subset, True, modified_edges,
            #                          self.vector_op_numeric_type)
            expand_memlet_expression(inner_state, edges_to_replace, modified_edges, self.vector_width)

        state.sdfg.save("x4.sdfg")

        # Extend interstate edges for all symbols used in tasklets / or interstate edges that access vectorized data
        for edge in inner_sdfg.all_interstate_edges():
            candidate_arrays = vector_width_arrays
            free_syms = set()
            for k, v in edge.data.assignments.items():
                free_syms.add(k)
                free_syms = free_syms.union({str(vv) for vv in dace.symbolic.SymExpr(v).free_symbols})
            print("BBB", free_syms, {fs in candidate_arrays for fs in free_syms})
            if any({fs in candidate_arrays for fs in free_syms}):
                self._expand_interstate_assignment(inner_sdfg, edge, free_syms, candidate_arrays)


        # 5
        for inner_state in inner_sdfg.all_states():
            nodes = {n for n in inner_state.nodes() if n not in modified_nodes}
            self._replace_tasklets_from_node_list(inner_state, nodes, vector_map_param)
            modified_nodes = modified_nodes.union(nodes)

        # TODO: fix, need to do it in add-array
        #for state in inner_sdfg.all_states():
        #    vector_tasklets = {n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)}
        #    for n in vector_tasklets:
        #        for e in state.out_edges(n):
        #            if e.data.data is not None:
        #                if state.sdfg.arrays[e.data.data].dtype != self.vector_op_numeric_type and state.sdfg.arrays[e.data.data].transient is True:
        #                    state.sdfg.arrays[e.data.data].dtype = self.vector_op_numeric_type
        #    #for e in {_e for _e in state.edges() if _e in modified_edges}:
        #    #    if e.data.data is not None:
        #    #        if state.sdfg.arrays[e.data.data].dtype != self.vector_op_numeric_type and state.sdfg.arrays[e.data.data].transient is True:
        #    #            state.sdfg.arrays[e.data.data].dtype = self.vector_op_numeric_type
        state.sdfg.save("x4_5.sdfg")

        # Add missing symbols
        print("OOOOOOO", inner_sdfg.free_symbols, nsdfg.symbol_mapping)

        # There might be missing expanded loop symbols, they are of form `loop_var{id}` where `{id}` is an integer
        # Construct back the loop variable and add assignments for them
        missing_symbols = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))

        map_symbols = assert_symbols_in_parent_map_symbols(missing_symbols, state, nsdfg)

        assignment_dict = dict()

        for missing_symbol in map_symbols:
            for i in range(self.vector_width):
                assignment_dict[f"{missing_symbol}{i}"] = f"{missing_symbol} + {i}"

        inner_sdfg.add_state_before(inner_sdfg.start_block,
                                    label="missing_sym_generate",
                                    is_start_block=True,
                                    assignments=assignment_dict)

        state.sdfg.save("x5.sdfg")

    def _duplicate_unstructured_writes(self, inner_sdfg: dace.SDFG, non_vectorizable_arrays: Set[str]):
        modified_edges = set()
        modified_nodes = set()
        for state in inner_sdfg.all_states():
            for node in state.nodes():
                if state.out_degree(node) == 0:
                    arr = state.sdfg.arrays[node.data]
                    if (arr.transient is False and
                        (isinstance(arr, dace.data.Scalar) or isinstance(arr, dace.data.Array) and arr.shape == (1, ))):
                        # If it is a reduction tasklet + number of edges matching vector unit it is ok
                        srcs = {ie.src for ie in state.in_edges(node)}
                        if not (len(srcs) == 1 and state.in_degree(next(iter(srcs))) == self.vector_width
                                and isinstance(next(iter(srcs)), dace.nodes.Tasklet)):
                            raise Exception(
                                "At this point of the pass, no write to non-transient scalar sinks should remain")
                    if arr.transient is False and (isinstance(arr, dace.data.Array) and
                                                   (arr.shape != (1, ) and arr.shape != (self.vector_width, )) and 
                                                   node.data in non_vectorizable_arrays):
                        touched_nodes, touched_edges = duplicate_access(state, node, self.vector_width)
                        modified_edges = modified_edges.union(touched_edges)
                        modified_nodes = modified_nodes.union(touched_nodes)
        return modified_nodes, modified_edges

    def _generate_loads_to_packed_storage(self, sdfg: dace.SDFG, array_accessed_to_be_packed: Set[str],
                                          candidate_arrays: Set[str]) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
        modified_nodes: Set[dace.nodes.Node] = set()
        modified_edges: Set[Edge[Memlet]] = set()
        expanded_symbols = set()

        # First expand intersate assignments
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in array_accessed_to_be_packed:
                    free_symbols = edge.data.free_symbols
                    # Look for the assignments in the interstate edges and expand them
                    non_expanded_free_symbols = free_symbols - expanded_symbols
                    expanded_symbols = expanded_symbols.union(free_symbols)
                    self._expand_interstate_assignments(sdfg, non_expanded_free_symbols, candidate_arrays)

        # Then do the other stuff
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in array_accessed_to_be_packed:
                    # Create a packed copy
                    # If it is a source node (no in edges, copy in), otherwise replace with the packed data
                    if state.in_degree(edge.src) == 0 and edge.src.data == edge.data.data:
                        src_node = edge.src
                        src_node.data = f"{edge.data.data}_packed"
                        old_data_name = f"{edge.data.data}"
                        non_packed_access = state.add_access(old_data_name)
                        modified_nodes.add(non_packed_access)
                        modified_nodes.add(src_node)

                        # Replace all symbols used e.g. i,j with `i0`, `j0`
                        # A -[j]> B becomes now
                        # A -[j0,j1,...,j7]-> A_packed -[0:8]-> B
                        for i in range(self.vector_width):
                            new_subset = repl_subset_to_symbol_offset(sdfg=state.sdfg,
                                                                      subset=edge.data.subset,
                                                                      symbol_offset=str(i),
                                                                      add_missing_symbols=False)
                            at = state.add_tasklet(
                                name=f"assign_{i}",
                                inputs={
                                    "_in",
                                },
                                outputs={
                                    "_out",
                                },
                                code="_out = _in",
                            )
                            at.add_in_connector("_in")
                            at.add_out_connector("_out")
                            e1 = state.add_edge(non_packed_access, None, at, "_in",
                                                dace.memlet.Memlet(
                                                    data=old_data_name,
                                                    subset=new_subset,
                                                ))
                            e2 = state.add_edge(at, "_out", src_node, None, dace.memlet.Memlet(f"{src_node.data}[{i}]"))
                            modified_nodes.add(at)
                            if isinstance(e1, dace.nodes.Node):
                                assert False
                            if isinstance(e2, dace.nodes.Node):
                                assert False
                            modified_edges.add(e1)
                            modified_edges.add(e2)

                        # Now update the subset
                        edge.data = dace.memlet.Memlet(expr=f"{src_node.data}[0:{self.vector_width}]")
                        if isinstance(edge, dace.nodes.Node):
                            assert False
                        modified_edges.add(edge)
        return modified_nodes, modified_edges

    def _expand_interstate_assignment(self, sdfg: dace.SDFG, edge: Edge[InterstateEdge], syms: Set[str],
                                      candidate_arrays: Set[str]):
        duplicated_symbols = set()
        syms_to_rm = set()
        new_assignments = dict()
        # Lets say we have
        # k = idx
        # then we need to do:
        # k0 = idx[0]
        # k1 = idx[1]
        # ...
        # k7 = idx[7]
        # Also need to consider the case where multiple symbols are involved
        # k0 = idx[0] + idy[0]
        for k, v in edge.data.assignments.items():
            if k in syms:
                
                for i in range(0, self.vector_width):
                    # Get all scalar accesses from v and replace with the array equivalent
                    # if we have j = k1 + k2
                    # we need to have j0 = k1[0] + k2[0], j1 = k1[1] + k2[1], ...
                    vcopy = copy.deepcopy(v)
                    nv = vcopy
                    #print("Candidate Arrays:", candidate_arrays, f"Edge assignment: {nv}")
                    for ca in candidate_arrays:
                        assert ca in sdfg.arrays
                        ca_data = sdfg.arrays[ca]
                        #print("Candidate Array Name:", ca, "Candidate Array:", ca_data)
                        if isinstance(ca_data, dace.data.Scalar) or (isinstance(ca_data, dace.data.Array)
                                                                     and ca_data.shape == (1, )):
                            ca_scl = ca_data
                            assert ca_scl.transient
                            sdfg.remove_data(ca, validate=False)
                            sdfg.add_array(
                                name=ca,
                                shape=(self.vector_width, ),
                                dtype=ca_scl.dtype,
                                storage=ca_scl.storage,
                                location=ca_scl.location,
                                transient=True,
                                lifetime=ca_scl.lifetime,
                                find_new_name=False,
                            )
                        #nv = nv.replace(ca, f"{ca}[{i}]")
                        nv_before = nv
                        nv = token_replace(nv, ca, f"{ca}[{i}]")
                        #print(f"Before: {nv_before}, After replacing {ca} with {ca}[{i}]: {nv}")
                    new_assignments[f"{k}{i}"] = nv
                    if i == 0:
                        new_assignments[k] = nv
                duplicated_symbols.add(k)
                syms_to_rm.add(k)
            else:
                new_assignments[k] = v
        edge.data.assignments = new_assignments
        #for sym in syms_to_rm:
        #    print("TRY REMOVE", sym)
        #    if sym in sdfg.symbols:
        #        print(f"SYM NOT IN {sdfg.symbols}")
        #        sdfg.remove_symbol(str(sym))
        #        assert (str(sym) not in sdfg.symbols)
        return duplicated_symbols

    def _expand_interstate_assignments(self, sdfg: dace.SDFG, syms: Set[str], candidate_arrays: Set[str]):
        duplicated_symbols = set()
        for edge in sdfg.all_interstate_edges():
            duplicated_symbols = duplicated_symbols.union(
                self._expand_interstate_assignment(sdfg, edge, syms, candidate_arrays))
        return duplicated_symbols

    def _extend_temporary_scalars(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges_to_rm = set()
        edges_to_add = set()
        nodes_to_rm = set()
        for node in nodes:
            if isinstance(node, dace.nodes.AccessNode):
                desc = state.parent_graph.sdfg.arrays[node.data]
                print(desc, desc.shape, type(desc), type(desc.shape))
                if (isinstance(desc, dace.data.Scalar) or (isinstance(desc, dace.data.Array) and desc.shape == (1, ))):
                    if f"{node.data}_vec" not in state.parent_graph.sdfg.arrays:
                        state.sdfg.add_array(
                            name=f"{node.data}_vec",
                            shape=(self.vector_width, ),
                            dtype=self.vector_op_numeric_type,
                            storage=self.vector_input_storage,
                            transient=True,
                            alignment=self.vector_width * desc.dtype.bytes,
                            find_new_name=False,
                        )

                new_an = state.add_access(f"{node.data}_vec")

                for ie in state.in_edges(node):
                    new_edge_tuple = (ie.src, ie.src_conn, new_an, None, copy.deepcopy(ie.data))
                    edges_to_rm.add(ie)
                    edges_to_add.add(new_edge_tuple)
                for oe in state.out_edges(node):
                    new_edge_tuple = (new_an, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
                    edges_to_rm.add(oe)
                    edges_to_add.add(new_edge_tuple)
                nodes_to_rm.add(node)

        rmed_data_names = set()
        for edge in edges_to_rm:
            state.remove_edge(edge)
        for node in nodes_to_rm:
            state.remove_node(node)
            rmed_data_names.add(node.data)
        for edge_tuple in edges_to_add:
            state.add_edge(*edge_tuple)

        # Refresh nodes
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        for edge in state.all_edges(*nodes):
            if edge.data is not None and edge.data.data in rmed_data_names:
                new_memlet = dace.memlet.Memlet(
                    data=f"{edge.data.data}_vec",
                    subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
                )
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _extend_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        edges = state.all_edges(*nodes)
        self._extend_memlets_from_node_and_edge_list(state, edges)

    def _extend_memlets_from_node_and_edge_list(self, state: SDFGState, edges: Iterable[Edge[Memlet]]):
        for edge in edges:
            memlet: dace.memlet.Memlet = edge.data
            # If memlet is equal to the array shape, we have a clear case of over-approximation
            # For the check we know: last dimension results in contiguous access,
            # We should memlets of form:
            # [(b, e, s), ...] the b, e, s may depend on i.
            # If b is i, e is i (inclusive range), then extension needs to result with (i, i + vector_width - 1, 1)
            # We can assume (and check) that s == 1
            # if b is 2*i and e is 2*i then we should extend the i in the end with 2*(i + vector_width - 1)
            map_entry: dace.nodes.MapEntry = state.entry_node(
                edge.src) if not isinstance(edge.src, dace.nodes.MapEntry) else state.entry_node(edge.dst)

            used_param = map_entry.map.params[
                -1]  # Regardless of offsets F / C we assume last parameter of the map is used

            new_range_list = [(b, e, s) for (b, e, s) in memlet.subset]
            stride_offset = 0 if self._stride_type == "F" else -1  # left-contig is Fortran and right-contig is C
            range_tup: Tuple[dace.symbolic.SymExpr, dace.symbolic.SymExpr,
                             dace.symbolic.SymExpr] = new_range_list[stride_offset]
            lb, le, ls = range_tup
            if isinstance(le, int):
                le = dace.symbolic.SymExpr(str(le))
            assert ls == 1, f"Previous checks must have ensured the final dimension should result in unit-stride access"
            new_range_list[stride_offset] = (
                lb, le.subs(used_param, dace.symbolic.SymExpr(f"({self.vector_width} - 1) + {used_param}")),
                ls) if lb == le else (lb,
                                      le.subs(used_param,
                                              dace.symbolic.SymExpr(f"({self.vector_width} * {used_param}) - 1")), ls)

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            new_memlet = dace.memlet.Memlet(
                data=memlet.data,
                subset=dace.subsets.Range(new_range_list),
            )
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _is_vectorizable(self, state: SDFGState, node: dace.nodes.Tasklet):
        # Arr1 -> Tasklet1 -> sc1 -> Tasklet2 -> sc3 -> Tasklet3 -> Arr
        # Arr2
        # Should return [Arr,Arr], [Arr]
        # It is important to determine if a tasklet is "vectorizable"
        # Only a tasklet that reads from arrays and writes to arrays are vectorizable
        # But if we only check Tasklet1, Tasklet2 or Tasklet3's ddirect neighbors we would consider
        # they can't, but they can

        # First check cache
        if node in self._tasklet_vectorizable_map:
            return self._tasklet_vectorizable_map[node]

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)
        input_types = set()
        output_types = set()
        while in_edges:
            in_edge = in_edges.pop()
            if state.in_degree(in_edge.src) == 0:
                if isinstance(in_edge.src, dace.nodes.AccessNode):
                    input_types.add(type(state.sdfg.arrays[in_edge.src.data]))
                else:
                    if in_edge.data is not None:
                        print(in_edge.data, in_edge.data is None)
                        raise Exception(
                            f"Unsupported Type for in_edge.src got type {type(in_edge.src)}, need AccessNode ({in_edge.data})"
                        )
            if not isinstance(in_edge.src, dace.nodes.MapEntry):
                in_edges += state.in_edges(in_edge.src)
            else:
                input_types.add(type(state.sdfg.arrays[in_edge.data.data]))
        while out_edges:
            out_edge = out_edges.pop()
            if state.out_degree(out_edge.dst) == 0:
                if isinstance(out_edge.dst, dace.nodes.AccessNode):
                    output_types.add(type(state.sdfg.arrays[out_edge.dst.data]))
                else:
                    if out_edge.data is not None:
                        raise Exception(
                            f"Unsupported Type for out_edge.dst got type {type(out_edge.dst)}, need AccessNode ({in_edge.data})"
                        )
            if not isinstance(out_edge.dst, dace.nodes.MapExit):
                out_edges = state.out_edges(out_edge.dst)
            else:
                output_types.add(type(state.sdfg.arrays[out_edge.data.data]))

        vectorizable = (all({isinstance(itype, dace.data.Array) or itype == dace.data.Array
                             for itype in input_types}) and
                        all({isinstance(otype, dace.data.Array) or otype == dace.data.Array
                             for otype in output_types}))
        self._tasklet_vectorizable_map[node] = vectorizable
        return vectorizable

    def _replace_tasklets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, vector_map_param: str):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        self._replace_tasklets_from_node_list(state, nodes, vector_map_param)

    def _replace_tasklets_from_node_list(self, state: SDFGState, nodes: Iterable[dace.nodes.Node],
                                         vector_map_param: str):
        for node in nodes:
            if isinstance(node, dace.nodes.Tasklet):
                tasklet_info = tutil.classify_tasklet(state, node)
                print("Tasklet:", node, " has info:", tasklet_info)
                instantiate_tasklet_from_info(state, node, tasklet_info, self.vector_width, self.templates,
                                              vector_map_param)

    def _offset_all_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, dataname: str, new_dataname: str):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        edges = state.all_edges(*nodes)
        self._offset_memlets_from_edge_list(state, edges, dataname, new_dataname)

    def _offset_memlets_from_edge_list(self, state: SDFGState, edges: Iterable[Edge[Memlet]], dataname: str,
                                       new_dataname: str):
        for edge in edges:
            if edge.data.data is None or edge.data.data != dataname:
                continue
            memlet: dace.memlet.Memlet = edge.data

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            new_memlet = dace.memlet.Memlet(
                data=new_dataname,
                subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
            )
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _iterate_on_path_from_map_entry_to_exit(self, state: SDFGState, map_exit: dace.nodes.MapExit,
                                                data_in_edge: Edge[Memlet], dataname: str, new_dataname: str):
        # IMPORTANT!
        # Get memlet paths until we reach map exit
        # This will create a problem if the input flows into the exit because we can't distinguish,
        # We will assume the first occurence flow to the exit
        edges_to_check = state.memlet_path(data_in_edge)

        while edges_to_check:
            for edge in edges_to_check:
                if edge.dst == map_exit:
                    edges_to_check = None
                    break

                if edge.data.data is None or edge.data.data != dataname:
                    continue

                memlet: dace.memlet.Memlet = edge.data

                assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

                new_memlet = dace.memlet.Memlet(
                    data=new_dataname,
                    subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
                )
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

            if edges_to_check is not None:
                sink_node = edges_to_check[-1].dst
                # Tasklet (SSA-ed) or access nodes can appear, this means one edge
                new_out_edges = state.out_edges(sink_node)
                assert len(new_out_edges
                           ) == 1, f"Sink node {sink_node} has out edges {new_out_edges}, excepted to have 1 out-edge"
                new_out_edge = new_out_edges[0]
                edges_to_check = state.memlet_path(new_out_edge)

            # the sink node is a code node and out data has the same array then we have a problem (not that we can fix but it needs to be preprocessed)
            if isinstance(sink_node, (CodeNode, dace.nodes.Tasklet)):
                for ie in state.out_edges(sink_node):
                    if ie.data.data is not None and ie.data.data == dataname:
                        print(
                            f"After sink node, the data {dataname} is still used, dangerous. Implementation assumes, the first one flows out we have two inputs that access the same"
                        )

    def _iterate_on_path_from_map_exit_to_entry(self, state: SDFGState, map_entry: dace.nodes.MapEntry,
                                                data_out_edge: Edge[Memlet], dataname: str, new_dataname: str):
        edges_to_check = state.memlet_path(data_out_edge)

        while edges_to_check:
            for edge in edges_to_check:
                if edge.src == map_entry:
                    edges_to_check = None
                    break

                if edge.data.data is None or edge.data.data != dataname:
                    continue

                memlet: dace.memlet.Memlet = edge.data

                assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

                new_memlet = dace.memlet.Memlet(
                    data=new_dataname,
                    subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
                )
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

            if edges_to_check is not None:
                source_node = edges_to_check[0].src
                # Tasklet (SSA-ed) or access nodes can appear, this means one edge
                new_in_edges = state.in_edges(source_node)
                #assert len(new_in_edges) == 1, f"Source node {source_node} has out edges {new_in_edges}, excepted to have 1 out-edge"
                # if length is not 1 we cant now what to do anymore
                if len(new_in_edges) == 1:
                    new_in_edge = new_in_edges[0]
                    edges_to_check = state.memlet_path(new_in_edge)
                else:
                    new_in_edge = None
                    edges_to_check = None

            # the sink node is a code node and out data has the same array then we have a problem (not that we can fix but it needs to be preprocessed)
            if isinstance(source_node, (CodeNode, dace.nodes.Tasklet)):
                for ie in state.in_edges(source_node):
                    if ie.data.data is not None and ie.data.data == dataname:
                        print(
                            f"After source node, the data {dataname} is still used, dangerous. Implementation assumes, the first one flows out we have two inputs that access the same"
                        )

    def _offset_memlets_on_path(self, state: SDFGState, map_entry: dace.nodes.MapEntry, dataname: str,
                                new_dataname: str):
        # Get memlet paths
        # And while memlet we have not encountered the map exit continue
        # if we find data name then we will replace
        print(f"Offset memlets on path {dataname} -> {new_dataname}")
        # Precondition: memlet-path and no tree (previous passes to explicit vectorization should have fixed that)
        # one in edge to the map entry with the vector data
        map_exit = state.exit_node(map_entry)

        # Get all in edges (need to do the same for the map exit later)
        # Go from map_entry -> map_exit for input data
        # map_exit -> map_entry for output data
        in_edges = state.in_edges(map_entry)
        # Filter by the data
        data_in_edges = {e for e in in_edges if e.data.data == new_dataname}
        assert len(data_in_edges) <= 1
        if len(data_in_edges) == 1:
            data_in_edge: Edge[Memlet] = next(iter(data_in_edges))
            self._iterate_on_path_from_map_entry_to_exit(state, map_exit, data_in_edge, dataname, new_dataname)

        # Now for map exit
        out_edges = state.out_edges(map_exit)
        # Filter by the data
        print(out_edges)
        data_out_edges = {e for e in out_edges if e.data.data == new_dataname}
        print(out_edges)
        assert len(data_out_edges) <= 1
        if len(data_out_edges) == 1:
            print("CCC", data_out_edges)
            data_out_edge: Edge[Memlet] = next(iter(data_out_edges))

            # IMPORTANT!
            # Get memlet paths until we reach map exit
            # This will create a problem if the input flows into the exit because we can't distinguish,
            # We will assume the first occurence flow to the exit
            self._iterate_on_path_from_map_exit_to_entry(state, map_entry, data_out_edge, dataname, new_dataname)

        assert len(data_in_edges) == 1 or len(
            data_out_edges
        ) == 1, f"{dataname} -> {new_dataname} no data in our out edges found | {in_edges}, {out_edges}"

    def _find_new_name(self, candidate: str):
        candidate2 = candidate
        i = 0
        while candidate2 in self._used_names:
            candidate2 = candidate + f"_{i}"
            i += 1
        self._used_names.add(candidate)
        return candidate2

    def _copy_in_and_copy_out(self, state: SDFGState, map_entry: dace.nodes.MapEntry, vectorization_number: int):
        map_exit = state.exit_node(map_entry)
        data_and_offsets = list()
        in_datas = set()
        for ie in state.in_edges(map_entry):
            # If input storage is not registers need to copy in
            array = state.parent_graph.sdfg.arrays[ie.data.data]
            if array.storage != self.vector_input_storage:
                # Add new array, if not there
                arr_name_to_use = self._find_new_name(f"{ie.data.data}_vec_k{vectorization_number}")
                if arr_name_to_use not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=arr_name_to_use,
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=self.vector_width * array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                in_datas.add(arr_name_to_use)
                an = state.add_access(arr_name_to_use)
                src, src_conn, dst, dst_conn, data = ie
                state.remove_edge(ie)
                state.add_edge(src, src_conn, an, None, copy.deepcopy(data))
                state.add_edge(an, None, map_entry, ie.dst_conn,
                               dace.memlet.Memlet(f"{arr_name_to_use}[0:{self.vector_width}]"))

                memlet: dace.memlet.Memlet = ie.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                #for data, _off in data_and_offsets:
                #    if data == dataname:
                #        if _off != offsets:
                #            state.sdfg.save("uwuowo.sdfg")
                #            raise ValueError(
                #                f"Cannot handle multiple input edges from the same array {dataname} to the same map {map_entry} in state {state}"
                #            )
                data_and_offsets.append((dataname, arr_name_to_use, offsets))
        print("Added in data:", in_datas)
        out_datas = set()
        for oe in state.out_edges(map_exit):
            array = state.parent_graph.sdfg.arrays[oe.data.data]
            print(array)
            if array.storage != self.vector_output_storage:
                # If the name exists in the inputs, reuse the name
                arr_name_to_use = f"{oe.data.data}_vec_k{vectorization_number}"
                if arr_name_to_use not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=arr_name_to_use,
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=self.vector_width * array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                out_datas.add(arr_name_to_use)
                an = state.add_access(arr_name_to_use)
                src, src_conn, dst, dst_conn, data = oe
                state.remove_edge(oe)
                state.add_edge(map_exit, src_conn, an, None,
                               dace.memlet.Memlet(f"{arr_name_to_use}[0:{self.vector_width}]"))
                state.add_edge(an, None, dst, dst_conn, copy.deepcopy(data))

                memlet: dace.memlet.Memlet = oe.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                #for data, _off in data_and_offsets:
                #    if data == dataname:
                #        if _off != offsets:
                #            raise NotImplementedError(
                #                f"Vectorization can't handle when data appears both in input and output sets of a map")
                data_and_offsets.append((dataname, arr_name_to_use, offsets))
        print("Out data:", out_datas)

        for dataname, new_dataname, offsets in data_and_offsets:
            self._offset_memlets_on_path(state, map_entry, dataname, new_dataname)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        stride_type = assert_strides_are_packed_C_or_packed_Fortran(sdfg)
        self._stride_type = stride_type
        assert_last_dim_of_maps_are_contigous_accesses(sdfg)
        sdfg.save("t1.sdfg")
        assert_maps_consist_of_single_nsdfg_or_no_nsdfg(sdfg)

        if self.vector_input_storage != self.vector_output_storage:
            raise NotImplementedError("Different input and output storage types not implemented yet")

        # 1. Broadcast used scalars to vectorized type
        # 1.1 E.g. if we do 2.0 * A[i:i+4] then we need to have [2.0, 2.0, 2.0, 2.0] * A[i:i+4]

        # 2. Vectorize Maps and Tasklets
        # 2.1 Map needs to be tiled using the vector unit length
        # 2.1.1 Assumption - the inner dimension of the map should be generating unit-stride accesses
        # 2.1.2 If not, we need to re-order the maps so that the innermost map is the one with unit-stride accesses
        # A op B -> needs to be replaced with  vectorized_op(A, B)
        # 2.2 All memlets need to be updated to reflect the vectorized access
        # 2.3 All tasklets ened to be replaced with the vectorized code (using templates)

        # 3. Insert data transfers
        # for (o = 0; o < 4; o ++){
        #     A[i + o] ...;
        # }
        # Needs to become:
        # vecA[0:4] = A[i:i+4]; if source of A is not input location of the vector unit
        # Same for the output
        # This needs to be done before the vectorized map from source(A) ->  source(vecA)
        # And after the map for destination(vecA) -> destination(A)

        # 4. Recursively done to nested SDFGs
        # If the inner body - if the nestedSDFG is within a map then the inner SDFG
        # needs to be vectorized as well

        # For all maps:
        # Map1 [ Map2 [ Map3 [Body]]]
        # Can vectorize the innermost map only
        # If NestedSDFG we need to know if we have a parent map

        # Need to vectorize innermost maps always, if a map has a map inside, vectorize that
        map_entries = list()
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                map_entry: dace.nodes.MapEntry = node
                map_entries.append((map_entry, graph))

        # We allow a map to have no nested SDFGs within its body, or only one node which is a nestedSDFG
        # If there is only a nested SDFG inside we need vectorize that.
        num_vectorized = 0
        vectorized_maps = set()
        sdfgs_to_vectorize = set()
        for i, (map_entry, state) in enumerate(map_entries):
            print("The map is:", map_entry, ", is innermost map:", is_innermost_map(map_entry, state))
            if is_innermost_map(map_entry, state):
                num_vectorized += 1
                all_nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
                self._vectorize_map(state, map_entry, vectorization_number=i)
                vectorized_maps.add(map_entry)
                if len(all_nodes_between) == 1 and isinstance(next(iter(all_nodes_between)), dace.nodes.NestedSDFG):
                    sdfgs_to_vectorize.add((next(iter(all_nodes_between)), state))

        # Assume we have :
        # --- Map Entry ---
        # -----------------
        #     NestedSDFG
        # tasklet without map
        # --- Map Exit  ---
        #
        # If the inside has no maps
        # We have problems. If no maps,
        # We need to add a parent map
        for node, state in sdfgs_to_vectorize:
            nested_sdfg: dace.nodes.NestedSDFG = node
            parent_scopes = self._get_all_parent_scopes(state, node)
            if len(parent_scopes) == 0:
                raise NotImplementedError(
                    "NestedSDFGs without parent map scopes are not supported, they must have been inlined if the pipeline has been called."
                    "If pipeline has been called verify why InlineSDFG failed, otherwise call InlineSDFG")

        sdfg.append_global_code(cpp_code=self.global_code, location=self.global_code_location)

        return None
