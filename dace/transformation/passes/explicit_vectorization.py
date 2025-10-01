# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import dace
import ast
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from dace import SDFG, Memlet, SDFGState, properties, transformation
from dace.sdfg.graph import Edge
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.passes import analysis as ap, pattern_matching as pmp
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import IntegerPowerToMult, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes import InlineSDFGs

from dace.transformation.passes.explicit_vectorization_utils import *


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

    def __init__(self, templates, vector_width, vector_input_storage, vector_output_storage, global_code,
                 global_code_location):
        super().__init__()
        self.templates = templates
        self.vector_width = vector_width
        self.vector_input_storage = vector_input_storage
        self.vector_output_storage = vector_output_storage
        self.global_code = global_code
        self.global_code_location = global_code_location
        self._tasklet_vectorizable_map = dict()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.States | ppl.Modifies.Tasklets | ppl.Modifies.NestedSDFGs | ppl.Modifies.Scopes | ppl.Modifies.Descriptors

    def depends_on(self):
        return {
            IntegerPowerToMult, SplitTasklets, RemoveFPTypeCasts, RemoveIntTypeCasts,
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
            options={"tile_sizes": tile_sizes, "skew": False},
        )
        new_inner_map = inner_map_entry
        new_inner_map.schedule = dace.dtypes.ScheduleType.Sequential
        old_inner_map = state.entry_node(new_inner_map)

        (b, e, s) = new_inner_map.map.range[0]
        assert len(new_inner_map.map.range) == 1
        try:
            int_size = int(e + 1 -b)
        except:
            int_size = None
        assert (int_size is not None and int_size == self.vector_width) or (e - b + 1).approx  == self.vector_width, f"MapTiling should have created a map with range of size {self.vector_width}, found {(e - b + 1)}"
        assert s == 1, f"MapTiling should have created a map with stride 1, found {s}"
        # Vector the range by for example making [0:4:1] to [0:4:4]
        assert e == b + self.vector_width - 1 or e.approx == b + self.vector_width - 1, f"(b,e,s): ({b}, {e}, {s}), b + vector = {b + self.vector_width - 1}"
        new_inner_map.map.range = dace.subsets.Range([(b, b + self.vector_width - 1, self.vector_width)])

        # Need to check that all tasklets within the map are vectorizable
        nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
        assert all(
            {self._is_vectorizable(state, node)
             for node in nodes if isinstance(node, dace.nodes.Tasklet)}
        ), f"All tasklets within maps need to be vectorizable. This means all inputs / outputs of the maps need to be arrays"

        has_single_nested_sdfg = len(nodes) == 1 and isinstance(next(iter(nodes)), dace.nodes.NestedSDFG)

        # Updates memlets from [k, i] to [k, i:i+4]
        self._extend_memlets(state, new_inner_map)
        # Replactes tasklets of form A op B to vectorized_op(A, B)
        self._replace_tasklets(state, new_inner_map)
        # Copies in data to the storage needed by the vector unit
        # Copies out data from the storage needed by the vector unit
        # Copy-in out needs to skip scalars and arrays that pass complete dimensions (over approximation must be due to a reason)
        if has_single_nested_sdfg:
            #self._copy_in_and_copy_out_with_nsdfg(state, new_inner_map, vectorization_number)
            pass
        else:
            self._copy_in_and_copy_out(state, new_inner_map, vectorization_number)
        # If tasklet -> sclar -> tasklet, now we have,
        # vector_tasklet -> scalar -> vector_tasklet
        # makes the scalar into vector
        self._extend_temporary_scalars(state, new_inner_map)

        # If the inner node is a NestedSDFG we need to vectorize that too
        if has_single_nested_sdfg:
            nsdfg_node = next(iter(nodes))
            add_copies_before_and_after_nsdfg(state, nsdfg_node, None, self.vector_input_storage)
            fix_nsdfg_connector_array_shapes_mismatch(state, nsdfg_node)
            check_nsdfg_connector_array_shapes_match(state, nsdfg_node)
            self._vectorize_nested_sdfg(state, nsdfg_node)

    def _vectorize_nested_sdfg(self, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG):
        inner_sdfg: dace.SDFG = nsdfg.sdfg
        # Imagine the case where
        # We do A[idx[i]]
        # DaCe generates this as | State 1 | -(sym = idx[i])-> | State 2 code = A[sym] ... |
        # This means when we vectorize access to 8 elements then we need to access:
        # idx[i:i+8]
        # And access A accordingly: A[idx[i], idx[i+1], ..., idx[i+8]]
        # Since we can't improve this we need to load the individual sym1, sym2, ..., sym8 individually
        # And populate A manually too.

        # First we track which inputs have the shape of the vector unit
        # We can copy all of them in the state e.g. idx[i:i+8]
        # Scalars need to be skipped (input subset is still a scalar)
        # Views that have different dimensions have indirect accesses
        # We need to load them individually, for that we need to add "A_packed"
        # and anytime we detect a load from A we need load the 8 elements accordingly

        # Any time we find a load from A[sym] we need to expand load to sym1, sym2, sym3, sym4 (need to find the sym assignment before)
        # and extend that state

        modified_edges = set()
        modified_nodes = set()
        arrays_need_to_be_packed = set()
        scalar_to_vector_width_arrays = set()
        src_dst_size = set()

        for ie in state.in_edges(nsdfg):
            src_dst_size.add((ie.data.data, ie.dst_conn, ie.data.volume))

        # Handle copy-in and creating of packed data types
        for src, dst, size in src_dst_size:
            print(src, dst, size, self.vector_width)
            if size != 1 and size != self.vector_width:
                inner_sdfg.add_array(
                    name=f"{dst}_packed",
                    shape=(self.vector_width, ),
                    dtype=inner_sdfg.arrays[dst].dtype,
                    transient=True,
                )
                arrays_need_to_be_packed.add(dst)

            elif size == self.vector_width:
                # No need to add vectorized arrays, they have been copied in front of the SDFG
                pass

        # If dst is a scalar but input is vector_width length then replace the scalar with an array
        # Keep a track what has been replaced from a scalar to an array
        scalar_to_vector_conversions = set()
        for src, dst, size in src_dst_size:
            dst_arr = inner_sdfg.arrays[dst]
            src_arr = state.sdfg.arrays[src]
            if size != dst_arr.total_size:
                if size == self.vector_width:
                    assert dst_arr.total_size == 1 and isinstance(dst_arr, dace.data.Scalar)
                    inner_sdfg.remove_data(dst, validate=False)
                    inner_sdfg.add_array(name=dst,
                                         shape=(self.vector_width, ),
                                         storage=self.vector_input_storage,
                                         dtype=dst_arr.dtype,
                                         transient=False,
                                         lifetime=dst_arr.lifetime)
                    if dst_arr.total_size == 1 and isinstance(dst_arr, dace.data.Scalar):
                        scalar_to_vector_conversions.add(dst)
                    scalar_to_vector_width_arrays.add(f"{dst}")

        # All transient scalars should be made into vectors
        for arr_name in copy.deepcopy(inner_sdfg.arrays):
            arr = inner_sdfg.arrays[arr_name]
            if ( isinstance(arr, dace.data.Scalar) or ( isinstance(arr, dace.data.Array) and arr.shape == (1,) ) ) and arr.transient is True:
                inner_sdfg.remove_data(arr_name, validate=False)
                inner_sdfg.add_array(
                    name=arr_name,
                    shape=(self.vector_width,),
                    dtype=arr.dtype,
                    storage=arr.storage,
                    location=arr.location,
                    transient=True,
                )
                scalar_to_vector_width_arrays.add(arr_name)


        # Get all scalars used in interstate edge assignments
        # They might need to be replaced with vector width arrays
        # These need to be also added to the candidate arrays
        interstate_edge_rhs_scalars = set()
        for edge in inner_sdfg.all_interstate_edges():
            free_syms = edge.data.free_symbols
            for k in edge.data.assignments:
                free_syms -= {k}
            scalar_free_syms = set()
            for k in free_syms:
                if k in inner_sdfg.arrays and isinstance(inner_sdfg.arrays[k], dace.data.Scalar):
                    scalar_free_syms.add(k)
            interstate_edge_rhs_scalars = interstate_edge_rhs_scalars.union(scalar_free_syms)

        # Handle indirect copies
        # Go through states find A[idx] pattern in any memlet
        expanded_symbols = set()
        for state in inner_sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in arrays_need_to_be_packed:
                    free_symbols = edge.data.free_symbols
                    # Look for the assignments in the interstate edges and expand them
                    non_expanded_free_symbols = free_symbols - expanded_symbols
                    expanded_symbols = expanded_symbols.union(free_symbols)
                    self._expand_interstate_assignments(inner_sdfg, non_expanded_free_symbols,
                                                        scalar_to_vector_width_arrays.union(interstate_edge_rhs_scalars))

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
                            new_subset = repl_subset_to_symbol_offset(subset=edge.data.subset, symbol_offset=str(i))
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
                            modified_edges.add(e1)
                            modified_edges.add(e2)

                        # Now update the subset
                        edge.data = dace.memlet.Memlet(expr=f"{src_node.data}[0:{self.vector_width}]")
                        modified_edges.add(edge)

        # If a state has a sink that is a scalar that has to be made into a vector array
        # We need to reduce it back to a scalar at the end of the day, do it here
        # First get input nodes and reduction types:
        for state in inner_sdfg.all_states():
            reduction_map: Dict[str, str] = dict()
            for edge in state.edges():
                src_node: dace.nodes.AccessNode = edge.src
                if isinstance(src_node, dace.nodes.AccessNode) and state.in_degree(src_node) == 0:
                    src_arr: dace.data.Data = state.sdfg.arrays[src_node.data]
                    if isinstance(src_arr, dace.data.Scalar) or (isinstance(src_arr, dace.data.Array) and src_arr.shape == (1,)):
                        assert isinstance(edge.dst, dace.nodes.Tasklet)
                        t: dace.nodes.Tasklet = edge.dst
                        op_str = get_op(t.code.as_string)
                        reduction_map[src_node.data] = op_str

            for edge in state.edges():
                if edge.data is not None:
                    dst_node: dace.nodes.AccessNode = edge.dst
                    if isinstance(dst_node, dace.nodes.AccessNode) and state.out_degree(dst_node) == 0:
                        dst_arr: dace.data.Array = state.sdfg.arrays[dst_node.data]
                        if isinstance(dst_arr, dace.data.Scalar) and dst_arr.transient is False:
                            old_data = f"{edge.data.data}"
                            dst_node.data = f"{edge.data.data}_vec"
                            vec_data = dst_node.data
                            edge.dst.data = vec_data
                            edge.data = dace.memlet.Memlet(expr=f"{old_data}[0:{self.vector_width}]", )
                            if vec_data not in state.sdfg.arrays:
                                orig_scalar = state.sdfg.arrays[old_data]
                                state.sdfg.add_array(name=vec_data,
                                                     shape=(self.vector_width, ),
                                                     dtype=orig_scalar.dtype,
                                                     storage=orig_scalar.storage,
                                                     transient=True,
                                                     lifetime=orig_scalar.lifetime)
                            modified_edges.add(edge)
                            dst_access = state.add_access(old_data)
                            modified_nodes.add(dst_access)
                            print(reduction_map)
                            state.sdfg.save("x.sdfg")
                            assert len(reduction_map) == 1
                            reduction_op = next(iter(reduction_map.values()))
                            nt = state.add_tasklet(
                                name="sum_up",
                                inputs={f"_in{i}"
                                        for i in range(0, self.vector_width)},
                                outputs={"_out"},
                                code="_out = " +
                                f" {reduction_op} ".join([f"_in{i}" for i in range(0, self.vector_width)]))
                            modified_nodes.add(nt)
                            for i in range(0, self.vector_width):
                                e1 = state.add_edge(edge.dst, None, nt, f"_in{i}",
                                                    dace.memlet.Memlet(expr=f"{vec_data}[{i}]"))
                                nt.add_in_connector(f"_in{i}")
                                modified_edges.add(e1)
                            e2 = state.add_edge(nt, "_out", dst_access, None, dace.memlet.Memlet(expr=f"{old_data}[0]"))
                            modified_edges.add(e2)
                            nt.add_out_connector("_out")
                            edge.data.data = vec_data

        # If we still have a scalar sink node that is not transient we need to de-duplicate writes
        for state in inner_sdfg.all_states():
            for node in state.nodes():
                if state.out_degree(node) == 0:
                    arr = state.sdfg.arrays[node.data]
                    if (arr.transient is False and (isinstance(arr, dace.data.Scalar) or
                        isinstance(arr, dace.data.Array) and arr.shape == (1,))):
                        raise Exception("At this point of the pass, no write to non-transient scalar sinks should remain")
                    if arr.transient is False and (isinstance(arr, dace.data.Array) and (arr.shape != (1,) or arr.shape != (self.vector_width,))):
                        touched_nodes, touched_edges = duplicate_access(state, node, self.vector_width)
                        modified_edges = modified_edges.union(touched_edges)
                        modified_nodes = modified_nodes.union(touched_nodes)

        # Now go through all edges and replace their subsets
        print("Scalar to vector width arrays:", scalar_to_vector_width_arrays)
        print("Array that need to be packed:", arrays_need_to_be_packed)
        for state in inner_sdfg.all_states():
            edges_to_replace = set()

            for edge in state.edges():
                if edge.data is not None:
                    if edge not in modified_edges:
                        src_node: dace.nodes.Node = edge.src
                        if state.in_degree(src_node) == 0:
                            if src_node.data not in scalar_to_vector_width_arrays and src_node not in arrays_need_to_be_packed:
                                continue
                        edges_to_replace.add(edge)

            old_subset = dace.subsets.Range([(0, 0, 1)])
            new_subset = dace.subsets.Range([(0, self.vector_width - 1, 1)])
            replace_memlet_expression(state, edges_to_replace, old_subset, new_subset, True)

        # Now replace tasklets
        for state in inner_sdfg.all_states():
            nodes = {n for n in state.nodes() if n not in modified_nodes}
            self._replace_tasklets_from_node_list(state, nodes)

    def _expand_interstate_assignments(self, sdfg: dace.SDFG, syms: Set[str], candidate_arrays: Set[str]):
        duplicated_symbols = set()
        syms_to_rm = set()
        for edge in sdfg.all_interstate_edges():
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
                        print("Candidate Arrays:", candidate_arrays)
                        for ca in candidate_arrays:
                            assert ca in sdfg.arrays
                            ca_data = sdfg.arrays[ca]
                            print("Candidate Array Name:", ca, "Candidate Array:", ca_data)
                            if isinstance(ca_data, dace.data.Scalar) or (isinstance(ca_data, dace.data.Array) and ca_data.shape == (1,)):
                                ca_scl = ca_data
                                assert ca_scl.transient
                                sdfg.remove_data(ca, validate=False)
                                sdfg.add_array(
                                    name=ca,
                                    shape=(self.vector_width,),
                                    dtype=ca_scl.dtype,
                                    storage=ca_scl.storage,
                                    location=ca_scl.location,
                                    transient=True,
                                    lifetime=ca_scl.lifetime,
                                    find_new_name=False,
                                )
                            #nv = nv.replace(ca, f"{ca}[{i}]")
                            nv = split_in_token(nv, ca, f"{ca}[{i}]")
                        new_assignments[f"{k}{i}"] = nv
                    duplicated_symbols.add(k)
                    syms_to_rm.add(k)
                else:
                    new_assignments[k] = v
            edge.data.assignments = new_assignments
        for sym in syms_to_rm:
            sdfg.remove_symbol(sym)
        return duplicated_symbols

    def _extend_temporary_scalars(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges_to_rm = set()
        edges_to_add = set()
        nodes_to_rm = set()
        for node in nodes:
            if isinstance(node, dace.nodes.AccessNode):
                desc = state.parent_graph.sdfg.arrays[node.data]
                if isinstance(desc, dace.data.Scalar):
                    if f"{node.data}_vec" not in state.parent_graph.sdfg.arrays:
                        state.sdfg.add_array(
                            name=f"{node.data}_vec",
                            shape=(self.vector_width, ),
                            dtype=desc.dtype,
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
                        raise Exception(f"Unsupported Type for in_edge.src got type {type(in_edge.src)}, need AccessNode ({in_edge.data})")
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
                        raise Exception(f"Unsupported Type for out_edge.dst got type {type(out_edge.dst)}, need AccessNode ({in_edge.data})")
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

    def _replace_tasklets(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        self._replace_tasklets_from_node_list(state, nodes)

    def _replace_tasklets_from_node_list(self, state: SDFGState, nodes: Iterable[dace.nodes.Node]):
        for node in nodes:
            if isinstance(node, dace.nodes.Tasklet):
                tasklet_info = classify_tasklet(state, node)
                print("Tasklet:", node, " has info:", tasklet_info)
                instantiate_tasklet_from_info(state, node, tasklet_info, self.vector_width, self.templates)

    def _offset_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, offsets: List[dace.symbolic.SymExpr],
                        dataname: str, vectorization_number: int):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        edges = state.all_edges(*nodes)
        self._offset_memlets_from_edge_list(state, edges, offsets, dataname, vectorization_number)

    def _offset_memlets_from_edge_list(self, state: SDFGState, edges: Iterable[Edge[Memlet]],
                                       offsets: List[dace.symbolic.SymExpr], dataname: str, vectorization_number: int):
        for edge in edges:
            if edge.data is None or edge.data.data != dataname:
                continue
            memlet: dace.memlet.Memlet = edge.data

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            new_memlet = dace.memlet.Memlet(
                data=f"{memlet.data}_vec_k{vectorization_number}",
                subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
            )
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _copy_in_and_copy_out(self, state: SDFGState, map_entry: dace.nodes.MapEntry, vectorization_number: int):
        map_exit = state.exit_node(map_entry)
        data_and_offsets = list()
        for ie in state.in_edges(map_entry):
            # If input storage is not registers need to copy in
            array = state.parent_graph.sdfg.arrays[ie.data.data]
            if array.storage != self.vector_input_storage:
                # Add new array, if not there
                if f"{ie.data.data}_vec_k{vectorization_number}" not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=f"{ie.data.data}_vec_k{vectorization_number}",
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=self.vector_width * array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                an = state.add_access(f"{ie.data.data}_vec_k{vectorization_number}")
                src, src_conn, dst, dst_conn, data = ie
                state.remove_edge(ie)
                state.add_edge(src, src_conn, an, None, copy.deepcopy(data))
                state.add_edge(an, None, map_entry, ie.dst_conn,
                               dace.memlet.Memlet(f"{ie.data.data}_vec_k{vectorization_number}[0:{self.vector_width}]"))

                memlet: dace.memlet.Memlet = ie.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                for data, _off in data_and_offsets:
                    if data == dataname:
                        if _off != offsets:
                            raise ValueError(
                                f"Cannot handle multiple input edges from the same array {dataname} to the same map {map_entry} in state {state}"
                            )
                data_and_offsets.append((dataname, offsets))

        for oe in state.out_edges(map_exit):
            array = state.parent_graph.sdfg.arrays[oe.data.data]
            if array.storage != self.vector_output_storage:
                if f"{oe.data.data}_vec_k{vectorization_number}" not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=f"{oe.data.data}_vec_k{vectorization_number}",
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=self.vector_width * array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                an = state.add_access(f"{oe.data.data}_vec_k{vectorization_number}")
                src, src_conn, dst, dst_conn, data = oe
                state.remove_edge(oe)
                state.add_edge(map_exit, src_conn, an, None,
                               dace.memlet.Memlet(f"{oe.data.data}_vec_k{vectorization_number}[0:{self.vector_width}]"))
                state.add_edge(an, None, dst, dst_conn, copy.deepcopy(data))

                memlet: dace.memlet.Memlet = oe.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                for data, _off in data_and_offsets:
                    if data == dataname:
                        if _off != offsets:
                            raise NotImplementedError(
                                f"Vectorization can't handle when data appears both in input and output sets of a map")
                data_and_offsets.append((dataname, offsets))

        for dataname, offsets in data_and_offsets:
            self._offset_memlets(state, map_entry, offsets, dataname, vectorization_number)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        stride_type = assert_strides_are_packed_C_or_packed_Fortran(sdfg)
        self._stride_type = stride_type
        assert_last_dim_of_maps_are_contigous_accesses(sdfg)
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
            if len(parent_scopes) > 0:
                #num_vectorized += self._vectorize_sdfg(sdfg=nested_sdfg.sdfg,
                #                                        has_parent_map=True,
                #                                        num_vectorized=num_vectorized)
                pass
            else:
                raise NotImplementedError(
                    "NestedSDFGs without parent map scopes are not supported, they must have been inlined if the pipeline has been called."
                    "If pipeline has been called verify why InlineSDFG failed, otherwise call InlineSDFG")

        sdfg.append_global_code(cpp_code=self.global_code, location=self.global_code_location)
        return None
