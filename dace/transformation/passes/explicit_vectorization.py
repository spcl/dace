# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import dace
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.passes import analysis as ap, pattern_matching as pmp
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import IntegerPowerToMult, RemoveFPTypeCasts
from dace.transformation.dataflow.tiling import MapTiling

class ExplicitVectorizationPipelineGPU(ppl.Pipeline):
    _gpu_global_code = """
__host__ __device__ __forceinline__ void vector_mult(const double * __restrict__ c, const double * __restrict__ a, double * __restrict__ b) {{
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}
__host__ __device__ __forceinline__ void vector_mult(const double * __restrict__ b, const double * __restrict__ a, const double constant) {{
    double cReg[{vector_width}];
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}
__host__ __device__ __forceinline__ void vector_add(const double * __restrict__ c, const double * __restrict__ a, double * __restrict__ b) {{
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}
__host__ __device__ __forceinline__ void vector_add(const double * __restrict__ b, const double * __restrict__ a, const double constant) {{
    double cReg[{vector_width}];
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}
__host__ __device__ __forceinline__ void vector_copy(const double * __restrict__ dst, const double * __restrict__ src) {{
    #pragma unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i];
    }}
}}
"""
    def __init__(self, vector_width):
        passes = [
            RemoveFPTypeCasts(),
            IntegerPowerToMult(),
            SplitTasklets(),
            ExplicitVectorization(
                templates={
                    "*": "vector_mult({lhs}, {rhs1}, {rhs2});",
                    "+": "vector_add({lhs}, {rhs1}, {rhs2});",
                    "=": "vector_copy({lhs}, {rhs1});",
                    "c+": "vector_add({lhs}, {rhs1}, {constant});",
                    "c*": "vector_mult({lhs}, {rhs1}, {constant});",
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.GPU_Global,
                vector_output_storage=dace.dtypes.StorageType.GPU_Global,
                global_code=ExplicitVectorizationPipelineGPU._gpu_global_code.format(vector_width=vector_width)
            )
        ]
        super().__init__(passes)

@properties.make_properties
@transformation.explicit_cf_compatible
class ExplicitVectorization(ppl.Pass):
    templates = properties.DictProperty(
        key_type=str,
        value_type=str,
    )
    vector_width = properties.Property(
        dtype=int,
        default=4
    )
    vector_input_storage = properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.Register
    )
    vector_output_storage = properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.Register
    )
    global_code = properties.Property(
        dtype=str,
        default=""
    )
    
    def __init__(self, templates, vector_width, vector_input_storage, vector_output_storage, global_code):
        super().__init__()
        self.templates = templates
        self.vector_width = vector_width
        self.vector_input_storage = vector_input_storage
        self.vector_output_storage = vector_output_storage
        self.global_code = global_code

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.States | ppl.Modifies.Tasklets | ppl.Modifies.NestedSDFGs | ppl.Modifies.Scopes | ppl.Modifies.Descriptors

    def depends_on(self):
        return {IntegerPowerToMult, SplitTasklets}

    def _get_all_parent_scopes(self, state: SDFGState, node: dace.nodes.NestedSDFG):
        parents = list()
        parent = state.scope_dict()[node]
        while parent is not None:
            parents.append(parent)
            parent = state.scope_dict()[parent]
        return parents

    def _vectorize_map(self, state: SDFGState, first_map_entry: dace.nodes.MapEntry):
        # Get the innermost maps 
        assert isinstance(first_map_entry, dace.nodes.MapEntry)
        parents = dict()
        for node in state.all_nodes_between(first_map_entry, state.exit_node(first_map_entry)).union({first_map_entry}):
            if isinstance(node, dace.nodes.MapEntry):
                parent = state.scope_dict()[node]
                parents[node] = parent

        maps_that_are_not_parents = set()
        for k in parents:
            if parents[k] not in parents:
                maps_that_are_not_parents.add(k)

        for inner_map_entry in maps_that_are_not_parents:
            tile_sizes = [1 for _ in inner_map_entry.map.range]
            tile_sizes[-1] = self.vector_width
            MapTiling.apply_to(
                sdfg=state.parent_graph.sdfg,
                map_entry=inner_map_entry,
                options={"tile_sizes": tile_sizes},
            )
            new_inner_map = inner_map_entry
            new_inner_map.schedule = dace.dtypes.ScheduleType.Sequential
            old_inner_map = state.entry_node(new_inner_map)

            (b,e,s) = new_inner_map.map.range[0]
            assert (e - b + 1).approx == self.vector_width, f"MapTiling should have created a map with range of size {self.vector_width}, found {(e - b + 1)}"
            assert s == 1, f"MapTiling should have created a map with stride 1, found {s}"
            # Vector the range by for example making [0:4:1] to [0:4:4]
            new_inner_map.map.range = dace.subsets.Range([(0, self.vector_width - 1, self.vector_width)])

            # Updates memlets from [k, i] to [k, i:i+4]
            self._extend_memlets(state, new_inner_map)
            # Replactes tasklets of form A op B to vectorized_op(A, B)
            self._replace_tasklets(state, new_inner_map)
            # Copies in data to the storage needed by the vector unit
            # Copies out data from the storage needed by the vector unit
            self._copy_in_and_copy_out(state, new_inner_map)
            # If tasklet -> sclar -> tasklet, now we have, 
            # vector_tasklet -> scalar -> vector_tasklet
            # makes the scalar into vector
            self._extend_temporary_scalars(state, new_inner_map)


    def _extend_temporary_scalars(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges_to_rm = set()
        edges_to_add = set()
        nodes_to_rm = set()
        state.sdfg.save("x00.sdfg")
        for node in nodes:
            if isinstance(node, dace.nodes.AccessNode):
                desc = state.parent_graph.sdfg.arrays[node.data]
                if isinstance(desc, dace.data.Scalar):
                    if f"{node.data}_vec" not in state.parent_graph.sdfg.arrays:
                        state.sdfg.add_array(
                            name=f"{node.data}_vec",
                            shape=(self.vector_width,),
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


    def _get_vector_templates(self, rhs1: str, rhs2: str, lhs: str, constant: Union[str, None], op: str):
        if rhs2 is None:
            if constant is None:
                new_code = self.templates[op].format(rhs1=rhs1, lhs=lhs, op=op, vector_width=self.vector_width)
            else:
                new_code = self.templates[op].format(rhs1=rhs1, constant=constant, lhs=lhs, op=op, vector_width=self.vector_width)
        else:
            new_code = self.templates[op].format(rhs1=rhs1, rhs2=rhs2, lhs=lhs, op=op, vector_width=self.vector_width)
        return new_code

    def _extend_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges = state.all_edges(*nodes)
        for edge in edges:
            memlet : dace.memlet.Memlet = edge.data
            # If memlet is equal to the array shape, we have a clear case of over-approximation
            # For the check we know: last dimension results in contiguous access,
            # We should memlets of form:
            # [(b, e, s), ...] the b, e, s may depend on i.
            # If b is i, e is i (inclusive range), then extension needs to result with (i, i + vector_width - 1, 1)
            # We can assume (and check) that s == 1
            # if b is 2*i and e is 2*i then we should extend the i in the end with 2*(i + vector_width - 1)
            map_entry : dace.nodes.MapEntry = state.entry_node(edge.src) if not isinstance(edge.src, dace.nodes.MapEntry) else state.entry_node(edge.dst)
            used_param = map_entry.map.params[-1] # Regardless of offsets F / C we assume last parameter of the map is useful
            new_range_list = [(b, e, s) for (b, e, s) in memlet.subset]
            stride_offset = 0 if self._stride_type == "F" else -1 # lest-contig is Fortran and right-contig is C
            range_tup: Tuple[dace.symbolic.SymExpr, dace.symbolic.SymExpr, dace.symbolic.SymExpr] = new_range_list[stride_offset]
            lb, le, ls = range_tup
            assert ls == 1, f"Previous checks must have ensured the final dimension should result in unit-stride access"
            new_range_list[stride_offset] = (lb, le.subs(used_param, dace.symbolic.SymExpr(f"({self.vector_width} - 1) + {used_param}")), ls) if lb == le else (lb, le.subs(used_param, dace.symbolic.SymExpr(f"({self.vector_width} * {used_param}) - 1")), ls)

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            new_memlet = dace.memlet.Memlet(
                data=memlet.data,
                subset=dace.subsets.Range(new_range_list),
            )
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _extract_constant(self, src: str) -> str:
        tree = ast.parse(src)
        
        for node in ast.walk(tree):
            # Direct constant
            if isinstance(node, ast.Constant):
                return str(node.value)
            # Unary operation on constant (like -3.14)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
                if isinstance(node.op, ast.USub):
                    return f"-{node.operand.value}"
                elif isinstance(node.op, ast.UAdd):
                    return str(node.operand.value)
        
        raise ValueError("No constant found")

    def _replace_tasklets(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        for node in nodes:
            if isinstance(node, dace.nodes.Tasklet):
                # Replace the code of the tasklet with vectorized code
                # For example:
                # a = b * c;  ->  for (int i = 0; i < 4; i++) { a[i] = b[i] * c[i]; }
                # Only accepted no op is a = b
                if len(node.in_connectors) == 1 and len(node.out_connectors) == 1:
                    in_conn = next(iter(node.in_connectors.keys()))
                    out_conn = next(iter(node.out_connectors.keys()))
                    if node.code.as_string == f"{out_conn} = {in_conn};" or node.code.as_string == f"{out_conn} = {in_conn}":
                        op = "="
                        rhs1 = list(node.in_connectors.keys())[0]
                        lhs = next(iter(node.out_connectors.keys()))
                        node.code = properties.CodeBlock(code=self._get_vector_templates(rhs1=rhs1, rhs2=None, constant=None, lhs=lhs, op=op), language=dace.Language.CPP)
                    else:
                        op = self._extract_single_op(node.code.as_string)
                        op = f"c{op}"
                        rhs1 = list(node.in_connectors.keys())[0]
                        lhs = next(iter(node.out_connectors.keys()))
                        constant = self._extract_constant(node.code.as_string)
                        node.code = properties.CodeBlock(code=self._get_vector_templates(rhs1=rhs1, rhs2=None, constant=constant, lhs=lhs, op=op), language=dace.Language.CPP)
                else:
                    assert len(node.in_connectors) == 2 or len(node.in_connectors) == 0, f"Only support tasklets with 2 inputs (binary ops) or 0 inputs (unary ops), found {node.in_connectors} in tasklet {node} in state {state}"
                    assert len(node.out_connectors) == 1, f"Only support tasklets with 1 output, found {node.out_connectors} in tasklet {node} in state {state}"
                    op = self._extract_single_op(node.code.as_string)
                    rhs1, rhs2 = list(node.in_connectors.keys())
                    lhs = next(iter(node.out_connectors.keys()))
                    node.code = properties.CodeBlock(code=self._get_vector_templates(rhs1=rhs1, rhs2=rhs2, lhs=lhs, constant=None, op=op), language=dace.Language.CPP)

    def _offset_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, offsets: List[dace.symbolic.SymExpr], dataname: str):
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges = state.all_edges(*nodes)
        for edge in edges:
            if edge.data is None or edge.data.data != dataname:
                continue
            memlet : dace.memlet.Memlet = edge.data

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            new_memlet = dace.memlet.Memlet(
                data=f"{memlet.data}_vec",
                subset=dace.subsets.Range([(0, self.vector_width - 1, 1)]),
            )
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _copy_in_and_copy_out(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        map_exit = state.exit_node(map_entry)
        data_and_offsets = list()
        for ie in state.in_edges(map_entry):
            # If input storage is not registers need to copy in
            array = state.parent_graph.sdfg.arrays[ie.data.data]
            if array.storage != self.vector_input_storage:
                # Add new array, if not there
                if f"{ie.data.data}_vec" not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(
                        name=f"{ie.data.data}_vec",
                        shape=(self.vector_width,),
                        dtype=array.dtype,
                        storage=self.vector_input_storage,
                        transient=True,
                        allow_conflicts=False,
                        alignment=self.vector_width * array.dtype.bytes,
                        find_new_name=False,
                        may_alias=False
                    )
                an = state.add_access(f"{ie.data.data}_vec")
                src, src_conn, dst, dst_conn, data = ie
                state.remove_edge(ie)
                state.add_edge(src, src_conn, an, None, copy.deepcopy(data))
                state.add_edge(an, None, map_entry, ie.dst_conn, dace.memlet.Memlet(f"{ie.data.data}_vec[0:{self.vector_width}]"))

                memlet: dace.memlet.Memlet = ie.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                for data, _off in data_and_offsets:
                    if data == dataname:
                        if _off != offsets:
                            raise ValueError(f"Cannot handle multiple input edges from the same array {dataname} to the same map {map_entry} in state {state}")
                data_and_offsets.append((dataname, offsets))

        for oe in state.out_edges(map_exit):
            array = state.parent_graph.sdfg.arrays[oe.data.data]
            if array.storage != self.vector_output_storage:
                if f"{oe.data.data}_vec" not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(
                        name=f"{oe.data.data}_vec",
                        shape=(self.vector_width,),
                        dtype=array.dtype,
                        storage=self.vector_input_storage,
                        transient=True,
                        allow_conflicts=False,
                        alignment=self.vector_width * array.dtype.bytes,
                        find_new_name=False,
                        may_alias=False
                    )
                an = state.add_access(f"{oe.data.data}_vec")
                src, src_conn, dst, dst_conn, data = oe
                state.remove_edge(oe)
                state.add_edge(map_exit, src_conn, an, None, dace.memlet.Memlet(f"{oe.data.data}_vec[0:{self.vector_width}]"))
                state.add_edge(an, None, dst, dst_conn, copy.deepcopy(data))

                memlet: dace.memlet.Memlet = ie.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                for data, _off in data_and_offsets:
                    if data == dataname:
                        if _off != offsets:
                            raise NotImplementedError(f"Vectorization can't handle when data appears both in input and output sets of a map")
                data_and_offsets.append((dataname, offsets))

        for dataname, offsets in data_and_offsets:
            self._offset_memlets(state, map_entry, offsets, dataname)

    def _vectorize_sdfg(self, sdfg: dace.SDFG, has_parent_map: bool):
        assert has_parent_map is True, f"Unhandled case, NestedSDFGs need to have a parent map that has been tiled"
        raise NotImplementedError("Vectorization of NestedSDFGs not implemented yet")

    def _check_stride(self, sdfg: dace.SDFG):
        stride_type = None

        for arr, desc in sdfg.arrays.items():
            if not isinstance(desc, dace.data.Array):
                continue

            # Check unit stride exists
            has_unit_stride = desc.strides[0] == 1 or desc.strides[-1] == 1
            assert has_unit_stride, f"Array {arr} needs unit stride in first or last dimension: {desc.strides}"

            # Determine stride type
            current_type = "F" if desc.strides[0] == 1 else "C"

            # Consistency check
            if stride_type is None:
                stride_type = current_type
            elif stride_type != current_type:
                raise ValueError("All arrays must have consistent stride ordering (all F or all C)")
        
        return stride_type

    def _check_last_dim_of_map_is_contigupus_access(self, sdfg: dace.SDFG):
        checked_map_entries = set()
        for state in sdfg.all_states():
            for node in state.nodes():
                # Ensure we work with innermost maps by skipping maps and getting parent nodes of tasklets and such
                if isinstance(node, dace.nodes.MapEntry) or isinstance(node, dace.nodes.MapExit):
                    continue

                # Ensure all tasklets have parent maps
                map_entry = state.scope_dict()[node]
                if map_entry is None:
                    if isinstance(node, dace.nodes.Tasklet):
                        raise ValueError(f"All nodes must be within a map, found node {node} outside of any map in state {state}.")
                    else:
                        continue
                else:
                    if not isinstance(map_entry, dace.nodes.MapEntry):
                        raise ValueError(f"Parent scope of node {node} is not a map, found {map_entry} in state {state}.")
                    assert map_entry is not None
                    checked_map_entries.add(map_entry)

                # If we have checked a map entry (and nodes within its body) then skip it
                if map_entry not in checked_map_entries:
                    assert isinstance(map_entry, dace.nodes.MapEntry), f"Parent scope of node {node} is not a map, returned value is {map_entry}."
                    nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
                    edges = state.all_edges(*nodes)
                    for edge in edges:
                        memlet : dace.memlet.Memlet = edge.data
                        free_symbols = memlet.subset.free_symbols
                        last_param = list(map_entry.map.params)[-1]
                        if last_param not in free_symbols:
                            raise ValueError(f"Last map parameter {last_param} must be in the memlet {memlet}, not in this case - edge: {edge}, state: {state}")

    def _extract_single_op(self, src: str) -> str:
        BINOP_SYMBOLS = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }

        UNARY_SYMBOLS = {
            ast.UAdd: "+",
            ast.USub: "-",
        }

        SUPPORTED = {'*', '+', '-', '/', 'abs', 'exp', 'max', 'min', 'sqrt'}

        tree = ast.parse(src)
        found = None

        for node in ast.walk(tree):
            op = None

            # Binary op (remove the float constant requirement)
            if isinstance(node, ast.BinOp):
                op = BINOP_SYMBOLS.get(type(node.op), None)

            # Unary op (remove the float constant requirement)
            elif isinstance(node, ast.UnaryOp):
                op = UNARY_SYMBOLS.get(type(node.op), None)

            # Function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    op = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    op = node.func.attr

            if op is None:
                continue

            if op not in SUPPORTED:
                raise ValueError(f"Unsupported operation: {op}")

            if found is not None:
                raise ValueError("More than one supported operation found")

            found = op

        if found is None:
            raise ValueError("No supported operation found")

        return found

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        self._stride_type = self._check_stride(sdfg)
        self._check_last_dim_of_map_is_contigupus_access(sdfg)

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
        top_level_maps = list()
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    map_entry : dace.nodes.MapEntry = node
                    if state.scope_dict()[map_entry] is None:
                        top_level_maps.append((map_entry, state))

        for map_entry, state in top_level_maps:
            self._vectorize_map(state, map_entry)


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
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    nested_sdfg : dace.nodes.NestedSDFG = node
                    parent_scopes = self._get_all_parent_scopes(state, node)
                    if len(parent_scopes) > 0:
                        self._vectorize_sdfg(nested_sdfg.sdfg)

        sdfg.append_global_code(cpp_code=self.global_code, location="cuda")
        return None
