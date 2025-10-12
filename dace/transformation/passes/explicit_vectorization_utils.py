# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re

import sympy
import dace
import ast
from typing import Dict, Iterable, Set, Tuple, Union
from dace import SDFGState
from dace import Any
from dace import List
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from enum import Enum
import dace.sdfg.utils as sdutil


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    # Convert string keys to dace symbols
    sym_repl_dict = {dace.symbolic.symbol(k) if isinstance(k, str) else k: v for k, v in repl_dict.items()}
    # Replace symbolic expressions according to the dictionary
    new_range_list = []
    for (b, e, s) in subset:
        new_b = copy.deepcopy(b).subs(sym_repl_dict)
        new_e = copy.deepcopy(e).subs(sym_repl_dict)
        new_s = copy.deepcopy(s).subs(sym_repl_dict)
        new_range_list.append((new_b, new_e, new_s))

    new_subset = dace.subsets.Range(ranges=new_range_list)
    return new_subset


def repl_subset_to_symbol_offset(subset: dace.subsets.Range, symbol_offset: str) -> dace.subsets.Range:
    free_syms = subset.free_symbols
    print("Free symbols in subset:", free_syms)

    repl_dict = {str(free_sym): str(free_sym) + symbol_offset for free_sym in free_syms}
    print("Generated replacement dictionary with offset:", repl_dict)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    print("Subset after symbol offset replacement:", new_subset)
    return new_subset


def replace_memlet_expression(state: SDFGState, edges: Iterable[Edge[Memlet]], old_subset_expr: dace.subsets.Range,
                              new_subset_expr: dace.subsets.Range, repl_scalars_with_arrays: bool,
                              edges_to_skip: Set[Edge[Memlet]]) -> Set[str]:
    arr_dim = [((e + 1 - b) // s) for (b, e, s) in new_subset_expr]

    for edge in edges:
        src_node: dace.nodes.Node = edge.src
        dst_node: dace.nodes.Node = edge.dst

        if edge.data is not None and edge.data.subset == old_subset_expr:
            if edge in edges_to_skip:
                raise Exception("AA")
            if repl_scalars_with_arrays:
                for data_node in [src_node, dst_node]:
                    if isinstance(data_node, dace.nodes.AccessNode):
                        arr = state.sdfg.arrays[data_node.data]
                        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array)
                                                                 and arr.shape == (1, )):
                            state.sdfg.remove_data(data_node.data, validate=False)
                            state.sdfg.add_array(name=data_node.data,
                                                 shape=tuple(arr_dim),
                                                 dtype=arr.dtype,
                                                 storage=arr.storage,
                                                 location=arr.location,
                                                 transient=arr.transient,
                                                 lifetime=arr.lifetime)
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))


def extract_constant(src: str) -> str:
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


def extract_single_op(src: str) -> str:
    print(f"Extract single op from {src}")
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

    CMP_SYMBOLS = {
        ast.Gt: ">",
        ast.Lt: "<",
        ast.GtE: ">=",
        ast.LtE: "<=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }

    SUPPORTED = {'*', '+', '-', '/', 'abs', 'exp', 'sqrt'}

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
        elif isinstance(node, ast.Compare):
            assert len(node.ops) == 1
            op = CMP_SYMBOLS.get(type(node.ops[0]), None)

        # Function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                op = node.func.id
            elif isinstance(node.func, ast.Attribute):
                op = node.func.attr

        if op is None:
            continue

        if op not in SUPPORTED:
            print(f"Found unsupported op {op} in {src}")

        if found is not None:
            raise ValueError("More than one supported operation found")

        found = op

    if found is None:
        raise ValueError(f"No supported operation found for code_str: {src}")

    return found


def has_maps(sdfg: dace.SDFG):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(map_entry: dace.nodes.MapEntry, state: SDFGState) -> bool:
    nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    if any({isinstance(node, dace.nodes.MapEntry) for node in nodes_between}):
        return False
    for node in nodes_between:
        if isinstance(node, dace.nodes.NestedSDFG):
            sdfg_has_maps = has_maps(node.sdfg)
            if sdfg_has_maps:
                return False
    return True


def assert_maps_consist_of_single_nsdfg_or_no_nsdfg(sdfg: dace.SDFG):
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            all_nodes = g.all_nodes_between(n, g.exit_node(n))
            assert (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)) or (len(
                {_n
                 for _n in all_nodes if isinstance(_n, dace.nodes.NestedSDFG)}) == 0)


def to_ints(sym_epxr: dace.symbolic.SymExpr):
    try:
        return int(sym_epxr)
    except:
        return None


def get_vector_max_access_ranges(state: SDFGState, node: dace.nodes.NestedSDFG):
    sdict = state.scope_dict()
    vmap = sdict[node]
    v_params_and_begins = dict()
    v_params_and_begins_rev = dict()
    d_params_and_begins = dict()
    d_params_and_ends = dict()
    d_params_and_begins_rev = dict()
    for p, (b, e, s) in zip(vmap.map.params, vmap.map.range):
        v_params_and_begins[p] = str(b)
        v_params_and_begins_rev[str(b)] = p
    dmap = sdict[vmap]
    for p, (b, e, s) in zip(dmap.map.params, dmap.map.range):
        d_params_and_begins[p] = str(b)
        d_params_and_begins_rev[str(b)] = p
        d_params_and_ends[p] = str(e)

    param_ends = dict()
    for p in vmap.map.params:
        param_ends[p] = d_params_and_ends[v_params_and_begins[p]]
    return param_ends


def add_copies_before_and_after_nsdfg(
    state: SDFGState,
    nsdfg_node: dace.nodes.NestedSDFG,
    required_shape: Tuple[int],
    copy_to_storage: dace.dtypes.StorageType,
):
    for ie in state.in_edges(nsdfg_node):
        if ie.data is not None:
            in_data_name: str = ie.data.data
            nsdfg_data_name: str = ie.dst_conn
            subset_lengths = tuple([(e + 1 - b) // s for b, e, s in ie.data.subset])
            subset_lengths_packed = tuple([(e + 1 - b) // s for b, e, s in ie.data.subset if (e + 1 - b) != 1])
            if (required_shape is None or subset_lengths == required_shape or subset_lengths_packed == required_shape):
                # Insert copy
                orig_arr = state.sdfg.arrays[in_data_name]
                if orig_arr.storage == copy_to_storage:
                    continue
                if orig_arr.transient is True and orig_arr.storage == dace.dtypes.StorageType.Default:
                    orig_arr.storage = copy_to_storage
                    continue

                if f"{in_data_name}_vec" not in state.sdfg.arrays:
                    arr_name, arr = state.sdfg.add_array(name=f"{in_data_name}_vec",
                                                         shape=subset_lengths,
                                                         dtype=orig_arr.dtype,
                                                         location=orig_arr.location,
                                                         transient=True,
                                                         find_new_name=False,
                                                         storage=copy_to_storage)
                    arr.setzero = True
                else:
                    arr_name = f"{in_data_name}_vec"
                    assert state.sdfg.arrays[arr_name].storage == copy_to_storage
                arr_access = state.add_access(arr_name)

                # Impl. proper min
                state.sdfg.save("x.sdfg")
                params_and_end_ranges = get_vector_max_access_ranges(state, nsdfg_node)
                nrange_list = []
                assert len(params_and_end_ranges) == 1
                max_range = next(iter(params_and_end_ranges.values()))
                for (b, e, s) in ie.data.subset:
                    ne = dace.symbolic.SymExpr(f"Min({e}, {max_range})")
                    nrange_list.append((b, ne, s))
                nmemlet = dace.memlet.Memlet(data=ie.data.data, subset=dace.subsets.Range(nrange_list))

                state.add_edge(ie.src, ie.src_conn, arr_access, None, nmemlet)
                full_subset_str = ", ".join([f"{0}:{sl}" for sl in subset_lengths])
                state.add_edge(arr_access, None, nsdfg_node, nsdfg_data_name,
                               dace.memlet.Memlet(f"{arr_name}[{full_subset_str}]"))

                # Remove only after adding to avoid disconnecting the component
                state.remove_edge(ie)

    for oe in state.out_edges(nsdfg_node):
        if oe.data is not None:
            out_data_name: str = oe.data.data
            nsdfg_data_name: str = oe.src_conn
            subset_lengths = tuple([(e + 1 - b) // s for b, e, s in ie.data.subset])
            subset_lengths_packed = tuple([(e + 1 - b) // s for b, e, s in ie.data.subset if (e + 1 - b) != 1])
            if (required_shape is None or subset_lengths == required_shape or subset_lengths_packed == required_shape):
                # Insert copy
                orig_arr = state.sdfg.arrays[out_data_name]
                if orig_arr.storage == copy_to_storage:
                    continue
                if orig_arr.transient is True and orig_arr.storage == dace.dtypes.StorageType.Default:
                    orig_arr.storage = copy_to_storage
                    continue

                if f"{out_data_name}_vec" not in state.sdfg.arrays:
                    arr_name, arr = state.sdfg.add_array(name=f"{out_data_name}_vec",
                                                         shape=subset_lengths,
                                                         dtype=orig_arr.dtype,
                                                         location=orig_arr.location,
                                                         transient=True,
                                                         find_new_name=False,
                                                         storage=copy_to_storage)
                    arr.setzero = True
                else:
                    arr_name = f"{out_data_name}_vec"
                    assert state.sdfg.arrays[arr_name].storage == copy_to_storage
                arr_access = state.add_access(arr_name)

                full_subset_str = ", ".join([f"{0}:{sl}" for sl in subset_lengths])
                state.add_edge(nsdfg_node, nsdfg_data_name, arr_access, None,
                               dace.memlet.Memlet(f"{arr_name}[{full_subset_str}]"))

                params_and_end_ranges = get_vector_max_access_ranges(state, nsdfg_node)
                nrange_list = []
                assert len(params_and_end_ranges) == 1
                max_range = next(iter(params_and_end_ranges.values()))
                for (b, e, s) in ie.data.subset:
                    ne = dace.symbolic.SymExpr(f"Min({e}, {max_range})")
                    nrange_list.append((b, ne, s))
                nmemlet = dace.memlet.Memlet(data=ie.data.data, subset=dace.subsets.Range(nrange_list))
                state.add_edge(arr_access, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

                # Remove only after adding to avoid disconnecting the component
                state.remove_edge(oe)


def get_op(expr_str: str):
    node = ast.parse(expr_str).body[0].value
    op_dict = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}

    return op_dict[type(node.op)]


def match_connector_to_data(state: dace.SDFGState, tasklet: dace.nodes.Tasklet):
    tdict = dict()
    for ie in state.in_edges(tasklet):
        if ie.data is not None:
            tdict[ie.dst_conn] = state.sdfg.arrays[ie.data.data]

    return tdict


def get_scalar_and_array_arguments(state: dace.SDFGState, tasklet: dace.nodes.Tasklet):
    tdict = match_connector_to_data(state, tasklet)
    scalars = {k for k, v in tdict.items() if isinstance(v, dace.data.Scalar)}
    arrays = {k for k, v in tdict.items() if isinstance(v, dace.data.Array)}
    return scalars, arrays


def assert_strides_are_packed_C_or_packed_Fortran(sdfg: dace.SDFG) -> Union[str, None]:
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


def find_state_of_nsdfg_node(root_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG) -> dace.SDFGState:
    for n, g in root_sdfg.all_nodes_recursive():
        if n == nsdfg_node:
            return root_sdfg
    raise Exception(f"State of the nsdfg node ({nsdfg_node}) not found in the root SDFG ({root_sdfg.label})")


def assert_last_dim_of_maps_are_contigous_accesses(sdfg: dace.SDFG):
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
                    parent_nsdfg = state.sdfg.parent_nsdfg_node
                    # Ok if no parent
                    if parent_nsdfg is None:
                        continue
                    parent_state = find_state_of_nsdfg_node(sdfg, node)
                    parent_scope = parent_state.scope_dict()[parent_nsdfg]
                    if parent_scope is None or (not isinstance(parent_scope, dace.nodes.MapEntry)):
                        raise Exception(
                            f"No NSDFGs that are not within Map scopes should be left, check {parent_nsdfg} in state {parent_state}. Call inlineSDFG"
                        )
                else:
                    continue
            else:
                if not isinstance(map_entry, dace.nodes.MapEntry):
                    raise ValueError(f"Parent scope of node {node} is not a map, found {map_entry} in state {state}.")
                assert map_entry is not None
                checked_map_entries.add(map_entry)

            # If we have checked a map entry (and nodes within its body) then skip it
            if map_entry not in checked_map_entries:
                assert isinstance(
                    map_entry,
                    dace.nodes.MapEntry), f"Parent scope of node {node} is not a map, returned value is {map_entry}."
                nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
                edges = state.all_edges(*nodes)
                for edge in edges:
                    memlet: dace.memlet.Memlet = edge.data
                    free_symbols = memlet.subset.free_symbols
                    last_param = list(map_entry.map.params)[-1]
                    if last_param not in free_symbols:
                        raise ValueError(
                            f"Last map parameter {last_param} must be in the memlet {memlet}, not in this case - edge: {edge}, state: {state}"
                        )


def token_replace(code: str, src: str, dst: str) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', code)

    # Replace tokens that exactly match src
    tokens = [dst if token.strip() == src else token for token in tokens]

    # Recombine everything
    return ''.join(tokens).strip()


def check_nsdfg_connector_array_shapes_match(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    for ie in parent_state.in_edges(nsdfg_node):
        if ie.data is not None:
            subset = ie.data.subset
            dst_arr_name = ie.dst_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b) // s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b) // s for (b, e, s) in subset if (e + 1 - b) // s != 1])
            assert dst_arr.shape == dst_shape_1 or dst_arr.shape == dst_shape_2 or dst_arr.shape == dst_shape_1_collapsed or dst_arr.shape == dst_shape_2_collapsed, \
                f"Shape mismatch for in-edge connector '{dst_arr_name}': dst_arr.shape={dst_arr.shape}, expected {dst_shape_1} or {dst_shape_2} or {dst_shape_1_collapsed} or {dst_shape_2_collapsed}"
    for oe in parent_state.out_edges(nsdfg_node):
        if oe.data is not None:
            subset = oe.data.subset
            dst_arr_name = oe.src_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b) // s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b) // s for (b, e, s) in subset if (e + 1 - b) // s != 1])
            assert dst_arr.shape == dst_shape_1 or dst_arr.shape == dst_shape_2 or dst_arr.shape == dst_shape_1_collapsed or dst_arr.shape == dst_shape_2_collapsed, \
                f"Shape mismatch for out-edge connector '{dst_arr_name}': dst_arr.shape={dst_arr.shape}, expected {dst_shape_1} or {dst_shape_2} or {dst_shape_1_collapsed} or {dst_shape_2_collapsed}"


def fix_nsdfg_connector_array_shapes_mismatch(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    for ie in parent_state.in_edges(nsdfg_node):
        if ie.data is not None:
            subset = ie.data.subset
            dst_arr_name = ie.dst_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b) // s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b) // s for (b, e, s) in subset if (e + 1 - b) // s != 1])
            if dst_arr.shape != dst_shape_1 and dst_arr.shape != dst_shape_2 and dst_arr.shape != dst_shape_1_collapsed and dst_arr.shape != dst_shape_2_collapsed:
                nsdfg_node.sdfg.remove_data(dst_arr_name, validate=False)
                nsdfg_node.sdfg.add_array(name=dst_arr_name,
                                          shape=dst_shape_1_collapsed,
                                          storage=dst_arr.storage,
                                          dtype=dst_arr.dtype,
                                          location=dst_arr.location,
                                          transient=False,
                                          lifetime=dst_arr.lifetime,
                                          debuginfo=dst_arr.debuginfo,
                                          allow_conflicts=dst_arr.allow_conflicts,
                                          find_new_name=False,
                                          alignment=dst_arr.alignment,
                                          may_alias=False)

    for oe in parent_state.out_edges(nsdfg_node):
        if oe.data is not None:
            subset = oe.data.subset
            dst_arr_name = oe.src_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b) // s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b) // s for (b, e, s) in subset if (e + 1 - b) // s != 1])
            if dst_arr.shape != dst_shape_1 and dst_arr.shape != dst_shape_2 and dst_arr.shape != dst_shape_1_collapsed and dst_arr.shape != dst_shape_2_collapsed:
                nsdfg_node.sdfg.remove_data(dst_arr_name, validate=False)
                nsdfg_node.sdfg.add_array(name=dst_arr_name,
                                          shape=dst_shape_1_collapsed,
                                          storage=dst_arr.storage,
                                          dtype=dst_arr.dtype,
                                          location=dst_arr.location,
                                          transient=False,
                                          lifetime=dst_arr.lifetime,
                                          debuginfo=dst_arr.debuginfo,
                                          allow_conflicts=dst_arr.allow_conflicts,
                                          find_new_name=False,
                                          alignment=dst_arr.alignment,
                                          may_alias=False)


def extract_non_connector_syms_from_tasklet(node: dace.nodes.Tasklet) -> Set[str]:
    assert isinstance(node, dace.nodes.Tasklet)
    assert node.code.language == dace.dtypes.Language.Python
    connectors = {str(s) for s in set(node.in_connectors.keys()).union(set(node.out_connectors.keys()))}
    code_rhs: str = node.code.as_string.split("=")[-1].strip()
    all_syms = {str(s) for s in dace.symbolic.SymExpr(code_rhs).free_symbols}
    real_free_syms = all_syms - connectors
    free_non_connector_syms = {str(s) for s in real_free_syms}
    return free_non_connector_syms


class TaskletType(Enum):
    ARRAY_ARRAY_ASSIGNMENT = "array_array_assignment"  # a = b
    ARRAY_SYMBOL_ASSIGNMENT = "array_symbol_assignment"  # a = sym
    SCALAR_SYMBOL = "scalar_symbol"  # a = scalar op sym
    ARRAY_SYMBOL = "array_symbol"  # a = array op sym
    ARRAY_SCALAR = "array_scalar"  # a = array op scalar
    ARRAY_ARRAY = "array_array"  # a = array1 op array2
    SCALAR_SCALAR = "scalar_scalar"  # a = scalar1 op scalar2
    SYMBOL_SYMBOL = "symbol_symbol"  # a = symbol1 op symbol2


def classify_tasklet(state: dace.SDFGState, node: dace.nodes.Tasklet) -> Dict:
    """
    Inspect `node` and `state` and return a dict describing the
    tasklet type (TaskletType) and metadata needed for code instantiation.
    """
    in_conns = list(node.in_connectors.keys())
    out_conns = list(node.out_connectors.keys())
    n_in = len(in_conns)
    n_out = len(out_conns)

    assert n_out <= 1, "Only support tasklets with at most 1 output in this pass"
    lhs = next(iter(node.out_connectors.keys())) if n_out == 1 else None

    assert isinstance(node, dace.nodes.Tasklet)
    code: CodeBlock = node.code
    assert code.language == dace.dtypes.Language.Python
    code_str: str = code.as_string

    # Try extract constant (used for array-constant templates)
    info_dict = {"type": None, "lhs": lhs, "rhs1": None, "rhs2": None, "constant1": None, "constant2": None, "op": None}

    assert n_out == 1

    # Single-input + single-output: assignment, scalar-symbol, array-symbol
    if n_in == 1:
        rhs = in_conns[0]
        # assignment case: a = b
        # find connected data description for rhs
        in_edges = {ie for ie in state.in_edges_by_connector(node, rhs)}
        assert len(in_edges) == 1, f"expected 1 in-edge for connector {rhs}, found {len(in_edges)}"
        rhs_data_name = in_edges.pop().data.data
        rhs_data = state.sdfg.arrays[rhs_data_name]
        out_edges = {oe for oe in state.out_edges_by_connector(node, lhs)}
        assert len(out_edges) == 1, f"expected 1 out-edge for connector {lhs}, found {len(out_edges)}"
        lhs_data_name = out_edges.pop().data.data
        lhs_data = state.sdfg.arrays[lhs_data_name]

        if code_str == f"{lhs} = {rhs}" or code_str == f"{lhs} = {rhs};":
            state.sdfg.save("x.sdfg")
            assert isinstance(lhs_data,
                              dace.data.Array), f"Expected lhs_data to be array but is: {lhs_data}, ({type(lhs_data)})"
            assert isinstance(rhs_data,
                              dace.data.Array), f"Expected rhs_data to be array but is: {rhs_data}, ({type(rhs_data)})"
            info_dict.update({"type": TaskletType.ARRAY_ARRAY_ASSIGNMENT, "op": "=", "rhs1": rhs})
            return info_dict

        has_constant = False
        constant = None
        try:
            constant = extract_constant(code_str)
            has_constant = True
        except Exception:
            has_constant = False

        # single input with an explicit numeric constant (e.g., a = arr + 5)
        free_non_connector_syms = extract_non_connector_syms_from_tasklet(node)
        if len(free_non_connector_syms) == 1:
            has_constant = True
            constant = free_non_connector_syms.pop()

        if not has_constant:
            # Tasklet might be something like in1 * in1
            info_dict.update({
                "type": TaskletType.ARRAY_ARRAY,
                "rhs1": rhs,
                "rhs2": rhs,
                "op": extract_single_op(code_str)
            })
            return info_dict
        else:
            if isinstance(rhs_data, dace.data.Array):
                info_dict.update({
                    "type": TaskletType.ARRAY_SYMBOL,
                    "rhs1": rhs,
                    "constant1": constant,
                    "op": extract_single_op(code_str)
                })
                return info_dict
            elif isinstance(rhs_data, dace.data.Scalar):
                info_dict.update({
                    "type": TaskletType.SCALAR_SYMBOL,
                    "rhs1": rhs,
                    "constant1": constant,
                    "op": extract_single_op(code_str)
                })
                return info_dict
            else:
                raise Exception("Unhandled case in tasklet type")

    # Two-input binary ops
    elif n_in == 2:
        op = extract_single_op(code_str)
        rhs1, rhs2 = in_conns[0], in_conns[1]
        lhs = next(iter(node.out_connectors.keys()))
        scalars, arrays = get_scalar_and_array_arguments(state, node)
        assert len(scalars) + len(arrays) == 2

        if len(arrays) == 2 and len(scalars) == 0:
            info_dict.update({"type": TaskletType.ARRAY_ARRAY, "rhs1": rhs1, "rhs2": rhs2, "op": op})
            return info_dict
        elif len(scalars) == 1 and len(arrays) == 1:
            array_arg = next(iter(arrays))
            scalar_arg = next(iter(scalars))
            info_dict.update({"type": TaskletType.ARRAY_SCALAR, "rhs1": array_arg, "constant1": scalar_arg, "op": op})
            return info_dict
        elif len(scalars) == 2:
            # preserve original behavior: save and raise
            state.sdfg.save("hmm.sdfg")
            info_dict.update({"type": TaskletType.SCALAR_SCALAR})
            raise Exception(
                "Hmm, check? SDFG saved as hmm.sdfg. There should be no SCALAR_SCALAR tasklets left when vectorizing")

    # Zero-input (symbol-only) tasklets: decide by lhs storage type
    if n_in == 0:
        free_syms = extract_non_connector_syms_from_tasklet(node)
        assert len(free_syms) == 2 or len(free_syms) == 1, f"{str(free_syms)}"
        if len(free_syms) == 2:
            free_sym1 = free_syms.pop()
            free_sym2 = free_syms.pop()
            info_dict.update({
                "type": TaskletType.SYMBOL_SYMBOL,
                "constant1": free_sym1,
                "constant2": free_sym2,
                "op": extract_single_op(code_str)
            })
            return info_dict
        elif len(free_syms) == 1:
            free_sym1 = free_syms.pop()
            info_dict.update({"type": TaskletType.ARRAY_SYMBOL_ASSIGNMENT, "constant1": free_sym1, "op": "="})
            return info_dict

    raise NotImplementedError("Unhandled case in detect tasklet type")


def offset_symbol_in_expression(expr_str: str, symbol_to_offset: str, offset: int) -> str:
    expr = dace.symbolic.SymExpr(expr_str)
    sym_to_change = None
    for free_sym in expr.free_symbols:
        if str(free_sym) == symbol_to_offset:
            sym_to_change = free_sym
            break
    if sym_to_change is None:
        return expr_str
    offsetted_expr = sym_to_change + offset
    offset_expr = expr.subs(sym_to_change, offsetted_expr)
    return sympy.pycode(offset_expr)


def instantiate_tasklet_from_info(state: dace.SDFGState, node: dace.nodes.Tasklet, info: dict, vector_width: int,
                                  templates: Dict[str, str], vector_map_param: str) -> None:
    """
    Given the classification `info` returned by `classify_tasklet`, set
    `node.code` to the appropriate vectorized properties.CodeBlock.

    Expects `info` to contain keys:
      - "type" (TaskletType)
      - "lhs", "rhs1", "rhs2", "constant1", "constant2", "op"
    """
    ttype: TaskletType = info.get("type")
    lhs = info.get("lhs")
    rhs1 = info.get("rhs1")
    rhs2 = info.get("rhs2")
    c1 = info.get("constant1")
    c2 = info.get("constant2")
    op = info.get("op")
    vw = vector_width

    def _str_to_float_or_str(s: Union[int, float, str, None]):
        if s is None:
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def _get_vector_templates(rhs1: str, rhs2: str, lhs: str, constant: Union[str, None], op: str):
        if op in templates:
            if rhs2 is None:
                if constant is None:
                    new_code = templates[op].format(rhs1=rhs1, lhs=lhs, op=op, vector_width=vector_width)
                else:
                    new_code = templates[op].format(rhs1=rhs1,
                                                    constant=_str_to_float_or_str(constant),
                                                    lhs=lhs,
                                                    op=op,
                                                    vector_width=vector_width)
            else:
                new_code = templates[op].format(rhs1=rhs1, rhs2=rhs2, lhs=lhs, op=op, vector_width=vector_width)
        else:
            print(f"Operator `{op}` is not in supported ops `{set(templates.keys())}`")
            print(f"Generating fall-back scalar code")
            if op in {">", ">=", "<", "<=", "c>", "c<", "c<=", "c>=", "==", "c==", "c!=", "!="}:
                comparison_set_suffix = "? 1.0 : 0.0"
            else:
                comparison_set_suffix = ""
            if constant is not None:
                assert op.startswith("c")
                op = op[1:]
                code_str = ""
                for i in range(vector_width):
                    code_str += f"{lhs}[{i}] = ({rhs1}[{i}] {op} {constant}){comparison_set_suffix};\n"
            else:
                op = op
                code_str = ""
                for i in range(vector_width):
                    code_str += f"{lhs}[{i}] = ({rhs1}[{i}] {op} {rhs2}[{i}]){comparison_set_suffix};\n"
            new_code = code_str
        return new_code

    # Helpers
    def set_template(rhs1_, rhs2_, constant_, lhs_, op_):
        node.code = dace.properties.CodeBlock(
            code=_get_vector_templates(rhs1=rhs1_,
                                       rhs2=rhs2_,
                                       constant=_str_to_float_or_str(constant_),
                                       lhs=lhs_,
                                       op=op_),
            language=dace.Language.CPP,
        )

    # 1) ARRAY-ARRAY assignment: a = b  (both arrays)
    if ttype == TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        # use direct template with op "="
        set_template(rhs1, None, None, lhs, "=")
        return

    # 1) ARRAY-SYMBOL assignment: a = sym
    if ttype == TaskletType.ARRAY_SYMBOL_ASSIGNMENT:
        code_lines = []
        for i in range(vw):
            # If detect inner symbol used we need to correctly offset it to have + lane_id
            code_lines.append(f"{lhs}[{i}] = {c1}{i};")
        node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.CPP)
        return

    # 2) Single-input with constant: array or scalar combined with a symbol/constant
    if ttype == TaskletType.ARRAY_SYMBOL:
        # rhs1 is an array, constant1 is the symbol/constant -> use constant template (op prefixed with 'c')
        set_template(rhs1, None, c1, lhs, "c" + op)
        return

    if ttype == TaskletType.SCALAR_SYMBOL:
        # rhs1 is a scalar, constant1 is the symbol -> generate explicit per-lane assignments
        # e.g., for i in 0..vw-1: lhs[i] = rhs1 op constant1;
        code_lines = []
        for i in range(vw):
            # If detect inner symbol used we need to correctly offset it to have + lane_id
            expr_str = f"({rhs1} {op} {c1})"
            corrected_expr_str = offset_symbol_in_expression(expr_str, vector_map_param, i)
            code_lines.append(f"{lhs}[{i}] = {corrected_expr_str};")
        node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.CPP)
        return

    # 3) Two-input binary ops
    if ttype == TaskletType.ARRAY_ARRAY:
        # array op array
        set_template(rhs1, rhs2, None, lhs, op)
        return

    if ttype == TaskletType.ARRAY_SCALAR:
        # array op scalar -> use constant template (prefix op with 'c')
        # note: info stores rhs1 as the array name and constant1 as the scalar name
        set_template(rhs1, None, c1, lhs, "c" + op)
        data_edge = state.in_edges_by_connector(node, rhs1)[0]
        data_name = data_edge.data.data
        data = state.sdfg.arrays[data_name]
        if data.transient is False:
            raise Exception(
                f"Array-Scalar tasklet is not currenlty supported by auto vectorization if input scalar is non-transient. Try to re-write the kernel it happens at {node}, state:{state}"
            )
        return

    if ttype == TaskletType.SCALAR_SCALAR:
        # preserved original behavior: this was saved + raised in classifier
        # keep behavior consistent by raising here as well (classifier already saved)
        raise Exception("Unhandled: two scalar operands (SCALAR_SCALAR). See saved hmm.sdfg")

    # 4) zero-input (symbol-symbol)
    if ttype == TaskletType.SYMBOL_SYMBOL:
        # we need to decide whether lhs is an array or scalar by checking SDFG arrays
        # get out-edge for the lhs connector to obtain the data descriptor name
        out_edges = {oe for oe in state.out_edges_by_connector(node, lhs)}
        assert len(out_edges) == 1, f"expected 1 out-edge for connector {lhs}, found {len(out_edges)}"
        lhs_data_name = out_edges.pop().data.data
        lhs_data = state.sdfg.arrays[lhs_data_name]

        expr = f"{c1} {op} {c2}"
        if isinstance(lhs_data, dace.data.Array):
            # replicate across lanes
            code_lines = []
            for i in range(vw):
                # Any appereance of the vector map param `j` needs to be replaced with `j + lane_id`
                code_lines.append(f"{lhs}[{i}] = {offset_symbol_in_expression(expr, vector_map_param, i)};")
            node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.CPP)
        else:
            # scalar lhs
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr};\n", language=dace.Language.CPP)
        return

    # Fallback: unknown classification
    raise NotImplementedError(f"Unhandled TaskletType in instantiation: {ttype}")


def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode,
                     vector_width: int) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
    touched_nodes = set()
    touched_edges = set()

    ies = state.in_edges(node)
    assert len(ies) == 1
    ie = ies[0]
    src = ie.src
    assert isinstance(src, dace.nodes.Tasklet), f"Writes to sink nodes need to go through assignment tasklets, do it"
    inc = next(iter(src.in_connectors))
    outc = next(iter(src.out_connectors))
    assert src.code.as_string == f"{outc} = {inc}", f"{src.code.as_string} != {inc} = {outc}"
    src.code = CodeBlock(code="\n".join([f"{outc}[{_i}] = {inc}[{_i}]" for _i in range(vector_width)]))
    touched_nodes.add(src)
    packed_access = state.add_access(f"{node.data}_packed")
    touched_nodes.add(packed_access)
    state.remove_edge(ie)
    if isinstance(ie, dace.nodes.Node):
        assert False
    touched_edges.add(ie)
    if f"{node.data}_packed" not in state.sdfg.arrays:
        dst_arr = state.sdfg.arrays[node.data]
        state.sdfg.add_array(name=f"{node.data}_packed",
                             shape=(vector_width, ),
                             storage=dst_arr.storage,
                             dtype=dst_arr.dtype,
                             location=dst_arr.location,
                             transient=True,
                             lifetime=dst_arr.lifetime,
                             debuginfo=dst_arr.debuginfo,
                             allow_conflicts=dst_arr.allow_conflicts,
                             find_new_name=False,
                             alignment=dst_arr.alignment,
                             may_alias=False)
    e = state.add_edge(ie.src, ie.src_conn, packed_access, None,
                       dace.memlet.Memlet(f"{node.data}_packed[0:{vector_width}]"))
    if isinstance(e, dace.nodes.Node):
        assert False
    touched_edges.add(e)

    for i in range(vector_width):
        t = state.add_tasklet(name=f"a_{i}", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
        touched_nodes.add(t)
        t.add_in_connector("_in")
        t.add_out_connector("_out")
        e1 = state.add_edge(
            packed_access, None, t, "_in",
            dace.memlet.Memlet(data=node.data + "_packed", subset=dace.subsets.Range([(str(i), str(i), 1)])))
        if isinstance(e1, dace.nodes.Node):
            assert False
        touched_edges.add(e1)

        new_subset = repl_subset_to_symbol_offset(ie.data.subset, str(i))

        e2 = state.add_edge(t, "_out", ie.dst, None, dace.memlet.Memlet(data=node.data, subset=new_subset))
        if isinstance(e2, dace.nodes.Node):
            assert False
        touched_edges.add(e2)

    return touched_nodes, touched_edges


def add_symbol_with_value(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG,
                          symbol_names_and_values: Dict[str, str]) -> None:
    inner_sdfg = nsdfg.sdfg
    for symbol_name in symbol_names_and_values:
        inner_sdfg.add_symbol(name=symbol_name, find_new_name=False, stype=dace.int64)
    inner_sdfg.add_state_before(inner_sdfg.start_block, assignments=symbol_names_and_values)

    # Get all newly needed symbols from rhs
    for v in symbol_names_and_values.values():
        v_symexpr_free_syms = dace.symbolic.SymExpr(v).free_symbols
        for free_sym in v_symexpr_free_syms:
            free_sym_str = str(free_sym)
            if free_sym_str in inner_sdfg.symbols:
                continue
            symbols_defined_at = state.symbols_defined_at(nsdfg)
            assert free_sym_str in symbols_defined_at, f"{free_sym_str} of {v} is not in symbols defined at the scope: {symbols_defined_at}."
            inner_sdfg.add_symbol(free_sym_str, symbols_defined_at[free_sym_str], False)
            nsdfg.symbol_mapping[free_sym_str] = free_sym_str


def replace_arrays_with_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any]) -> None:
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(name=arr_name,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)


def copy_arrays_with_a_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                 name_suffix: str) -> None:
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.add_array(name=arr_name + name_suffix,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)


def get_scalar_source_nodes(sdfg: dace.SDFG,
                            non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    source_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.in_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, )):
                    if non_transient_only is False or arr.transient is False:
                        source_nodes.append((state, node))
    return source_nodes


def get_scalar_sink_nodes(sdfg: dace.SDFG,
                          non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    sink_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.out_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or isinstance(arr, dace.data.Array) and arr.shape == (1, ):
                    if non_transient_only is False or arr.transient is False:
                        sink_nodes.append((state, node))
    return sink_nodes


def add_transient_arrays_from_list(sdfg: dace.SDFG, arr_name_shape_storage_dtype: Iterable[Tuple[str, Any, Any,
                                                                                                 Any]]) -> None:
    for arr_name, shape, storage, dtype in arr_name_shape_storage_dtype:
        sdfg.add_array(
            name=arr_name,
            shape=shape,
            storage=storage,
            dtype=dtype,
            transient=True,
            find_new_name=False,
        )


def is_assignment_tasklet(node: dace.nodes.Tasklet) -> bool:
    if (len(node.in_connectors) == 1 and len(node.out_connectors) == 1):
        in_conn = next(iter(node.in_connectors.keys()))
        out_conn = next(iter(node.out_connectors.keys()))

        return (node.code.as_string == f"{out_conn} = {in_conn}" or node.code.as_string == f"{out_conn} = {in_conn};")
    return False


def check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg: dace.SDFG,
                                                                scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                                                                              dace.nodes.AccessNode]]):
    for state, sink_node in scalar_sink_nodes:
        in_edges = state.in_edges(sink_node)
        if len(in_edges) != "1":
            raise Exception("All scalar sink nodes should have at max 1 incoming edge")
        in_edge = in_edges[0]
        src = in_edge.src
        if not (isinstance(src, dace.nodes.Tasklet) and is_assignment_tasklet(src)):
            raise Exception("All write to scalar should happen through an assignment tasklet")


def only_one_flop_after_source(state: dace.SDFGState, node: dace.nodes.AccessNode):
    nodes_to_check = [node]
    tasklets_with_flops = 0
    checked_nodes = []

    while nodes_to_check:
        cur_node = nodes_to_check.pop(0)
        checked_nodes.append(cur_node)
        if isinstance(cur_node, dace.nodes.Tasklet) and not is_assignment_tasklet(cur_node):
            tasklets_with_flops += 1
        nodes_to_check += [e.dst for e in state.out_edges(cur_node)]
        if tasklets_with_flops > 1:
            return False, []

    return tasklets_with_flops <= 1, checked_nodes


def input_is_zero_and_transient_accumulator(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG,
                                            inner_state: dace.SDFGState, source_node: dace.nodes.AccessNode,
                                            sink_node: dace.nodes.AccessNode):
    # Make sure the data of in and out edges refer to the same name
    sink_data = sink_node.data
    source_data = source_node.data
    sink_connector = nsdfg.out_connectors[sink_data]
    source_connector = nsdfg.in_connectors[source_data]
    sink_edges = state.out_edges_by_connector(nsdfg, sink_data)
    source_edges = state.in_edges_by_connector(nsdfg, source_data)

    out_source_datas = {ie.data.data for ie in source_edges if ie.data is not None}
    out_sink_datas = {oe.data.data for oe in sink_edges if oe.data is not None}
    if len(out_sink_datas) != 1:
        return False, ""
    if len(out_source_datas) != 1:
        return False, ""
    out_sink_data = out_sink_datas.pop()
    out_source_data = out_source_datas.pop()

    if out_source_data != out_sink_data:
        return False, ""

    # Find the first access node of the source node outside
    source_edges = list(state.in_edges_by_connector(nsdfg, source_data))
    assert len(source_edges) == 1, f"{source_edges} for in connector {source_data} of {nsdfg}"
    source_edge = source_edges[0]
    mpath = state.memlet_path(source_edge)
    src_acc_node = mpath[0].src
    if not isinstance(src_acc_node, dace.nodes.AccessNode):
        print(f"{src_acc_node} of the memlet path {mpath} is not an access node")
        return False, ""

    # Ensure the access node directly connects to a memset-0 tasklet
    if state.in_degree(src_acc_node) != 1:
        print(f"In degree of {src_acc_node} not one")
        return False, ""

    in_tasklet = state.in_edges(src_acc_node)[0].src
    if not isinstance(in_tasklet, dace.nodes.Tasklet):
        print(f"In neighbor {in_tasklet} is not a tasklet")
        return False, ""

    code_str = in_tasklet.code.as_string
    if len(in_tasklet.out_connectors) != 1:
        return False, ""
    out_conn = next(iter(in_tasklet.out_connectors))
    if not (code_str.strip() != f"{out_conn} = 0" or code_str.strip() != f"{out_conn} = 0;"):
        return False, ""

    # If all true return true and accumulator name
    return True, src_acc_node.data


def replace_all_access_subsets(state: dace.SDFGState, name: str, new_subset_expr: str):
    for edge in state.edges():
        if edge.data is not None and edge.data.data == name:
            nm = dace.memlet.Memlet(expr=f"{name}[{new_subset_expr}]")
            edge.data = nm


def expand_assignment_tasklets(state: dace.SDFGState, name: str, vector_length: int):
    for e in state.edges():
        if (isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == name and isinstance(e.src, dace.nodes.Tasklet)):
            code = e.src.code
            in_conns = e.src.in_connectors
            out_conns = e.src.out_connectors
            if len(in_conns) != 0:
                assert False, "Non-assignemnt taskelt found for accumulator, unsupported case"
            assert len(out_conns) == 1, f"{out_conns}"
            out_conn = next(iter(out_conns))
            assert code.language == dace.dtypes.Language.Python
            assert code.as_string.startswith(f"{out_conn} =")
            rhs = code.as_string.split("=")[-1].strip()
            ncode_str = "\n".join([f"{out_conn}[{i}] = {rhs}" for i in range(vector_length)])
            e.src.code = dace.properties.CodeBlock(ncode_str)


def reduce_before_use(state: dace.SDFGState, name: str, vector_width: int, op: str):
    # Any time a tasklet reads name[0:vector_length] then we need to reduce it before
    # In a reduction tasklet
    for edge in state.edges():
        dst = edge.dst
        src = edge.src
        if isinstance(dst, dace.nodes.Tasklet) and edge.data is not None and edge.data.data == name:
            arr = state.sdfg.arrays[name]
            state.sdfg.add_scalar(name=name + "_scl",
                                  dtype=arr.dtype,
                                  storage=arr.storage,
                                  transient=True,
                                  lifetime=arr.lifetime)
            an = state.add_access(name + "_scl")
            t = state.add_tasklet(name=f"scalarize_{name}",
                                  inputs={"_in"},
                                  outputs={"_out"},
                                  code="_out =" + f" {op} ".join([f"_in[{i}]" for i in range(vector_width)]))
            t.add_in_connector("_in")
            t.add_out_connector("_out")
            state.add_edge(src, None, t, "_in", copy.deepcopy(edge.data))
            state.add_edge(t, "_out", an, None, dace.memlet.Memlet(f"{name}_scl[0]"))
            state.add_edge(an, None, edge.dst, edge.dst_conn, dace.memlet.Memlet(f"{name}_scl[0]"))

            state.remove_edge(edge)


def move_out_reduction(scalar_source_nodes, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, inner_sdfg: dace.SDFG,
                       vector_width):
    num_flops, node_path = only_one_flop_after_source(scalar_source_nodes[0][0], scalar_source_nodes[0][1])
    is_inout_accumulator, accumulator_name = input_is_zero_and_transient_accumulator(
        state, nsdfg, scalar_source_nodes[0][0], scalar_source_nodes[0][1], node_path[-1])
    op = extract_single_op(node_path[1].code.as_string)
    print(is_inout_accumulator, num_flops, accumulator_name)
    if num_flops <= 1 and is_inout_accumulator:
        source_data = scalar_source_nodes[0][1].data
        sink_data = node_path[-1].data
        print("Source data", source_data, "Sink data", sink_data)
        replace_arrays_with_new_shape(inner_sdfg, {source_data, sink_data}, (vector_width, ))
        replace_arrays_with_new_shape(state.sdfg, {accumulator_name}, (vector_width, ))
        replace_all_access_subsets(state, accumulator_name, f"0:{vector_width}")
        expand_assignment_tasklets(state, accumulator_name, vector_width)
        reduce_before_use(state, accumulator_name, vector_width, op)
