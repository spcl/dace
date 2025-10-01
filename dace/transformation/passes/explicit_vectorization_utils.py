# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import sympy
import dace
import ast
from typing import Dict, Iterable, Set, Tuple, Union
from dace import SDFGState
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from enum import Enum

def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    # Convert string keys to dace symbols
    sym_repl_dict = {
        dace.symbolic.symbol(k) if isinstance(k, str) else k: v 
        for k, v in repl_dict.items()
    }
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

def replace_memlet_expression(state: SDFGState,
                              edges: Iterable[Edge[Memlet]],
                              old_subset_expr: dace.subsets.Range,
                              new_subset_expr: dace.subsets.Range,
                              repl_scalars_with_arrays: bool) -> Set[str]:
    arr_dim = [((e + 1 - b) // s) for (b, e, s) in new_subset_expr]

    for edge in edges:
        src_node : dace.nodes.Node = edge.src
        dst_node : dace.nodes.Node = edge.dst

        if edge.data is not None and edge.data.subset == old_subset_expr:
            if repl_scalars_with_arrays:
                for data_node in [src_node, dst_node]:
                    if isinstance(data_node, dace.nodes.AccessNode):
                        arr = state.sdfg.arrays[data_node.data]
                        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1,)):
                            state.sdfg.remove_data(data_node.data, validate=False)
                            state.sdfg.add_array(
                                name=data_node.data,
                                shape=tuple(arr_dim),
                                dtype=arr.dtype,
                                storage=arr.storage,
                                location=arr.location,
                                transient=arr.transient,
                                lifetime=arr.lifetime
                            )
            edge.data = dace.memlet.Memlet(
                data=edge.data.data,
                subset=copy.deepcopy(new_subset_expr)
            )

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
            assert (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)) or (len({_n for _n in all_nodes if isinstance(_n, dace.nodes.NestedSDFG)}) == 0)

def to_ints(sym_epxr: dace.symbolic.SymExpr):
    try:
        return int(sym_epxr)
    except:
        return None

def add_copies_before_and_after_nsdfg(state: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                      exact_volumes_per_dimension: Tuple[int],
                                      copy_to_storage: dace.dtypes.StorageType):
    for ie in state.in_edges(nsdfg_node):
        if ie.data is not None:
            in_data_name : str = ie.data.data
            nsdfg_data_name : str = ie.dst_conn
            subset_lengths = tuple([(e + 1 - b)//s for b,e,s in ie.data.subset])
            if (
                (exact_volumes_per_dimension is None and all(to_ints(sl) is not None for sl in subset_lengths)) or
                (exact_volumes_per_dimension is not None and subset_lengths == exact_volumes_per_dimension)
                ):
                # Insert copy
                orig_arr = state.sdfg.arrays[in_data_name]
                if f"{in_data_name}_vec" not in state.sdfg.arrays:
                    arr_name, arr = state.sdfg.add_array(
                        name=f"{in_data_name}_vec",
                        shape=subset_lengths,
                        dtype=orig_arr.dtype,
                        location=orig_arr.location,
                        transient=True,
                        find_new_name=False,
                        storage=copy_to_storage
                    )
                else:
                    arr_name = f"{in_data_name}_vec"
                    assert state.sdfg.arrays[arr_name].storage == copy_to_storage
                arr_access = state.add_access(arr_name)
                state.remove_edge(ie)
                state.add_edge(
                    ie.src,
                    ie.src_conn,
                    arr_access,
                    None,
                    copy.deepcopy(ie.data)
                )
                full_subset_str = ", ".join([f"{0}:{sl}" for sl in subset_lengths])
                state.add_edge(
                    arr_access,
                    None,
                    nsdfg_node,
                    nsdfg_data_name,
                    dace.memlet.Memlet(f"{arr_name}[{full_subset_str}]")
                )
    for oe in state.out_edges(nsdfg_node):
        if oe.data is not None:
            out_data_name: str = oe.data.data
            nsdfg_data_name: str = oe.src_conn
            subset_lengths = tuple([(e + 1 - b)//s for b, e, s in oe.data.subset])
            if (
                (exact_volumes_per_dimension is None and all(to_ints(sl) is not None for sl in subset_lengths)) or
                (exact_volumes_per_dimension is not None and subset_lengths == exact_volumes_per_dimension)
            ):
                # Insert copy
                orig_arr = state.sdfg.arrays[out_data_name]
                if f"{out_data_name}_vec" not in state.sdfg.arrays:
                    arr_name, arr = state.sdfg.add_array(
                        name=f"{out_data_name}_vec",
                        shape=subset_lengths,
                        dtype=orig_arr.dtype,
                        location=orig_arr.location,
                        transient=True,
                        find_new_name=False,
                        storage=copy_to_storage
                    )
                else:
                    arr_name = f"{out_data_name}_vec"
                    assert state.sdfg.arrays[arr_name].storage == copy_to_storage
                arr_access = state.add_access(arr_name)
                state.remove_edge(oe)
                full_subset_str = ", ".join([f"{0}:{sl}" for sl in subset_lengths])
                state.add_edge(
                    nsdfg_node,
                    nsdfg_data_name,
                    arr_access,
                    None,
                    dace.memlet.Memlet(f"{arr_name}[{full_subset_str}]")
                )
                state.add_edge(
                    arr_access,
                    None,
                    oe.dst,
                    oe.dst_conn,
                    copy.deepcopy(oe.data)
                )

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
                    raise ValueError(
                        f"All nodes must be within a map, found node {node} outside of any map in state {state}.")
                else:
                    continue
            else:
                if not isinstance(map_entry, dace.nodes.MapEntry):
                    raise ValueError(
                        f"Parent scope of node {node} is not a map, found {map_entry} in state {state}.")
                assert map_entry is not None
                checked_map_entries.add(map_entry)

            # If we have checked a map entry (and nodes within its body) then skip it
            if map_entry not in checked_map_entries:
                assert isinstance(map_entry, dace.nodes.MapEntry
                                    ), f"Parent scope of node {node} is not a map, returned value is {map_entry}."
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

def split_in_token(expr_str: str, src: str, dst: str) -> str:
    tokens = expr_str.split()
    ntokens = []
    for token in tokens:
        if token == src:
            ntokens.append(dst)
        else:
            ntokens.append(token)
    return " ".join(ntokens)

def check_nsdfg_connector_array_shapes_match(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    for ie in parent_state.in_edges(nsdfg_node):
        if ie.data is not None:
            subset = ie.data.subset
            dst_arr_name = ie.dst_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b)//s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b)//s for (b, e, s) in subset if (e + 1 - b)//s != 1])
            assert dst_arr.shape == dst_shape_1 or dst_arr.shape == dst_shape_2 or dst_arr.shape == dst_shape_1_collapsed or dst_arr.shape == dst_shape_2_collapsed, \
                f"Shape mismatch for in-edge connector '{dst_arr_name}': dst_arr.shape={dst_arr.shape}, expected {dst_shape_1} or {dst_shape_2} or {dst_shape_1_collapsed} or {dst_shape_2_collapsed}"
    for oe in parent_state.out_edges(nsdfg_node):
        if oe.data is not None:
            subset = oe.data.subset
            dst_arr_name = oe.src_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b)//s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b)//s for (b, e, s) in subset if (e + 1 - b)//s != 1])
            assert dst_arr.shape == dst_shape_1 or dst_arr.shape == dst_shape_2 or dst_arr.shape == dst_shape_1_collapsed or dst_arr.shape == dst_shape_2_collapsed, \
                f"Shape mismatch for out-edge connector '{dst_arr_name}': dst_arr.shape={dst_arr.shape}, expected {dst_shape_1} or {dst_shape_2} or {dst_shape_1_collapsed} or {dst_shape_2_collapsed}"

def fix_nsdfg_connector_array_shapes_mismatch(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    for ie in parent_state.in_edges(nsdfg_node):
        if ie.data is not None:
            subset = ie.data.subset
            dst_arr_name = ie.dst_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b)//s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b)//s for (b, e, s) in subset if (e + 1 - b)//s != 1])
            if dst_arr.shape != dst_shape_1 and dst_arr.shape != dst_shape_2 and dst_arr.shape != dst_shape_1_collapsed and dst_arr.shape != dst_shape_2_collapsed :
                nsdfg_node.sdfg.remove_data(dst_arr_name, validate=False)
                nsdfg_node.sdfg.add_array(
                    name=dst_arr_name,
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
                    may_alias=False
                )

    for oe in parent_state.out_edges(nsdfg_node):
        if oe.data is not None:
            subset = oe.data.subset
            dst_arr_name = oe.src_conn
            dst_arr = nsdfg_node.sdfg.arrays[dst_arr_name]
            dst_shape_1 = tuple([(e + 1 - b) for (b, e, s) in subset])
            dst_shape_2 = tuple([(e + 1 - b)//s for (b, e, s) in subset])
            dst_shape_1_collapsed = tuple([(e + 1 - b) for (b, e, s) in subset if (e + 1 - b) != 1])
            dst_shape_2_collapsed = tuple([(e + 1 - b)//s for (b, e, s) in subset if (e + 1 - b)//s != 1])
            if dst_arr.shape != dst_shape_1 and dst_arr.shape != dst_shape_2 and dst_arr.shape != dst_shape_1_collapsed and dst_arr.shape != dst_shape_2_collapsed :
                nsdfg_node.sdfg.remove_data(dst_arr_name, validate=False)
                nsdfg_node.sdfg.add_array(
                    name=dst_arr_name,
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
                    may_alias=False
                )

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
    info_dict = {
        "type": None,
        "lhs": lhs,
        "rhs1": None,
        "rhs2": None,
        "constant1": None,
        "constant2": None,
        "op": None
    }

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
            assert isinstance(lhs_data, dace.data.Array)
            assert isinstance(rhs_data, dace.data.Array)
            info_dict.update({
                "type": TaskletType.ARRAY_ARRAY_ASSIGNMENT,
                "op": "=",
                "rhs1": rhs
            })
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

        assert has_constant, f"{has_constant} is False for {node.code.as_string} ({free_non_connector_syms})"

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
            info_dict.update({
                "type": TaskletType.ARRAY_ARRAY,
                "rhs1": rhs1,
                "rhs2": rhs2,
                "op": op
            })
            return info_dict
        elif len(scalars) == 1 and len(arrays) == 1:
            array_arg = next(iter(arrays))
            scalar_arg = next(iter(scalars))
            info_dict.update({
                "type": TaskletType.ARRAY_SCALAR,
                "rhs1": array_arg,
                "constant1": scalar_arg,
                "op": op
            })
            return info_dict
        elif len(scalars) == 2:
            # preserve original behavior: save and raise
            state.sdfg.save("hmm.sdfg")
            info_dict.update({"type": TaskletType.SCALAR_SCALAR})
            raise Exception("Hmm, check? SDFG saved as hmm.sdfg. There should be no SCALAR_SCALAR tasklets left when vectorizing")

    # Zero-input (symbol-only) tasklets: decide by lhs storage type
    if n_in == 0:
        free_syms = extract_non_connector_syms_from_tasklet(node)
        assert len(free_syms) == 2
        free_sym1, free_sym2 = free_syms[0:2]
        info_dict.update({
            "type": TaskletType.SYMBOL_SYMBOL,
            "constant1": free_sym1,
            "constant2": free_sym2,
            "op": extract_single_op(code_str)
        })
        return info_dict

    raise NotImplementedError("Unhandled case in detect tasklet type")

def instantiate_tasklet_from_info(state: dace.SDFGState,
                                  node: dace.nodes.Tasklet, info: dict,
                                  vector_width: int,
                                  templates: Dict[str, str]) -> None:
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

    def _get_vector_templates(rhs1: str, rhs2: str, lhs: str, constant: Union[str, None], op: str):
        if rhs2 is None:
            if constant is None:
                new_code = templates[op].format(rhs1=rhs1, lhs=lhs, op=op, vector_width=vector_width)
            else:
                new_code = templates[op].format(rhs1=rhs1,
                                                     constant=constant,
                                                     lhs=lhs,
                                                     op=op,
                                                     vector_width=vector_width)
        else:
            new_code = templates[op].format(rhs1=rhs1, rhs2=rhs2, lhs=lhs, op=op, vector_width=vector_width)
        return new_code

    # Helpers
    def set_template(rhs1_, rhs2_, constant_, lhs_, op_):
        node.code = dace.properties.CodeBlock(
            code=_get_vector_templates(rhs1=rhs1_, rhs2=rhs2_, constant=constant_, lhs=lhs_, op=op_),
            language=dace.Language.CPP,
        )

    # 1) ARRAY-ARRAY assignment: a = b  (both arrays)
    if ttype == TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        # use direct template with op "="
        set_template(rhs1, None, None, lhs, "=")
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
            code_lines.append(f"{lhs}[{i}] = {rhs1} {op} {c1};")
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
                code_lines.append(f"{lhs}[{i}] = {expr};")
            node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.CPP)
        else:
            # scalar lhs
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr};\n", language=dace.Language.CPP)
        return

    # Fallback: unknown classification
    raise NotImplementedError(f"Unhandled TaskletType in instantiation: {ttype}")

def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode, vector_width: int) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
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
    src.code = CodeBlock(
        code="\n".join([f"{outc}[{_i}] = {inc}[{_i}]" for _i in range(vector_width)])
    )
    touched_nodes.add(src)
    packed_access = state.add_access(f"{node.data}_packed")
    touched_nodes.add(packed_access)
    state.remove_edge(ie)
    if f"{node.data}_packed" not in state.sdfg.arrays:
        dst_arr = state.sdfg.arrays[node.data]
        state.sdfg.add_array(
            name=f"{node.data}_packed",
            shape=(vector_width,),
            storage=dst_arr.storage,
            dtype=dst_arr.dtype,
            location=dst_arr.location,
            transient=False,
            lifetime=dst_arr.lifetime,
            debuginfo=dst_arr.debuginfo,
            allow_conflicts=dst_arr.allow_conflicts,
            find_new_name=False,
            alignment=dst_arr.alignment,
            may_alias=False
        )
    e = state.add_edge(ie.src, ie.src_conn, packed_access, None, dace.memlet.Memlet(f"{node.data}_packed[0:{vector_width}]"))
    touched_edges.add(e)

    for i in range(vector_width):
        t = state.add_tasklet(
            name=f"a{i}",
            inputs={"_in"},
            outputs={"_out"},
            code="_out = _in"
        )
        touched_nodes.add(t)
        t.add_in_connector("_in")
        t.add_out_connector("_out")
        e = state.add_edge(packed_access, None, t, "_in", dace.memlet.Memlet(f"{node.data}_packed[{i}]"))
        touched_edges.add(e)

        new_subset = repl_subset_to_symbol_offset(ie.data.subset, str(i))

        e = state.add_edge(t, "_out", ie.dst, None, dace.memlet.Memlet(data=node.data, subset=new_subset))
        touched_edges.add(e)

    return touched_nodes, touched_edges