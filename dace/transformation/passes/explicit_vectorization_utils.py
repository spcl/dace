# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import ast
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from dace import SDFGState
from dace.memlet import Memlet
from dace.sdfg.graph import Edge


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