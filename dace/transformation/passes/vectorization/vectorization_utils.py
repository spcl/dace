# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import typing
import sympy
import dace
import ast
import math
from typing import Dict, Iterable, Set, Tuple, Union
from dace import SDFGState, typeclass
from dace import Any
from dace import List
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace.sdfg.state import ConditionalBlock, LoopRegion
import dace.sdfg.tasklet_utils as tutil
import dace.sdfg.construction_utils as cutil
import dace.sdfg.utils as sdutil
from dace.symbolic import DaceSympyPrinter


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    """ Convenience wrapper to make the .replace not in-place """
    new_subset = copy.deepcopy(subset)
    new_subset.replace(repl_dict)
    return new_subset


def repl_subset_to_use_laneid_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbol_offset: str,
                                     vector_map_param: str) -> dace.subsets.Range:
    """
    Apply a symbolic offset to all free symbols in a subset.

    This function replaces each free symbol in the subset with a new symbol
    that has the offset appended to its name (e.g., 'i' becomes 'i_{offset}' where offset is an integer).
    New symbols are automatically added to the SDFG if they don't exist.

    If symbol is vector map param always add + 1 instead of laneid

    Args:
        sdfg: The SDFG containing the subset
        subset: The subset whose symbols should be offset
        symbol_offset: String to append to each symbol name (should be an integer)
        add_missing_symbols: If True, adds symbol mappings and assignments for
                           free symbols in the parent SDFG

    Returns:
        A new subset with offset symbols applied

    Example:
        If subset contains symbol 'i' and symbol_offset is '_v':
        - 'i' becomes 'i_v'
        - Symbol 'i_v' is added to SDFG if not present
    """
    # Offset needs to be positive integer
    assert symbol_offset.isdigit()
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols
    #print(f"Symbol offset: {symbol_offset}")
    #print("Free symbols in subset:", free_syms)

    repl_dict = {
        str(free_sym):
        str(free_sym) + "_laneid_" + str(symbol_offset) if str(free_sym) != vector_map_param else "(" + str(free_sym) +
        " + " + str(symbol_offset) + ")"
        for free_sym in free_syms
    }

    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        if str(free_sym) != vector_map_param:
            offset_symbol_name = str(free_sym) + "_laneid_" + str(symbol_offset)
            if offset_symbol_name not in sdfg.symbols:
                sdfg.add_symbol(offset_symbol_name, stype)
    #print("Generated replacement dictionary with offset:", repl_dict)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    #print("Subset before symbol offset replacement:", subset)
    #print("Subset after symbol offset replacement:", new_subset)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols - prev_sdfg_free_syms:
            #print(free_sym, "is in free symbols of the sdfg")
            raise Exception(
                "`repl_subset_to_use_laneid_offset` has introduced new free symbols (this will cause problems as the new symbols should not be free). This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue"
            )
    return new_subset


def repl_subset_to_use_with_int_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbols_to_offset: Set[str],
                                       int_offset: int) -> dace.subsets.Range:
    """
    Apply a int offset to all free symbols appearing on `symbols_to_offset` in a subset.

    that has the offset appended to its name (e.g., 'i' becomes 'i + {int_offset}' where offset is an integer).
    No new symbol is added
    """
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols
    new_range_list = []
    repl_dict = {str(free_sym): "(" + str(free_sym) + " + " + str(int_offset) + ")" for free_sym in symbols_to_offset}
    for (b, e, s) in subset:
        #print(b.free_symbols, e.free_symbols, s.free_symbols)
        if hasattr(b, "subs"):
            nb = b.subs(repl_dict)
        else:
            nb = b
        if hasattr(e, "subs"):
            ne = e.subs(repl_dict)
        else:
            ne = e
        ns = 1
        new_range_list.append((nb, ne, ns))

    new_subset = dace.subsets.Range(new_range_list)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols - prev_sdfg_free_syms:
            raise Exception(
                "`repl_subset_to_use_with_int_offset` has introduced new free symbols (this will cause problems as the new symbols should not be free). This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue"
            )

    return new_subset


def replace_memlet_expression(state: SDFGState,
                              edges: Iterable[Edge[Memlet]],
                              old_subset_expr: dace.subsets.Range,
                              new_subset_expr: dace.subsets.Range,
                              repl_scalars_with_arrays: bool,
                              edges_to_skip: Set[Edge[Memlet]],
                              vector_numeric_type: typeclass,
                              dataname: Union[str, None] = None) -> Set[str]:
    """
    Replace memlet subsets matching a pattern with a new subset expression.

    Optionally converts scalar/size-1 arrays to arrays that match the new_subset_expr's sizes
    using the `vector_numeric_type` as dtype to accommodate the new subset dimensions.

    Args:
        state: The SDFG state containing the edges
        edges: Edges whose memlets should be checked and potentially replaced
        old_subset: The subset pattern to match
        new_subset: The replacement subset
        convert_scalars_to_arrays: If True, converts Scalar/size-1 Array nodes
                                  to proper Arrays with shape matching new_subset
        edges_to_skip: Set of edges that should not be modified (validation)
        vector_dtype: Data type to use when converting scalars to arrays
        dataname: if not None checks for memlet data too

    Raises:
        Exception: If an edge marked to skip is encountered during replacement
        because it indicates a bug in the auto-vectorization logic

    Side Effects:
        - Modifies memlet subsets on matching edges
        - May remove and re-add array data descriptors with new shapes
    """
    arr_dim = [((e + 1 - b) // s) for (b, e, s) in new_subset_expr]

    for edge in edges:
        src_node: dace.nodes.Node = edge.src
        dst_node: dace.nodes.Node = edge.dst

        if edge.data is not None and edge.data.subset == old_subset_expr:
            if edge in edges_to_skip:
                raise Exception("AA")
            if edge.data.data != dataname and dataname is not None:
                continue
            if repl_scalars_with_arrays:
                for data_node in [src_node, dst_node]:
                    if isinstance(data_node, dace.nodes.AccessNode):
                        arr = state.sdfg.arrays[data_node.data]
                        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array)
                                                                 and arr.shape == (1, )):
                            state.sdfg.remove_data(data_node.data, validate=False)
                            state.sdfg.add_array(name=data_node.data,
                                                 shape=tuple(arr_dim),
                                                 dtype=vector_numeric_type,
                                                 storage=arr.storage,
                                                 location=arr.location,
                                                 transient=arr.transient,
                                                 lifetime=arr.lifetime)
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))


def expand_memlet_expression(state: SDFGState, edges: Iterable[Edge[Memlet]], edges_to_skip: Set[Edge[Memlet]],
                             vector_width: int) -> Set[Edge[Memlet]]:
    """
    Expand single-element memlet subsets along stride-1 dimensions to a given vector length.
    Pre-condition: all subset dimensions need to be 1

    For each memlet edge, this function modifies subsets that represent a single element
    and extend them to cover `vector_width` elements when the corresponding array stride is 1.
    Trying to modify an edge listed in `edges_to_skip` raises an error as it indicates a
    bug in the auto-vectorization logic.

    Args:
        state (SDFGState): The SDFG state containing the edges.
        edges (Iterable[Edge[Memlet]]): The memlet edges to inspect and possibly modify.
        edges_to_skip (Set[Edge[Memlet]]): Edges that should not be expanded.
        vector_width (int): The number of elements to expand contiguous subsets to.

    Returns:
        Set[Edge[Memlet]]: The set of edges whose memlets were modified.
    """
    modified_edges = set()
    for edge in edges:
        if edge.data is not None:
            assert all(
                ((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset
            ), f"Subset: {[(b, e, s) for b, e, s in edge.data.subset]}, is length one: {[((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset]}"
            if edge in edges_to_skip:
                raise Exception("AA")

            new_subset_list = []
            for (b, e, s), stride in zip(edge.data.subset, state.sdfg.arrays[edge.data.data].strides):
                if stride == 1:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, b + vector_width - 1, s))
                else:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, e, s))
            new_subset_expr = dace.subsets.Range(new_subset_list)

            if new_subset_expr != edge.data.subset:
                edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))
                modified_edges.add(edge)
    return modified_edges


def has_maps(sdfg: dace.SDFG) -> bool:
    """
    Check if the given SDFG or any nested SDFG contains a MapEntry node.

    Args:
        sdfg (dace.SDFG): The SDFG to inspect.

    Returns:
        bool: True if any MapEntry exists in the SDFG hierarchy, False otherwise.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map is innermost (contains no nested maps or maps inside nested SDFGs).

    Args:
        map_entry (dace.nodes.MapEntry): The map entry node to test.
        state (SDFGState): The state containing the map entry.

    Returns:
        bool: True if the map has no inner maps or maps in nested SDFGs, False otherwise.
    """
    nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    if any(isinstance(node, dace.nodes.MapEntry) for node in nodes_between):
        return False
    return not any(isinstance(node, dace.nodes.NestedSDFG) and has_maps(node.sdfg) for node in nodes_between)


def map_consists_of_single_nsdfg_or_no_nsdfg(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map contains either a single NestedSDFG or none at all.

    Args:
        map_entry (dace.nodes.MapEntry): The map entry to check.
        graph: The graph containing the map.

    Returns:
        bool: True if the map contains a single NestedSDFG or no NestedSDFG, False otherwise.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    return (len(all_nodes) == 1 and isinstance(next(
        iter(all_nodes)), dace.nodes.NestedSDFG)) or not any(isinstance(_n, dace.nodes.NestedSDFG) for _n in all_nodes)


def assert_maps_consist_of_single_nsdfg_or_no_nsdfg(sdfg: dace.SDFG) -> None:
    """
    Assert that each map contains either a single NestedSDFG or none at all.

    Args:
        sdfg (dace.SDFG): The SDFG to validate.

    Raises:
        AssertionError: If a map body contains more than one node or a mix of NestedSDFG and other nodes.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            if not map_consists_of_single_nsdfg_or_no_nsdfg(g, n):
                all_nodes = {
                    k
                    for k in g.all_nodes_between(n, g.exit_node(n))
                    if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
                }
                nsdfg_count = len(set([node for node in all_nodes if isinstance(node, dace.nodes.NestedSDFG)]))
                raise AssertionError(f"Got nodes {all_nodes} has {nsdfg_count} nSDFGs")


def no_other_subset(state, recursive: bool = True) -> bool:
    """
    Check if any edge in the state has an 'other_subset' attribute set.

    Args:
        state: The state to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has other_subset set, False otherwise.
    """
    for edge in state.edges():
        if edge.data.other_subset is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not no_other_subset_sdfg(node.sdfg, True):
                    return False
    return True


def no_other_subset_sdfg(sdfg: dace.SDFG, recursive: bool = True) -> bool:
    """
    Check if any edge in the SDFG has an 'other_subset' attribute set.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has other_subset set, False otherwise.
    """
    for state in sdfg.all_states():
        if not no_other_subset(state, recursive):
            return False
    return True


def assert_no_other_subset(sdfg: dace.SDFG, recursive: bool = True):
    """
    Assert that no edge in the SDFG has an 'other_subset' attribute set.

    This validation is needed because vectorization does not support other subsets.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs
    """
    assert no_other_subset_sdfg(sdfg, recursive), "Found edge with other_subset set"


def no_wcr(state, recursive: bool = True) -> bool:
    """
    Check if any edge in the state has a write-conflict resolution (WCR) operation.

    Args:
        state: The state to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has WCR set, False otherwise.
    """
    for edge in state.edges():
        if edge.data.wcr is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not no_wcr_sdfg(node.sdfg, True):
                    return False
    return True


def no_wcr_sdfg(sdfg: dace.SDFG, recursive: bool = True) -> bool:
    """
    Check if any edge in the SDFG has a write-conflict resolution (WCR) operation.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has WCR set, False otherwise.
    """
    for state in sdfg.all_states():
        if not no_wcr(state, recursive):
            return False
    return True


def assert_no_wcr(sdfg: dace.SDFG, recursive: bool = True):
    """
    Assert that no edge in the SDFG has a write-conflict resolution (WCR) operation.

    WCR operations handle conflicting writes to the same memory location. This assertion
    ensures that the SDFG doesn't contain such operations, which are not supported by
    auto-vectorization.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs
    """
    assert no_wcr_sdfg(sdfg, recursive), "Found edge with WCR set"


def last_dim_of_map_is_contiguous_accesses(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if the last dimension of a map performs contiguous accesses.

    Args:
        map_entry (dace.nodes.MapEntry): The map entry to check.
        state: The state containing the map.

    Returns:
        bool: True if all memlets contain the last map parameter, False otherwise.
    """
    nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes)
    # Currently this enforced the map parameter to be involved in the memlet access
    # This will fail in a case such as:
    # _s2 = map_param + 1
    # A[_s2] as `map_param` will not be detected anymore.
    # This can be fixed by improving the analysis, as is an open TODO task.
    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data
        if memlet.subset is None:
            continue
        stride_one_idx = [i for i, s in enumerate(state.sdfg.arrays[edge.data.data].strides) if s == 1][0]
        b, e, s = memlet.subset[stride_one_idx]
        b_free_syms = b.free_symbols if hasattr(b, "free_symbols") else set()
        e_free_syms = e.free_symbols if hasattr(e, "free_symbols") else set()
        free_symbols = {str(s) for s in b_free_syms.union(e_free_syms)}
        last_param = str(list(map_entry.map.params)[-1])
        if last_param not in free_symbols and free_symbols != set():
            return False
    return True


def assert_last_dim_of_maps_are_contigous_accesses(sdfg: dace.SDFG):
    """
    Assert that the last dimension of all maps in an SDFG performs contiguous accesses.

    For each innermost map, this function checks that the last map parameter
    appears in every memlet within the map body, ensuring the last dimension
    corresponds to contiguous memory accesses. It also validates that all tasklets
    are properly enclosed in map scopes or nested SDFGs.

    Args:
        sdfg (dace.SDFG): The SDFG to check.

    Raises:
        Exception: If a tasklet is not enclosed in any map or valid nested SDFG scope.
        ValueError: If a node's parent scope is not a map, or the last map parameter
            does not appear in a memlet subset expression.
        AssertionError: If internal consistency assumptions about map nesting fail.
    """
    checked_map_entries = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            # Skip map entries/exits
            if isinstance(node, (dace.nodes.MapEntry, dace.nodes.MapExit)):
                continue

            map_entry = state.scope_dict()[node]
            if map_entry is None:
                if isinstance(node, dace.nodes.Tasklet):
                    parent_nsdfg = state.sdfg.parent_nsdfg_node
                    if parent_nsdfg is None:
                        continue
                    parent_state = find_state_of_nsdfg_node(sdfg, node)
                    parent_scope = parent_state.scope_dict()[parent_nsdfg]
                    if parent_scope is None or not isinstance(parent_scope, dace.nodes.MapEntry):
                        raise Exception(f"No NSDFGs that are not within Map scopes should be left, "
                                        f"check {parent_nsdfg} in state {parent_state}. Call inlineSDFG")
                else:
                    continue
            else:
                if not isinstance(map_entry, dace.nodes.MapEntry):
                    raise ValueError(f"Parent scope of node {node} is not a map, found {map_entry} in state {state}.")
                checked_map_entries.add(map_entry)

            if map_entry not in checked_map_entries:
                assert isinstance(
                    map_entry, dace.nodes.MapEntry), f"Parent scope of node {node} is not a map, returned {map_entry}."
                # Currently this enforced the map parameter to be involved in the memlet access
                # This will fail in a case such as:
                # _s2 = map_param + 1
                # A[_s2] as `map_param` will not be detected anymore.
                # This can be fixed by improving the analysis, as is an open TODO task.
                if not last_dim_of_map_is_contiguous_accesses(state, map_entry):
                    raise ValueError(f"Last map parameter must be in the memlet, "
                                     f"not in this case {map_entry}, {state}")


def to_ints(sym_epxr: dace.symbolic.SymExpr) -> typing.Union[int, None]:
    """
    Try to convert a symbolic expression to an integer.

    Args:
        sym_epxr (dace.symbolic.SymExpr): The symbolic expression to convert.

    Returns:
        int | None: The integer value if conversion succeeds, otherwise None.
    """
    try:
        return int(sym_epxr)
    except Exception:
        return None


def get_vector_max_access_ranges(state: SDFGState, node: dace.nodes.NestedSDFG) -> Dict[str, str]:
    """
    Extract the maximum access range for vectorized map parameters.

    This function analyzes the nested map hierarchy to determine the maximum
    iteration range for vector map parameters. It walks up the scope hierarchy (two steps)
    from the nested SDFG through its vectorized map (vmap) to the outer data-parallel map (dmap),
    extracting the end bounds that constrain vector access ranges.

    The typical use case is for vectorization where you have:
    - Outer data-parallel map (dmap): iterates over independent data chunks
    - Inner vector map (vmap): 1-vector op, map of form (i:i+vector_simd_len:vector_simd_len)
    - Nested SDFG: contains the actual computation

    Args:
        state: The SDFG state containing the nested SDFG node
        node: The nested SDFG node whose vector access ranges to determine

    Returns:
        Dictionary mapping vector map parameter names to their maximum values
        (end bounds from the data-parallel map)

    Example:
        For a hierarchy:
        ```
        map i=0:N (data-parallel map)
          map i_v=i:i+4:4 (vector map, vectorizing over 'i')
            NestedSDFG
        ```
        Returns: {'i_v': 'N'}

    Note:
        This assumes a two-level map hierarchy with the nested SDFG inside
        a vector map, which is itself inside a data-parallel map.
    """
    # Get scope hierarchy: nsdfg -> vector_map -> data_map
    scope_dict = state.scope_dict()

    # Vector map is the immediate parent of the nested SDFG
    vector_map = scope_dict[node]

    # Build mapping: vector_param -> vector_begin_expr
    # and reverse: vector_begin_expr -> vector_param
    v_params_to_begins = {}
    v_begins_to_params = {}
    for param, (begin, end, step) in zip(vector_map.map.params, vector_map.map.range):
        v_params_to_begins[param] = str(begin)
        v_begins_to_params[str(begin)] = param

    # Data-parallel map is the parent of the vector map
    data_map = scope_dict[vector_map]

    # Build mappings for data-parallel map parameters
    d_params_to_begins = {}
    d_begins_to_params = {}
    d_params_to_ends = {}
    for param, (begin, end, step) in zip(data_map.map.params, data_map.map.range):
        d_params_to_begins[param] = str(begin)
        d_begins_to_params[str(begin)] = param
        d_params_to_ends[param] = str(end)

    # For each vector parameter, find its maximum bound from the data map
    # The vector map begin expression should match a data map parameter,
    # allowing us to look up the corresponding end bound
    param_max_ranges = {}
    for v_param in vector_map.map.params:
        # Get the begin expression of the vector parameter (e.g., 'i')
        v_begin_expr = v_params_to_begins[v_param]

        # Look up the corresponding data map end bound (e.g., 'N')
        param_max_ranges[v_param] = d_params_to_ends[v_begin_expr]

    return param_max_ranges


def prepare_vectorized_array(state: dace.SDFGState,
                             inner_sdfg: dace.SDFG,
                             inner_arr_name: str,
                             orig_dataname: str,
                             orig_arr: dace.data.Data,
                             subset: dace.subsets.Range,
                             vector_width: dace.symbolic.SymExpr,
                             vector_storage: dace.dtypes.StorageType,
                             reuse_name_if_existing: bool = False,
                             use_name: str = None):
    """
    Prepares a vectorized array by creating the vector array in outer SDFG
    and replacing the inner array with vectorized version.

    Args:
        state: The SDFG state
        inner_sdfg: The inner SDFG containing the array
        inner_arr_name: Name of the array to vectorize
        orig_dataname: Original data array name
        orig_arr: Original outer array descriptor
        memlet: Memlet for determining offsets
        vector_width: Width of the vector
        vector_storage: Storage type for the vector
        reuse_name_if_existing: Does not find a new name
    Returns:
        tuple: (vector_dataname, inner_offset or 0)
    """
    # Create vector array in outer SDFG
    #print(f"perpare, {inner_arr_name}, {orig_dataname}, {subset}")

    vector_dataname_candidate = orig_dataname + "_vec_k" if use_name is None else use_name
    if reuse_name_if_existing:
        vector_dataname = vector_dataname_candidate
        if vector_dataname not in state.sdfg.arrays:
            state.sdfg.add_array(name=vector_dataname_candidate,
                                 shape=(vector_width, ),
                                 dtype=orig_arr.dtype,
                                 location=orig_arr.location,
                                 transient=True,
                                 find_new_name=False,
                                 storage=vector_storage)
    else:
        vector_dataname, _ = state.sdfg.add_array(name=vector_dataname_candidate,
                                                  shape=(vector_width, ),
                                                  dtype=orig_arr.dtype,
                                                  location=orig_arr.location,
                                                  transient=True,
                                                  find_new_name=True,
                                                  storage=vector_storage)

    # Replace the array inside inner SDFG
    prev_inner_arr = inner_sdfg.arrays[inner_arr_name]
    inner_sdfg.remove_data(inner_arr_name, False)
    inner_sdfg.add_array(name=inner_arr_name,
                         shape=(vector_width, ),
                         dtype=orig_arr.dtype,
                         location=orig_arr.location,
                         transient=False,
                         find_new_name=False,
                         storage=vector_storage)

    # Handle multi-dimensional arrays
    inner_offset = 0
    if len(orig_arr.shape) > 1:
        keep_mask = [0 for _ in orig_arr.shape]
        keep_mask[-1] = 1  # Not always correct
        drop_dims(inner_sdfg, inner_arr_name, keep_mask)

        # Calculate offset for inner array
        #inner_offset = [b for (b, e, s), stride in zip(subset, prev_inner_arr.strides) if stride == 1][0]
        inner_offset2 = [(b, b, 1) for (b, e, s), stride in zip(subset, prev_inner_arr.strides)]

        #print("IJ", inner_offset, inner_offset2)
        #print("B", subset)
        # Input must have offset this already
        if not (reuse_name_if_existing is True and use_name is not None):
            offset_memlets(inner_sdfg, inner_arr_name, inner_offset2)
        #print("A", subset)

    return vector_dataname, inner_offset


def compute_edge_subset(edge_subset, subset, orig_arr, inner_offset, vector_width):
    """
    Computes the copy subset based on stride and offset.

    Args:
        edge_subset: Subset from the edge
        subset: Subset from the memlet
        orig_arr: Original array descriptor
        inner_offset: Offset value
        vector_width: Width of the vector

    Returns:
        dace.subsets.Range: The copy subset
    """
    # Get stride-1 begin value
    if len(subset) == len(orig_arr.strides):
        stride_one_subset = [b for (b, e, s), stride in zip(subset, orig_arr.strides) if stride == 1]
        assert len(stride_one_subset) == 1, f"{stride_one_subset} != 1: {orig_arr.strides}, {subset}"
        stride_one_begin = stride_one_subset[0]
        stride_one_indices = [i for i, stride in enumerate(orig_arr.strides) if stride == 1]
        #print("C1", edge_subset)
        #print("C2", stride_one_subset)
        #print("C3", stride_one_begin)
        #print("C4", stride_one_begin == 0)
        #print("C5", subset)
        #print((inner_offset, inner_offset + vector_width - 1, 1))

        # If the inner subset starts from 0, then to the SDFG just the subset accessed is passed
        # In that case we copy the edge as it is
        # Otherwise we need to generate the mapping (using the subst (and not edge subset))
        stride_one_idx = stride_one_indices[0]
        stride_one_begin = subset[stride_one_idx][0]

        if stride_one_begin != 0:
            new_subset = list(subset)
            b, e, s = new_subset[stride_one_idx]
            new_subset[stride_one_idx] = (b + inner_offset, b + inner_offset + vector_width - 1, 1)
            return dace.subsets.Range(new_subset)
        else:
            #print("R2", edge_subset)
            return copy.deepcopy(edge_subset)
    else:
        # Definitely a smaller subset has ben taken due to the dimension change
        return copy.deepcopy(edge_subset)


def process_in_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                     vector_width: int, vector_storage: dace.dtypes.StorageType) -> Set[str]:
    """
    Process input edges for movable arrays.
    Returns added array names.

    Args:
        state: The SDFG state
        nsdfg_node: The nested SDFG node
        movable_arrays: List of (array_name, memlet) tuples
        vector_width: Width of the vector
        vector_storage: Storage type for the vector
    """
    assert isinstance(nsdfg_node, dace.nodes.NestedSDFG)
    inner_sdfg = nsdfg_node.sdfg

    vectorized_datanames = set()
    for movable_arr_name, subset in movable_arrays:
        #print("SS", movable_arr_name, subset, type(subset))
        in_edges = list(state.in_edges_by_connector(nsdfg_node, movable_arr_name))
        assert len(in_edges) <= 1

        for ie in in_edges:
            orig_arr = state.sdfg.arrays[ie.data.data]
            inner_arr_name = ie.dst_conn

            # Prepare vectorized arrays
            # This subset will be offset, copy the prev one
            prev_subset = copy.deepcopy(subset)
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_arr_name, ie.data.data,
                                                                     orig_arr, subset, vector_width, vector_storage)
            vectorized_datanames.add(vector_dataname)

            # Compute copy subset
            #print("subset", subset, "prev subset", prev_subset)
            copy_subset = compute_edge_subset(ie.data.subset, prev_subset, orig_arr, inner_offset, vector_width)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            #an.setzero = True
            state.remove_edge(ie)
            state.add_edge(ie.src, ie.src_conn, an, None, dace.memlet.Memlet(data=ie.data.data, subset=copy_subset))
            state.add_edge(an, None, ie.dst, ie.dst_conn,
                           dace.memlet.Memlet.from_array(vector_dataname, state.sdfg.arrays[vector_dataname]))

    return vectorized_datanames


def process_out_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                      vector_width: int, vector_storage: dace.dtypes.StorageType):
    """
    Process output edges for movable arrays.

    Args:
        state: The SDFG state
        nsdfg_node: The nested SDFG node
        movable_arrays: List of (array_name, memlet) tuples
        vector_width: Width of the vector
        vector_storage: Storage type for the vector
    """
    inner_sdfg = nsdfg_node.sdfg

    for id, (movable_arr_name, subset) in enumerate(movable_arrays):
        out_edges = list(state.out_edges_by_connector(nsdfg_node, movable_arr_name))
        assert len(out_edges) <= 1

        for oe in out_edges:
            orig_arr = state.sdfg.arrays[oe.data.data]
            inner_arr_name = oe.src_conn

            inout_data_name = None
            # Check inout connector if nsdfg
            if isinstance(oe.src, dace.nodes.NestedSDFG) and oe.src_conn in oe.src.in_connectors:
                # Inout connector means, this array should have been added
                ie_datas = {ie.data.data for ie in state.in_edges_by_connector(nsdfg_node, oe.src_conn)}
                assert len(ie_datas) == 1
                ie_data = ie_datas.pop()  # This can be vectorized
                assert oe.data.data == ie_data or ie_data.startswith(
                    oe.data.data + "_vec"
                ), f"{oe.data.data} != {ie_data} and {ie_data} not startswith {oe.data.data + '_vec'} (from {inner_arr_name}) not in {state.sdfg.arrays}"
                inout_data_name = ie_data

            # Prepare vectorized arrays
            # Copy it to avoid it changing
            prev_subset = copy.deepcopy(subset)
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_arr_name, oe.data.data,
                                                                     orig_arr, subset, vector_width, vector_storage,
                                                                     True, inout_data_name)

            # Compute copy subset
            copy_subset = compute_edge_subset(oe.data.subset, prev_subset, orig_arr, inner_offset, vector_width)
            #print("CO", prev_subset, copy_subset)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            #an.setzero = True
            state.remove_edge(oe)
            assert oe.src == nsdfg_node
            assert oe.src_conn is not None
            assert len(set(state.out_edges_by_connector(nsdfg_node, oe.src_conn))) == 0
            state.add_edge(oe.src, oe.src_conn, an, None,
                           dace.memlet.Memlet.from_array(vector_dataname, state.sdfg.arrays[vector_dataname]))
            state.add_edge(an, None, oe.dst, oe.dst_conn, dace.memlet.Memlet(data=oe.data.data, subset=copy_subset))

    state.sdfg.validate()


def offset_memlets(sdfg: dace.SDFG, dataname: str, offsets: List[dace.symbolic.SymExpr]):
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None and edge.data.data == dataname:
                #print(edge.data.subset)
                subset = edge.data.subset.offset_new(dace.subsets.Range(offsets), negative=True)
                # If subset is not one dimensional we need to collapse 0 accesses
                collapsed_subset_list = [(b, e, s) for (b, e, s) in subset if (e + 1 - b) // s != 1]
                edge.data.subset = dace.subsets.Range(collapsed_subset_list)


def match_connector_to_data(state: dace.SDFGState, tasklet: dace.nodes.Tasklet) -> dict[str, dace.data.Data]:
    """
    Map tasklet input connectors to their corresponding array descriptors.

    Args:
        state (dace.SDFGState): The state containing the tasklet.
        tasklet (dace.nodes.Tasklet): The tasklet whose connectors are inspected.

    Returns:
        dict[str, dace.data.Data]: A mapping from tasklet input connector names
        to the corresponding array descriptors in the SDFG.
    """
    tdict = dict()
    for ie in state.in_edges(tasklet):
        if ie.data is not None:
            tdict[ie.dst_conn] = state.sdfg.arrays[ie.data.data]
    return tdict


def assert_strides_are_packed_C_or_packed_Fortran(sdfg: dace.SDFG) -> Union[str, None]:
    """
    Verify that all arrays in an SDFG are packed in consistent C or Fortran order.

    Each array must have a unit stride in either the first or last dimension for auto-vectorization.
    The function determines whether the layout is consistently Fortran (F) or
    C-style (C). One-dimensional arrays are allowed in both and default to 'F'.

    Args:
        sdfg (dace.SDFG): The SDFG whose arrays are checked.

    Returns:
        str | None: "C" or "F" indicating the stride ordering, or None if no arrays are found.

    Raises:
        AssertionError: If an array lacks a unit stride in both first and last dimension.
        ValueError: If arrays have mixed C and Fortran stride ordering.
    """
    stride_type = None
    has_one_d_arrays = False
    current_type = None
    for arr_name, desc in sdfg.arrays.items():
        if not isinstance(desc, dace.data.Array):
            continue

        has_unit_stride = desc.strides[0] == 1 or desc.strides[-1] == 1
        assert has_unit_stride, f"Array {arr_name} needs unit stride in first or last dimension: {desc.strides}"

        if len(desc.shape) == 1:
            has_one_d_arrays = True
        else:
            current_type = "F" if desc.strides[0] == 1 else "C"

        if stride_type is None:
            stride_type = current_type
        elif stride_type != current_type:
            raise ValueError("All arrays must have consistent stride ordering (all F or all C)")

    if has_one_d_arrays and stride_type is None:
        stride_type = "F"

    return stride_type


def find_state_of_nsdfg_node(root_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG) -> dace.SDFGState:
    for n, g in root_sdfg.all_nodes_recursive():
        if n == nsdfg_node:
            return root_sdfg
    raise Exception(f"State of the nsdfg node ({nsdfg_node}) not found in the root SDFG ({root_sdfg.label})")


def check_nsdfg_connector_array_shapes_match(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    """
    Validate that nested SDFG connector arrays match their memlet subset shapes.
    This is to avoid memlet-squeezing issues going to the nested SDFGs

    This function checks both input and output edges of a nested SDFG to ensure
    that the array shapes inside the nested SDFG match the shapes implied by
    the memlet subsets on the edges connecting to the parent SDFG.

    The validation considers multiple shape interpretations:
    1. Full shape with unit stride: (end + 1 - begin)
    2. Shape accounting for stride: (end + 1 - begin) // stride
    3. Collapsed shapes (excluding size-1 dimensions)

    Args:
        parent_state: The state in the parent SDFG containing the nested SDFG node
        nsdfg_node: The nested SDFG node whose connector shapes to validate

    Raises:
        AssertionError: If any connector array shape doesn't match any of the
                       expected shape interpretations, with detailed error message


    Note:
        This is a validation-only function - it does not modify the SDFG.
        Use fix_nsdfg_connector_array_shapes_mismatch() to automatically
        correct detected mismatches.
    """
    # ===== Validate Input Edges =====
    for in_edge in parent_state.in_edges(nsdfg_node):
        if in_edge.data.data is None:
            continue

        subset = in_edge.data.subset
        connector_name = in_edge.dst_conn  # Connector name in nested SDFG
        connector_array = nsdfg_node.sdfg.arrays[connector_name]

        # Calculate expected shapes based on subset
        # Shape 1: Full dimension size (end - begin + 1)
        expected_shape_full = tuple([(end + 1 - begin) for begin, end, step in subset])

        # Shape 2: Effective size accounting for stride
        expected_shape_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset])

        # Shape 3: Collapsed (remove size-1 dimensions from full shape)
        expected_shape_collapsed_full = tuple([(end + 1 - begin) for begin, end, step in subset
                                               if (end + 1 - begin) != 1])

        # Shape 4: Collapsed with stride (remove size-1 dimensions from strided)
        expected_shape_collapsed_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset
                                                  if (end + 1 - begin) // step != 1])

        # Validate: array shape must match one of the expected shapes
        shape_matches = (connector_array.shape == expected_shape_full or connector_array.shape == expected_shape_strided
                         or connector_array.shape == expected_shape_collapsed_full
                         or connector_array.shape == expected_shape_collapsed_strided)

        assert shape_matches, (f"Shape mismatch for input connector '{connector_name}':\n"
                               f"  Array shape: {connector_array.shape}\n"
                               f"  Expected one of:\n"
                               f"    Full:              {expected_shape_full}\n"
                               f"    Strided:           {expected_shape_strided}\n"
                               f"    Collapsed full:    {expected_shape_collapsed_full}\n"
                               f"    Collapsed strided: {expected_shape_collapsed_strided}")

    # ===== Validate Output Edges =====
    for out_edge in parent_state.out_edges(nsdfg_node):
        if out_edge.data is None:
            continue

        subset = out_edge.data.subset
        connector_name = out_edge.src_conn  # Connector name in nested SDFG
        connector_array = nsdfg_node.sdfg.arrays[connector_name]

        # Calculate expected shapes (same logic as input edges)
        expected_shape_full = tuple([(end + 1 - begin) for begin, end, step in subset])

        expected_shape_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([(end + 1 - begin) for begin, end, step in subset
                                               if (end + 1 - begin) != 1])

        expected_shape_collapsed_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset
                                                  if (end + 1 - begin) // step != 1])

        # Validate: array shape must match one of the expected shapes
        shape_matches = (connector_array.shape == expected_shape_full or connector_array.shape == expected_shape_strided
                         or connector_array.shape == expected_shape_collapsed_full
                         or connector_array.shape == expected_shape_collapsed_strided)

        assert shape_matches, (f"Shape mismatch for output connector '{connector_name}':\n"
                               f"  Array shape: {connector_array.shape}\n"
                               f"  Expected one of:\n"
                               f"    Full:              {expected_shape_full}\n"
                               f"    Strided:           {expected_shape_strided}\n"
                               f"    Collapsed full:    {expected_shape_collapsed_full}\n"
                               f"    Collapsed strided: {expected_shape_collapsed_strided}")


def fix_nsdfg_connector_array_shapes_mismatch(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG) -> None:
    """
    Automatically fix shape mismatches in nested SDFG connector arrays.

    This function detects and corrects shape mismatches between connector arrays
    inside a nested SDFG and their corresponding memlet subsets in the parent SDFG.
    (see also `check_nsdfg_connector_array_shapes_match`)

    Fix strategy:
    1. Calculate expected shape from memlet subset (collapsed, removing size-1 dims)
    2. If shape mismatch detected, recreate array with correct shape and strides
    3. Update all accesses inside the nested SDFG using drop_dims() transformation

    This is particularly useful after transformations that:
    - Modify memlet subsets (e.g., vectorization, tiling)
    - Collapse dimensions (e.g., constant folding, loop unrolling)
    - Change access patterns (e.g., stride modifications)

    Args:
        parent_state: The state in the parent SDFG containing the nested SDFG node
        nsdfg_node: The nested SDFG node whose connector shapes to fix

    """

    # ===== Fix Input Edge Connector Arrays =====
    for in_edge in parent_state.in_edges(nsdfg_node):
        if in_edge.data.data is None:
            continue

        subset = in_edge.data.subset
        connector_name = in_edge.dst_conn
        connector_array = nsdfg_node.sdfg.arrays[connector_name]
        original_shape = connector_array.shape

        # Calculate all possible expected shapes
        expected_shape_full = tuple([(end + 1 - begin).simplify() for begin, end, step in subset])

        expected_shape_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset
                                                  if ((end + 1 - begin) // step).simplify() != 1])

        # Calculate strides for collapsed shape (excluding size-1 dimensions)
        strides_collapsed = tuple([
            stride for (begin, end, step), stride in zip(subset, connector_array.strides)
            if (end + 1 - begin).simplify() != 1
        ])

        # Check if shape matches any expected pattern
        shape_matches = (connector_array.shape == expected_shape_full or connector_array.shape == expected_shape_strided
                         or connector_array.shape == expected_shape_collapsed_full
                         or connector_array.shape == expected_shape_collapsed_strided)

        if shape_matches:
            continue  # No fix needed

        # ===== Mismatch detected - fix it =====

        # Remove old array descriptor
        nsdfg_node.sdfg.remove_data(connector_name, validate=False)

        # Recreate array with collapsed shape and adjusted strides
        nsdfg_node.sdfg.add_array(
            name=connector_name,
            shape=expected_shape_collapsed_full,
            strides=strides_collapsed,
            storage=connector_array.storage,
            dtype=connector_array.dtype,
            location=connector_array.location,
            transient=False,  # Connectors are non-transient
            lifetime=connector_array.lifetime,
            debuginfo=connector_array.debuginfo,
            allow_conflicts=connector_array.allow_conflicts,
            find_new_name=False,
            alignment=connector_array.alignment,
            may_alias=False)

        # Determine which dimensions to keep (1) vs drop (0)
        # Keep dimensions that have size > 1
        dims_to_keep = [1 if (end + 1 - begin) != 1 else 0 for begin, end, step in subset]

        # Update all accesses inside nested SDFG if:
        # 1. Not a 1D array (len > 1)
        # 2. Original shape matches the subset dimensionality
        # 3. Original shape had more dimensions than the collapsed shape
        should_drop_dims = (len(dims_to_keep) != 1 and len(original_shape) == len(dims_to_keep)
                            and len(original_shape) > len(expected_shape_collapsed_full))

        if should_drop_dims:
            drop_dims(nsdfg_node.sdfg, dims_to_keep, connector_name)

    # ===== Fix Output Edge Connector Arrays =====
    for out_edge in parent_state.out_edges(nsdfg_node):
        if out_edge.data is None:
            continue

        subset = out_edge.data.subset
        connector_name = out_edge.src_conn
        connector_array = nsdfg_node.sdfg.arrays[connector_name]
        original_shape = connector_array.shape

        # Calculate all possible expected shapes
        expected_shape_full = tuple([(end + 1 - begin).simplify() for begin, end, step in subset])

        expected_shape_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset
                                                  if ((end + 1 - begin) // step).simplify() != 1])

        # Calculate strides for collapsed shape (excluding size-1 dimensions)
        strides_collapsed = tuple(
            [stride for (begin, end, step), stride in zip(subset, connector_array.strides) if (end + 1 - begin) != 1])

        # Check if shape matches any expected pattern
        shape_matches = (connector_array.shape == expected_shape_full or connector_array.shape == expected_shape_strided
                         or connector_array.shape == expected_shape_collapsed_full
                         or connector_array.shape == expected_shape_collapsed_strided)

        if shape_matches:
            continue  # No fix needed

        # ===== Mismatch detected - fix it =====

        # Remove old array descriptor
        nsdfg_node.sdfg.remove_data(connector_name, validate=False)

        # Recreate array with collapsed shape and adjusted strides
        nsdfg_node.sdfg.add_array(
            name=connector_name,
            shape=expected_shape_collapsed_full,
            strides=strides_collapsed,
            storage=connector_array.storage,
            dtype=connector_array.dtype,
            location=connector_array.location,
            transient=False,  # Connectors are non-transient
            lifetime=connector_array.lifetime,
            debuginfo=connector_array.debuginfo,
            allow_conflicts=connector_array.allow_conflicts,
            find_new_name=False,
            alignment=connector_array.alignment,
            may_alias=False)

        # Determine which dimensions to keep (1) vs drop (0)
        dims_to_keep = [1 if (end + 1 - begin) != 1 else 0 for begin, end, step in subset]

        # Debug output for tracking dimension dropping

        # Update all accesses inside nested SDFG if:
        # 1. Not a 1D array (len > 1)
        # 2. Original shape matches the subset dimensionality
        # 3. Original shape had more dimensions than the collapsed shape
        should_drop_dims = (len(dims_to_keep) != 1 and len(original_shape) == len(dims_to_keep)
                            and len(original_shape) > len(expected_shape_collapsed_full))

        if should_drop_dims:
            drop_dims(nsdfg_node.sdfg, dims_to_keep, connector_name)


def extract_bracket_contents(expr: str, name: str):
    """
    Extracts the contents inside the brackets following a given variable name in an expression.

    This function searches for a pattern of the form `<name>[...]` within a given string and
    extracts the full matched substring and the individual elements inside the brackets. It
    correctly handles nested brackets and quoted strings, ensuring that commas within nested
    structures or strings are not treated as separators.

    Args:
        expr (str): The input expression to search within.
        name (str): The name of the variable or function to match before the brackets.

    Returns:
        tuple[str, list[str]]:
            A tuple containing:
              - The full matched string (e.g., `"A[1, (2, 3), 'x']"`), or an empty string if no match is found.
              - A list of strings representing the top-level comma-separated elements inside the brackets.
    """
    # Find <name>[...]
    pattern = rf'\b{name}\s*\[(.*)\]'
    match = re.search(pattern, expr)
    if not match:
        return "", []

    full_match = match.group(0)
    inside = match.group(1)
    parts = []
    current = []
    depth = 0
    in_string = None

    for ch in inside:
        if in_string:
            current.append(ch)
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            current.append(ch)
        elif ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)

    # Add last part
    if current:
        parts.append(''.join(current).strip())

    return full_match, parts


def drop_dims_from_str(src_str: str, dim_mask: Tuple[int], dataname: str) -> str:
    """
    Remove dimensions from array accesses in a code string.

    This function finds array access patterns for a specific array name in a
    string and removes dimensions according to a binary mask. Dimensions with
    mask value 0 are removed, dimensions with mask value 1 are kept.

    Args:
        src_str: Source string containing potential array accesses
        dim_mask: Tuple of 0s and 1s indicating which dimensions to keep
                 (1 = keep, 0 = drop)
        dataname: Name of the array whose accesses should be modified

    Returns:
        Modified string with dimensions dropped from array accesses

    Examples:
        >>> drop_dims_from_str("A[i, 0, j] + B[k]", (1, 0, 1), "A")
        "A[i, j] + B[k]"


    Raises:
        AssertionError: If the number of dimensions in the access doesn't match
                        the length of dim_mask

    Notes:
        - If no array access is found, returns the original string unchanged
    """
    # Find and parse array access in the string
    matched_substring, access_tokens = extract_bracket_contents(src_str, dataname)

    if access_tokens == []:
        # No array access found - return unchanged
        return src_str

    # Verify dimension count matches
    assert len(access_tokens) == len(dim_mask), (
        f"Dimension mismatch: array access has {len(access_tokens)} dimensions "
        f"but dim_mask has {len(dim_mask)} entries")

    # Filter dimensions: keep only those where mask is 1
    filtered = [dimension_expr for dimension_expr, mask_bit in zip(access_tokens, dim_mask) if mask_bit]

    filtered_str = "[" + ", ".join(filtered) + "]"

    return src_str.replace(matched_substring, dataname + filtered_str)


def drop_dims(sdfg: dace.SDFG, dim_mask: Tuple[int], dataname: str) -> None:
    """
    Remove dimensions from all accesses to a specific array throughout an SDFG.

    This function traverses an entire SDFG and updates all references to a
    specific array, removing dimensions according to the provided mask. It
    updates:
    1. Memlet subsets on edges
    2. Loop conditions, init statements, and update statements
    3. Conditional branch conditions
    4. Interstate edge assignments

    Args:
        sdfg: The SDFG to modify
        dim_mask: Tuple of 0s and 1s indicating which dimensions to keep
                 (1 = keep, 0 = drop). Length must match array dimensionality.
        dataname: Name of the array whose accesses to update

    """

    for cfg_region in sdfg.all_control_flow_regions():

        # Handle loop regions: update loop condition, init, and update statements
        if isinstance(cfg_region, LoopRegion):
            # Update loop condition (e.g., "i < N and A[i] > 0")
            cfg_region.loop_condition = CodeBlock(
                drop_dims_from_str(cfg_region.loop_condition.as_string, dim_mask, dataname),
                cfg_region.loop_condition.language)

            # Update loop update statement (e.g., "i = i + 1")
            cfg_region.update_statement = CodeBlock(
                drop_dims_from_str(cfg_region.update_statement.as_string, dim_mask, dataname),
                cfg_region.update_statement.language)

            # Update loop initialization (e.g., "i = A[0]")
            cfg_region.init_statement = CodeBlock(
                drop_dims_from_str(cfg_region.init_statement.as_string, dim_mask, dataname),
                cfg_region.init_statement.language)

        # Handle conditional blocks: update branch conditions
        elif isinstance(cfg_region, ConditionalBlock):
            # Each branch is a (condition, block) tuple
            for i, (condition, block) in enumerate(cfg_region.branches):
                # Update condition expression
                new_condition = CodeBlock(drop_dims_from_str(condition.as_string, dim_mask, dataname),
                                          condition.language)
                # Replace the branch with updated condition
                cfg_region.branches[i] = (new_condition, block)

    for state in sdfg.all_states():
        if not isinstance(state, SDFGState):
            continue

        for edge in state.edges():
            # Check if this edge accesses the target array
            if edge.data.data is None or edge.data.data != dataname:
                continue

            # Verify dimension count matches
            assert len(edge.data.subset) == len(dim_mask), (f"Memlet subset has {len(edge.data.subset)} dimensions "
                                                            f"but dim_mask has {len(dim_mask)} entries")

            # Build new subset with filtered dimensions
            new_subset = []
            for (begin, end, step), mask_bit in zip(edge.data.subset, dim_mask):
                if mask_bit:  # Keep this dimension
                    new_subset.append((begin, end, step))

            # Create new memlet with reduced dimensionality
            edge.data = dace.memlet.Memlet(data=dataname, subset=dace.subsets.Range(new_subset))

    for interstate_edge in sdfg.all_interstate_edges():
        # Skip edges without assignments
        if interstate_edge.data.assignments == {}:
            continue

        # Update each assignment expression
        new_assignments = {}
        for var_name, expr in interstate_edge.data.assignments.items():
            # Apply dimension dropping to the expression
            new_assignments[var_name] = drop_dims_from_str(expr, dim_mask, dataname)

        # Replace assignments
        interstate_edge.data.assignments = new_assignments


def offset_symbol_in_expression(expr_str: str, symbol_to_offset: str, offset: int) -> str:
    """
    Returns a new expression string where a specified symbol is incremented by a given offset.

    Args:
        expr_str (str): The original expression as a string.
        symbol_to_offset (str): The symbol within the expression to offset.
        offset (int): The integer value to add to the symbol.

    Returns:
        str: A new expression string with the symbol offset.
             If the symbol is not found in the expression, returns the original expression string unchanged.
    """
    if "Eq(" in expr_str:
        raise Exception(expr_str)
    expr = dace.symbolic.SymExpr(expr_str)
    sym_to_change = None
    for free_sym in expr.free_symbols:
        if str(free_sym) == symbol_to_offset:
            sym_to_change = free_sym
            break
    if sym_to_change is None:
        return expr_str
    offsetted_expr = f"({sym_to_change} + {offset})"
    offset_expr = expr.subs(sym_to_change, offsetted_expr)
    return sympy.pycode(offset_expr)


def use_laneid_symbol_in_expression(expr_str: str, symbol_to_offset: str, offset: int) -> str:
    """
    Returns a new expression string where a specified symbol is replaced with laneid-ed version
    `sym1` -> `sym1_laneid_{offset}`

    Args:
        expr_str (str): The original expression as a string.
        symbol_to_offset (str): The symbol within the expression to offset.
        offset (int): The integer value to add to the symbol.

    Returns:
        str: A new expression string with the symbol offset.
             If the symbol is not found in the expression, returns the original expression string unchanged.
    """
    if "Eq(" in expr_str:
        raise Exception(expr_str)
    expr = dace.symbolic.SymExpr(expr_str)
    sym_to_change = None
    for free_sym in expr.free_symbols:
        if str(free_sym) == symbol_to_offset:
            sym_to_change = free_sym
            break
    if sym_to_change is None:
        return expr_str
    offsetted_expr = f"({sym_to_change}_laneid_{offset})"
    offset_expr = expr.subs(sym_to_change, offsetted_expr)
    return sympy.pycode(offset_expr)


def instantiate_tasklet_from_info(state: dace.SDFGState, node: dace.nodes.Tasklet, info: dict, vector_width: int,
                                  templates: Dict[str, str], vector_map_param: str, vector_dtype: typeclass,
                                  tasklet_prefix:str, fail_on_fallback: bool = True) -> None:
    """
    Instantiates a tasklet's code block in vectorized form based on classification info.

    This function takes a tasklet and its classification `info` (from `classify_tasklet`) and
    updates `node.code` to a vectorized CodeBlock using the provided templates. Handles
    different tasklet types (array-array, array-scalar, scalar-symbol, etc.) and supports
    vectorization over the specified width.

    Args:
        state: The SDFGState containing the tasklet.
        node: The tasklet node to instantiate.
        info: Classification dictionary containing:
            - "type": TaskletType enum describing operand types.
            - "lhs": Left-hand side variable.
            - "rhs1": First right-hand side variable.
            - "rhs2": Second right-hand side variable (optional).
            - "constant1": First constant operand (optional).
            - "constant2": Second constant operand (optional).
            - "op": Operation string (e.g., "+", "*", "=").
        vector_width: Number of lanes for vectorization.
        templates: Mapping from operation strings to template strings for code generation.
        vector_map_param: Name of the map parameter used for lane indexing in vectorization.
    """
    # Extract classification info
    ttype: tutil.TaskletType = info.get("type")
    lhs, rhs1, rhs2 = info.get("lhs"), info.get("rhs1"), info.get("rhs2")
    c1, c2, op = info.get("constant1"), info.get("constant2"), info.get("op")
    vw = vector_width
    is_commutative = op in {"+", "*", "==", "!="}

    # Cast boolean constants to C-compatible names
    PYTHON_TO_CPP_OPERATORS = {"and": "&&", "or": "||", "not": "!"}
    op = PYTHON_TO_CPP_OPERATORS.get(op, op)

    ies = state.in_edges(node)
    oes = state.out_edges(node)
    in_dtypes = {state.sdfg.arrays[ie.data.data].dtype for ie in ies if ie.data.data is not None}
    out_dtypes = {state.sdfg.arrays[oe.data.data].dtype for oe in oes if oe.data.data is not None}
    all_dtypes = in_dtypes.union(out_dtypes)

    fallbackcode_due_to_types = len(all_dtypes) != 1

    if node.label == "_Add__split_0":
        if ttype == tutil.TaskletType.UNARY_SYMBOL:
            op = "vfill"
            ttype = tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT
        #raise Exception(c1, c2, op, lhs, rhs1, rhs2, ttype)
    if node.label == "_Mult__split_0":
        if ttype == tutil.TaskletType.UNARY_SYMBOL:
            op = "vfill"
            ttype = tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT


    def _str_to_float_or_str(s: Union[int, float, str, None]):
        """Convert string constants to float if possible."""
        if s is None:
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def _is_number(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _generate_code(rhs1_, rhs2_, const1_, const2_, lhs_, op_, tasklet: dace.nodes.Tasklet):
        """
        Generate the C++ vectorized code string using templates or fallbacks.

        Handles:
        - Array-array, array-scalar, scalar-array
        - Commutative and non-commutative ops
        - Single constant + array/scalar (or array/scalar + constant)
        - Fallback loops if operator not supported (hope compiler will do it)
        """

        # Get out edge and its dtype
        out_edges = state.out_edges(node)
        assert len(out_edges) == 1
        out_edge = out_edges[0]

        if out_edge.data.data is None:
            dtype_ = dace.dtypes.TYPECLASS_TO_STRING[vector_dtype]
        else:
            data_dtype = state.sdfg.arrays[out_edge.data.data].dtype
            dtype_ = dace.dtypes.TYPECLASS_TO_STRING[data_dtype]

        rhs_left = rhs1_ if rhs1_ is not None else const1_
        rhs_right = rhs2_ if rhs2_ is not None else const2_

        #if tasklet.label == "_Add__split_0":
        #    raise Exception(rhs1_, rhs2_, const1_, const2_, lhs_, op_, tasklet)

        # Multiple dtypes involved - fallback code should be used
        print(op_, templates)
        print(op_ in templates)

        if not fallbackcode_due_to_types:
            # Use template if available
            print("C1", op_, op_ in templates)
            if op_ == "vfill" or op_ == "vfillc":
                assert op_ in templates
            if op_ in templates:
                # One array + optional constant
                print("C1.1", rhs1_, rhs2_)
                if rhs1_ is None or rhs2_ is None:
                    rhs = rhs1_ if rhs1_ is not None else rhs2_
                    constant = const1_ if const1_ is not None else const2_
                    print("C1.1.1", constant)
                    if constant is None:
                        # Single array or repeated array case
                        if is_commutative:
                            tasklet.label = tasklet_prefix + tasklet.label
                            return templates[op_].format(rhs1=rhs,
                                                         rhs2=rhs,
                                                         lhs=lhs_,
                                                         op=op_,
                                                         vector_width=vw,
                                                         dtype=dtype_)
                        tasklet.label = tasklet_prefix + tasklet.label
                        return templates[op_].format(rhs1=rhs,
                                                     rhs2=rhs,
                                                     lhs=lhs_,
                                                     op=op_,
                                                     vector_width=vw,
                                                     dtype=dtype_)
                    else:
                        # Single array + constant
                        cop_ = None
                        if is_commutative or op_ == "=":
                            cop_ = op_ + "c"
                        elif constant == const1_:
                            cop_ = "c" + op_
                        else:
                            assert constant == const2_
                            cop_ = op_ + "c"
                        # Maybe this constant version is not implemented in templates
                        print("C1.1.2", constant)
                        print(cop_, cop_ in templates)
                        if cop_ in templates:
                            tasklet.label = tasklet_prefix + tasklet.label
                            return templates[cop_].format(rhs1=rhs,
                                                          constant=_str_to_float_or_str(constant),
                                                          lhs=lhs_,
                                                          op=op_,
                                                          vector_width=vw,
                                                          dtype=dtype_)

                else:
                    print("C1.2", rhs1_, rhs2_)
                    # Two arrays
                    tasklet.label = tasklet_prefix + tasklet.label
                    return templates[op_].format(rhs1=rhs1_,
                                                 rhs2=rhs2_,
                                                 lhs=lhs_,
                                                 op=op_,
                                                 vector_width=vw,
                                                 dtype=dtype_)


        comparison_suffix = "? 1.0 : 0.0" if op_ in {">", ">=", "<", "<=", "==", "!="} else ""
        code_lines = [f""]
        code_lines.append(f"for (int _vi = 0; _vi < {vw}; _vi += 1) {{")

        # Determine operand order
        lhs_expr = lhs_ + "[_vi]"
        rhs_left = rhs1_ if rhs1_ is not None else const1_
        rhs_right = rhs2_ if rhs2_ is not None else const2_
        OPERATORS = {"+", "-", "/", "*", "%", "&&", "||", "==", "!=", "<", "<=", ">", ">="}
        UNARY_OPERATORS = {"+", "!", "-", "sqrt", "exp", "log"}

        if rhs_left is None or rhs_right is None:
            if op not in UNARY_OPERATORS and op in OPERATORS:
                raise Exception(
                    f"Invalid operand configuration for fallback vectorization. {rhs_left}, {rhs_right}, {lhs_expr}, {op}"
                )

        state.sdfg.save("failing.sdfg")
        if ttype == tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
            if state.in_degree(tasklet) == 1 and state.out_degree(tasklet) == 1:
                oes = state.out_edges(tasklet)
                oe = oes[0]
                ies = state.in_edges(tasklet)
                ie = ies[0]
                state.remove_node(tasklet)
                state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(ie.data))
            else:
                raise Exception(f"V2 SoftHier backend does not support fallback code, types: {fallbackcode_due_to_types}, op: {op}, ttype:{ttype}, {tasklet}")
        else:
            raise Exception(f"V1 SoftHier backend does not support fallback code, types: {fallbackcode_due_to_types}, op: {op}, ttype:{ttype}, {tasklet}",
                            rhs1, rhs2, lhs, c1, c2)

        if rhs_left is None or rhs_right is None:
            rhs = rhs_left if rhs_left is not None else rhs_right
            const = const1_ if const1_ is not None else const2_
            if op_ in UNARY_OPERATORS:
                if rhs_left == const:
                    code_lines.append(f"{lhs_expr} = {op_}{rhs}{comparison_suffix};")
                else:
                    code_lines.append(f"{lhs_expr} = {op_}({rhs}[_vi]){comparison_suffix};")
            elif op_ == "=":
                if rhs_left == const1_:
                    code_lines.append(f"{lhs_expr} = {rhs};")
                else:
                    code_lines.append(f"{lhs_expr} = {rhs}[_vi];")
            else:
                if rhs_left == const:
                    code_lines.append(f"{lhs_expr} = {op_}({rhs}){comparison_suffix};")
                else:
                    code_lines.append(f"{lhs_expr} = {op_}({rhs}[_vi]){comparison_suffix};")
        else:
            if op_ in OPERATORS:
                if rhs_left == const1_:
                    code_lines.append(f"{lhs_expr} = ({rhs_left} {op_} {rhs_right}[_vi]){comparison_suffix};")
                elif rhs_right == const2_:
                    code_lines.append(f"{lhs_expr} = ({rhs_left}[_vi] {op_} {rhs_right}){comparison_suffix};")
                else:
                    code_lines.append(f"{lhs_expr} = ({rhs_left}[_vi] {op_} {rhs_right}[_vi]){comparison_suffix};")
            else:
                if rhs_left == const1_:
                    code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}, {rhs_right}[_vi])){comparison_suffix};")
                elif rhs_right == const2_:
                    code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}[_vi], {rhs_right})){comparison_suffix};")
                else:
                    code_lines.append(f"{lhs_expr} = ({op_}({rhs_left}[_vi], {rhs_right}[_vi])){comparison_suffix};")

        code_lines.append("}")
        return "\n".join(code_lines)

    def _set_template(rhs1_, rhs2_, const1_, const2_, lhs_, op_, ttype, tasklet):
        """Helper to set tasklet code from template/fallback."""
        node.code = dace.properties.CodeBlock(code=_generate_code(rhs1_, rhs2_, _str_to_float_or_str(const1_),
                                                                  _str_to_float_or_str(const2_), lhs_, op_, tasklet),
                                              language=dace.Language.CPP)

    # Cast python boolean to C++ compatible string
    if c1 == "False":
        c1 = "0"
    if c1 == "True":
        c1 = "1"
    if c2 == "False":
        c2 = "0"
    if c2 == "True":
        c2 = "1"

    # Dispatch based on tasklet type
    if ttype == tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        _set_template(rhs1, rhs2, c1, c2, lhs, "=", ttype, node)
    elif ttype == tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT:
        raise Exception("0", c1, c2)
        if _is_number(str(c2)):
            _set_template(None, None, c2, None, lhs, "fillc", ttype, node)
        else:
            raise Exception("ARRAY SCALAR ASSIGNMETN WITHOUT NUMBER")
        #node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c2};" for i in range(vw)]) + "\n",
        #                                      language=dace.Language.CPP)
    elif ttype == tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT:
        # It is either a symbol or a constant
        #raise Exception("1", c1, c2)
        assert c2 is None
        if _is_number(str(c1)):
            _set_template(None, None, c1, None, lhs, "fill", ttype, node)
        else:
            raise Exception("ARRAY SYMBOL WITHOUT NUMBER")
            #node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c1}_laneid_{i};"
            #                                                      for i in range(vw)]) + "\n",
            #                                      language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.ARRAY_SYMBOL, tutil.TaskletType.ARRAY_ARRAY}:
        if ttype == tutil.TaskletType.ARRAY_SYMBOL:
            if op == "=":
                raise Exception("2", c1, c2)
        _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype, node)
    elif ttype in {tutil.TaskletType.UNARY_ARRAY}:
        arr_name = rhs1 if rhs1 is not None else rhs2
        occurences = tutil.count_name_occurrences(node.code.as_string.split(" = ")[1].strip(), arr_name)
        assert occurences == 1
        if op == "-":
            # Implement (-A) as (0 - A)
            _set_template(None, arr_name, "0.0", None, lhs, op, tutil.TaskletType.ARRAY_SYMBOL, node)
        elif op == "+":
            raise Exception("Unary + operator is not supported")
        else:
            _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype, node)
    elif ttype in {
            tutil.TaskletType.SCALAR_ARRAY,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(None, rhs2, rhs1, None, lhs, op, ttype, node)
    elif ttype in {
            tutil.TaskletType.ARRAY_SCALAR,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(rhs1, None, None, rhs2, lhs, op, ttype, node)
    elif ttype == tutil.TaskletType.SCALAR_SYMBOL:
        code_lines = []
        symbols = state.symbols_defined_at(node)
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        c = c1 if c1 is not None else c2
        for i in range(vw):
            expr = f"({l_op} {op} {r_op})"
            if str(c) in symbols:
                expr = offset_symbol_in_expression(expr, vector_map_param, i)
            else:
                if l_op == c:
                    expr = f"({l_op} {op} {r_op})"
                else:
                    expr = f"({l_op} {op} {r_op}{i})"
            code_lines.append(f"{lhs}[{i}] = {expr}")
        node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.SCALAR_SCALAR:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        expr = f"({l_op} {op} {r_op})"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {expr}" for i in range(vw)]) + "\n",
                                                  language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.SYMBOL_SYMBOL:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        r_op = rhs2 if rhs2 is not None else c2
        c = c1 if c1 is not None else c2
        expr = f"({l_op} {op} {r_op})"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(
                code="\n".join([f"{lhs}[{i}] = {use_laneid_symbol_in_expression(expr, c, i)}"
                                for i in range(vw)]) + "\n",
                language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}\n", language=dace.Language.Python)
    elif ttype == tutil.TaskletType.UNARY_SCALAR or ttype == tutil.TaskletType.UNARY_SYMBOL:
        out_edges = list(state.out_edges_by_connector(node, lhs))
        assert len(out_edges) == 1
        lhs_data = state.sdfg.arrays[out_edges[0].data.data]
        l_op = rhs1 if rhs1 is not None else c1
        expr = f"{op}{l_op}"
        if isinstance(lhs_data, dace.data.Array):
            node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {expr}" for i in range(vw)]) + "\n",
                                                  language=dace.Language.Python)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr}\n", language=dace.Language.Python)
    else:
        state.sdfg.save("failing.sdfg")
        raise NotImplementedError(f"Unhandled TaskletType: {ttype}, from: {node.code.as_string} ({node})")


def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode, vector_width: int,
                     vector_map_param: str) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
    """
    Duplicates an access node into a packed vector of a given width, updating all relevant tasklets and memlets.
    It writes to a packed storage by using the duplicated symbols.

    Args:
        state: The SDFG state containing the node.
        node: The AccessNode to duplicate.
        vector_width: Number of elements to pack.

    Returns:
        A tuple of sets: touched nodes and touched edges created during duplication.
    """

    touched_nodes = set()
    touched_edges = set()

    ies = state.in_edges(node)
    assert len(ies) == 1
    ie = ies[0]
    src = ie.src
    assert isinstance(src, dace.nodes.Tasklet), f"Writes to sink nodes need to go through assignment tasklets, do it"
    inc = next(iter(src.in_connectors))
    outc = next(iter(src.out_connectors))
    if src.code.as_string != f"{outc} = {inc}":
        # If prev tasklet is not assignment then add an intermediate scalar
        scl_name, scl = state.sdfg.add_scalar("tmp",
                                              dtype=state.sdfg.arrays[node.data].dtype,
                                              storage=dace.dtypes.StorageType.Register,
                                              transient=True,
                                              find_new_name=True)
        scl_an = state.add_access(scl_name)
        e = state.add_edge(src, ie.src_conn, scl_an, None, dace.memlet.Memlet(scl_name))
        state.remove_edge(ie)
        assign_tasklet = state.add_tasklet("assign_t", {"_in"}, {"_out"}, "_out = _in")
        e2 = state.add_edge(scl_an, None, assign_tasklet, "_in", dace.memlet.Memlet(scl_name))
        e3 = state.add_edge(assign_tasklet, "_out", ie.dst, ie.dst_conn,
                            dace.memlet.Memlet(data=ie.data.data, subset=copy.deepcopy(ie.data.subset)))
        # These edges and nodes still need to be vectorized
        #touched_edges.add(e)
        #touched_edges.add(e2)
        #touched_edges.add(e3)
        #touched_nodes.add(scl_an)
        #touched_nodes.add(assign_tasklet)
        # Update ndoes/edges
        src = assign_tasklet
        ie = e3
        inc = next(iter(src.in_connectors))
        outc = next(iter(src.out_connectors))

    assert src.code.as_string == f"{outc} = {inc}", f"{src.code.as_string} != {inc} = {outc}"

    src.code = CodeBlock(code="\n".join([f"{outc}[{_i}] = {inc}[{_i}]" for _i in range(vector_width)]))
    touched_nodes.add(src)
    packed_access = state.add_access(f"{node.data}_packed")
    #packed_access.setzero = True
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

        new_subset = repl_subset_to_use_laneid_offset(state.sdfg, ie.data.subset, str(i), vector_map_param)

        e2 = state.add_edge(t, "_out", ie.dst, None, dace.memlet.Memlet(data=node.data, subset=new_subset))
        if isinstance(e2, dace.nodes.Node):
            assert False
        touched_edges.add(e2)

    return touched_nodes, touched_edges


def replace_arrays_with_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                  new_type: typeclass) -> None:
    """
    Replaces existing arrays in an SDFG with new shapes (and optionally a new dtype).

    Args:
        sdfg: The SDFG containing the arrays.
        array_namelist: Set of array names to replace.
        new_shape: The new shape for the arrays.
        new_type: Optional new data type for arrays.
    """
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(name=arr_name,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype if new_type is None else new_type,
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
    """
    Creates copies of existing arrays with a new shape and a name suffix.

    Args:
        sdfg: The SDFG containing the arrays.
        array_namelist: Set of array names to copy.
        new_shape: Shape of the new arrays.
        name_suffix: Suffix to append to new array names.
    """
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


def get_scalar_source_nodes(
    sdfg: dace.SDFG, non_transient_only: bool,
    skip: Set[str] = set()) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns source nodes (in-degree 0 access nodes) for scalars (or shape-1 arrays) with no incoming edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient scalars.

    Returns:
        List of tuples (state, AccessNode).
    """

    source_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.in_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, )):
                    if non_transient_only is False or arr.transient is False:
                        if node.data not in skip:
                            source_nodes.append((state, node))
    return source_nodes


def get_array_source_nodes(sdfg: dace.SDFG,
                           non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns source nodes for arrays with more than one element (shape != (1,)) and no incoming edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient arrays.

    Returns:
        List of tuples (state, AccessNode).
    """

    source_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.in_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if (isinstance(arr, dace.data.Array) and (arr.shape != (1, ) and arr.shape != [
                        1,
                ])):
                    if non_transient_only is False or arr.transient is False:
                        source_nodes.append((state, node))
    return source_nodes


def get_scalar_sink_nodes(sdfg: dace.SDFG, non_transient_only: bool,
                          skip: Set[str]) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns sink nodes for scalars (or shape-1 arrays) with no outgoing edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient scalars.

    Returns:
        List of tuples (state, AccessNode).
    """

    sink_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.out_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or isinstance(arr, dace.data.Array) and arr.shape == (1, ):
                    if non_transient_only is False or arr.transient is False:
                        if node.data not in skip:
                            sink_nodes.append((state, node))
    return sink_nodes


def get_array_sink_nodes(sdfg: dace.SDFG,
                         non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns sink nodes for arrays with shape > 1 and no outgoing edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient arrays.

    Returns:
        List of tuples (state, AccessNode).
    """
    sink_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.out_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Array) and arr.shape != (1, ):
                    if non_transient_only is False or arr.transient is False:
                        sink_nodes.append((state, node))
    return sink_nodes


def add_transient_arrays_from_list(sdfg: dace.SDFG, arr_name_shape_storage_dtype: Iterable[Tuple[str, Any, Any,
                                                                                                 Any]]) -> None:
    """
    Adds transient arrays to an SDFG given a list of (name, shape, storage, dtype) tuples.

    Args:
        sdfg: The SDFG to modify.
        arr_name_shape_storage_dtype: Iterable of array specifications.
    """

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
    """
    Checks if a tasklet is a simple assignment (one input to one output).
    Checks `a = b` or `a = b;`
    Args:
        node: The Tasklet to check.

    Returns:
        True if it is a single assignment tasklet, False otherwise.
    """
    if (len(node.in_connectors) == 1 and len(node.out_connectors) == 1):
        in_conn = next(iter(node.in_connectors.keys()))
        out_conn = next(iter(node.out_connectors.keys()))
        return (node.code.as_string == f"{out_conn} = {in_conn}" or node.code.as_string == f"{out_conn} = {in_conn};")
    return False


def check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg: dace.SDFG,
                                                                scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                                                                              dace.nodes.AccessNode]]):
    """
    Ensures all writes to scalar sink nodes occur through simple assignment tasklets.
    Assignments can also occur through AccessNode -Edge-> AccessNode where `other_subset` is not none.
    Auto-vectorization does not support that.

    Args:
        sdfg: The SDFG to check.
        scalar_sink_nodes: List of scalar sink nodes to validate.

    Raises:
        Exception if a scalar sink write is not via an assignment tasklet.
    """
    for state, sink_node in scalar_sink_nodes:
        in_edges = state.in_edges(sink_node)
        if len(in_edges) != "1":
            raise Exception("All scalar sink nodes should have at max 1 incoming edge")
        in_edge = in_edges[0]
        src = in_edge.src
        if not (isinstance(src, dace.nodes.Tasklet) and is_assignment_tasklet(src)):
            raise Exception("All write to scalar should happen through an assignment tasklet")


def only_one_flop_after_source(state: dace.SDFGState, node: dace.nodes.AccessNode):
    """
    Checks whether only one computational tasklet (non-assignment) occurs after a given source node.
    Does BFS starting from the access node.

    Args:
        state: The SDFG state containing the node.
        node: The source AccessNode.

    Returns:
        Tuple (bool, List of nodes) indicating if the condition holds and the nodes checked.
    """

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
    """
    Checks if a transient accumulator is initialized to zero and used in an in-place reduction pattern.
    `nsdfg` is the parent nsdfg node and the state is where the nsdfg resides in.

    It traverses the nsdfg node backwards using the find a zero-assignment to the accumulator.
    The accumulator is the `source_node.data`. For it to be an accumulator source and sink needs to be the
    same too.

    Args:
        state: The parent SDFG state.
        nsdfg: The NestedSDFG node.
        inner_state: Inner state of the NestedSDFG.
        source_node: Source access node feeding the accumulator.
        sink_node: Sink access node consuming the accumulator.

    Returns:
        Tuple (bool, accumulator_name) indicating if the pattern is valid and the accumulator's name.
    """

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
        #print(f"{src_acc_node} of the memlet path {mpath} is not an access node")
        return False, ""

    # Ensure the access node directly connects to a memset-0 tasklet
    if state.in_degree(src_acc_node) != 1:
        #print(f"In degree of {src_acc_node} not one")
        return False, ""

    in_tasklet = state.in_edges(src_acc_node)[0].src
    if not isinstance(in_tasklet, dace.nodes.Tasklet):
        #print(f"In neighbor {in_tasklet} is not a tasklet")
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
    """
    Replaces all memlet subsets for a given array in a state with a new subset expression.

    Args:
        state: The SDFG state to modify.
        name: Array name whose accesses are replaced.
        new_subset_expr: The new subset expression (e.g., "0:4").
    """
    for edge in state.edges():
        if edge.data is not None and edge.data.data == name:
            nm = dace.memlet.Memlet(expr=f"{name}[{new_subset_expr}]")
            edge.data = nm


def expand_assignment_tasklets(state: dace.SDFGState, name: str, vector_width: int):
    """
    Expands assignment tasklets writing to an array at a to be over the vector length
    over the unit stride dimension a[0] = ..., a[1] = ..., ...
    For assignment tasklets the dataname given as name.

    Args:
        state: The SDFG state to modify.
        name: The array being written.
        vector_width: Length of the vector to expand to.
    """
    for e in state.edges():
        if (isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == name and isinstance(e.src, dace.nodes.Tasklet)):
            code = e.src.code
            in_conns = e.src.in_connectors
            out_conns = e.src.out_connectors
            if len(in_conns) != 0:
                assert False, "Non-assignemnt tasklet found for accumulator, unsupported case"
            assert len(out_conns) == 1, f"{out_conns}"
            out_conn = next(iter(out_conns))
            assert code.language == dace.dtypes.Language.Python
            assert code.as_string.startswith(f"{out_conn} =")
            rhs = code.as_string.split("=")[-1].strip()
            ncode_str = "\n".join([f"{out_conn}[{i}] = {rhs}" for i in range(vector_width)])
            e.src.code = dace.properties.CodeBlock(ncode_str)


def reduce_before_use(state: dace.SDFGState, name: str, vector_width: int, op: str):
    """
    Adds a reduction tasklet to reduce a vectorized array into a scalar before its use.

    Args:
        state: The SDFG state.
        name: Array to reduce.
        vector_width: Number of vector elements.
        op: Reduction operation (e.g., "+", "*").
    """
    # TODO: Reduction can be optimized (e.g. logarithmic depth or checking of vector templates have a reduction op)

    # Any time a tasklet reads name[0:vector_width] then we need to reduce it before
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
            #an.setzero = True
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
                       vector_width) -> Tuple[bool, str, str]:
    """
    Moves a reduction out of a NestedSDFG, vectorizing transient accumulators and adjusting tasklets.

    This function is typically used when a computation pattern consists of:
      1. A scalar source feeding a NestedSDFG,
      2. A transient accumulator initialized to zero (outside nested SDFG)
      3. A single computational tasklet updating the accumulator (e.g. acc = acc + some_var)
      4. A scalar sink at the end of the nsdfg for the accumulator.

    The transformation performs the following steps:
        1. Checks that there is at most one floating-point operation after the source. (For condition 3)
        2. Validates that the accumulator is a transient scalar initialized to zero. (For condition 1 and 2)
        3. Extracts the operation performed on the accumulator (e.g., addition, multiplication).
        4. Reshapes the source, sink, and accumulator arrays to a vectorized form of size `vector_width`.
        5. Updates all memlets accessing the accumulator to cover the full vector range.
        6. Expands assignment tasklets to operate on all vector elements.
        7. Inserts a reduction tasklet that combines the vector elements back to a scalar before use.

    Args:
        scalar_source_nodes: List of tuples `(state, node)` representing source scalar nodes feeding the NestedSDFG.
        state: Parent SDFGState containing the NestedSDFG node.
        nsdfg: NestedSDFG node where the reduction occurs.
        inner_sdfg: Inner SDFG of the NestedSDFG.
        vector_width: The width of vectorization for the accumulator.

    Notes:
        - Only supports simple reduction patterns with one operation and transient accumulators.
        - The function assumes that the scalar source and sink nodes are properly connected through the NestedSDFG.
        - The reduction operation is extracted automatically from the first tasklet after the source.

    """
    num_flops, node_path = only_one_flop_after_source(scalar_source_nodes[0][0], scalar_source_nodes[0][1])
    is_inout_accumulator, accumulator_name = input_is_zero_and_transient_accumulator(
        state, nsdfg, scalar_source_nodes[0][0], scalar_source_nodes[0][1], node_path[-1])
    op = tutil._extract_single_op(node_path[1].code.as_string)
    #print(is_inout_accumulator, num_flops, accumulator_name)
    if num_flops <= 1 and is_inout_accumulator:
        source_data = scalar_source_nodes[0][1].data
        sink_data = node_path[-1].data
        replace_arrays_with_new_shape(inner_sdfg, {source_data, sink_data}, (vector_width, ), None)
        replace_arrays_with_new_shape(state.sdfg, {accumulator_name}, (vector_width, ), None)
        replace_all_access_subsets(state, accumulator_name, f"0:{vector_width}")
        expand_assignment_tasklets(state, accumulator_name, vector_width)
        reduce_before_use(state, accumulator_name, vector_width, op)

        return True, source_data, sink_data
    return False, source_data, sink_data


def assert_symbols_in_parent_map_symbols(missing_symbols: Set[str], state: dace.SDFGState,
                                         nsdfg: dace.nodes.NestedSDFG):
    """
    Validates that given symbols correspond to loop variables in parent map scopes of a NestedSDFG.

    Args:
        missing_symbols: Symbols to validate (e.g., {"i_laneid_0", "j_laneid_1"}).
        state: The SDFG state.
        nsdfg: NestedSDFG node.

    Returns:
        Set of loop variable names found in the parent scopes.

    Raises:
        AssertionError if a symbol is not found in the loop scopes.
    """

    def validate_and_strip(strings):
        valid = []
        for s in strings:
            match = re.fullmatch(r'([A-Za-z_]\w*?)(\d+)$', s)
            assert match, f"No match in {strings} for a variable name"
            if match:
                name, num = match.groups()
                valid.append((name, int(num)))
        return valid

    stripped_symbols = validate_and_strip(missing_symbols)
    loop_vars = {var for var, int_id in stripped_symbols}

    sdict = state.scope_dict()
    first_parent_map = sdict[nsdfg]
    parent_maps_and_loops = cutil.get_parent_map_and_loop_scopes(state.sdfg, first_parent_map, state)

    loop_symbols = set()
    for p in first_parent_map.map.params:
        loop_symbols.add(p)

    for map_or_loop in parent_maps_and_loops:
        if isinstance(map_or_loop, dace.nodes.MapEntry):
            for p in map_or_loop.map.params:
                loop_symbols.add(p)
        elif isinstance(map_or_loop, LoopRegion):
            loop_symbols.add(map_or_loop.loop_variable)

    for loop_var in loop_vars:
        loop_var = loop_var[:-len("_laneid_")] if loop_var.endswith("_laneid_") else loop_var
        assert loop_var in loop_symbols, f"{loop_var} not in {loop_symbols}"

    return loop_vars


def find_symbol_assignment(sdfg: dace.SDFG, sym_name: str) -> str:
    """
    Finds the assignment expression of a given symbol by traversing the SDFG backwards.

    Args:
        sdfg: The SDFG to search.
        sym_name: Symbol to find.

    Returns:
        Assignment expression as a string, or None if not found.
    """

    # Pre-condition for vectorization
    assert all({isinstance(s, dace.SDFGState) for s in sdfg.nodes()})
    sink_state = {s for s in sdfg.nodes() if sdfg.out_degree(s) == 0}.pop()
    edges_to_check = sink_state.parent_graph.in_edges(sink_state)
    while edges_to_check:
        edge = edges_to_check.pop()

        for k, v in edge.data.assignments.items():
            if k == sym_name:
                return v

        edges_to_check += sink_state.parent_graph.in_edges(edge.src)

    return None
    #raise Exception("Symbol assignment not found")


def _all_atoms(expr, ignored=()):
    """
    Return a set of all atomic elements in a SymPy expression, including:
    - Symbols
    - Indexed symbols / arrays
    - Function calls
    - Numbers (optional)

    ignored: tuple of types to ignore, e.g., (sympy.Number,)
    """
    # Use expr.atoms to get all different types of atoms
    atoms = set()

    # Get all symbols
    atoms.update(expr.atoms(sympy.Symbol))

    # Get all Indexed (arrays)
    atoms.update(expr.atoms(sympy.Indexed))

    # Get all function symbols (but not the class, only instances)
    funcs = expr.atoms(sympy.Function)
    for f in funcs:
        if f.func not in ignored:
            atoms.add(f)
            # Also include arguments of the function
            atoms.update(f.args)

    return atoms


def collect_vectorizable_arrays(sdfg: dace.SDFG, parent_nsdfg_node: dace.nodes.NestedSDFG, parent_state: SDFGState,
                                invariant_scalars: Set[str]) -> Set[str]:
    """
    Determines which arrays can be vectorized based on their access patterns and symbol usage.
    The symbols used for accessing should not have any indirectness, meaning that they should
    not be accessing other Arrays on interstate assignemnts, this is expressed as a free function
    in sympy.

    The map parameter involve in vectorization should not appear in a multiplicaiton expression.
    E.g. loop (int i = 0; i < N; i ++) and access A[i] is ok but, A[i*2] means it is strided and it
    needs to be packed

    Consider the case A[for_it_88, 0, jo] and interstate assignment has jo = B[for_it_88, 0]
    And the loop is over 0->for_it_88, this not vectorizable, so if any dimension involved uses the loop map
    param return false

    Args:
        sdfg: The SDFG to analyze.
        parent_nsdfg_node: NestedSDFG node.
        parent_state: State containing the NestedSDFG.
        invariant_scalars: Set of scalar names that are invariant across lanes (means these
            scalars to do not prevent vectorization)

    Returns:
        Dictionary mapping array names to a boolean indicating vectorizability.
    """
    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = parent_state.scope_dict()[parent_nsdfg_node]
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]

    all_accesses_to_arrays = collect_accesses_to_array_name(sdfg)
    #print(all_accesses_to_arrays)

    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    state.sdfg.save("d.sdfg")

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # Get the stride 1 dimension
            stride_one_dims = {i for i, stride in enumerate(sdfg.arrays[arr_name].strides) if stride == 1}
            if len(stride_one_dims) == 0:
                array_is_vectorizable[arr_name] = False
                break

            stride_one_dim = stride_one_dims.pop()
            b, e, s = access_subset[stride_one_dim]
            assert b == e
            assert s == 1

            # Evaluate the expression (b == e)
            access_expr = b  # use b since b==e
            #print(access_expr, type(access_expr))
            #print(isinstance(access_expr, (dace.symbolic.SymExpr, dace.symbolic.symbol, sympy.Expr)))
            if isinstance(access_expr, (dace.symbolic.SymExpr, sympy.Expr)):
                # Check for multipliers
                # If map_param appears multiplied in the expression, it is strided
                free_syms = {str(s) for s in access_expr.free_symbols}
                #print(free_syms)
                #if map_param in free_syms:
                #    print(f"Map param {map_param} in free syms {free_syms} of {access_expr}")
                # If map param appears in a multiplication expression then it is not vectorizable
                if len({
                        term
                        for term in access_expr.atoms(sympy.Mul)
                        if isinstance(term, sympy.Mul) and map_param in free_syms
                }) > 0:
                    array_is_vectorizable[arr_name] = False
                    raise Exception("TODO - I have not analyzed this case yet")

            if isinstance(b, (dace.symbolic.SymExpr, dace.symbolic.symbol)):
                if isinstance(b, dace.symbolic.SymExpr):
                    free_syms = {str(s) for s in b.free_syms}
                else:
                    free_syms = {b}
                for free_sym in free_syms:
                    # Accessing map param is ok
                    if str(free_sym) == map_param:
                        continue
                    else:
                        # Other free symbols should not have indirect accesses
                        # Analysis tries find the first assignment in the CFG
                        assignment = find_symbol_assignment(sdfg, str(free_sym))
                        if assignment is None:
                            sdfg.save("failing_vectorization.sdfg")
                        assert assignment is not None, f"Could not find an iedge assignment for {free_sym}"
                        assignment_expr = dace.symbolic.SymExpr(assignment)
                        # Define functions to ignore (common arithmetic + piecewise + rounding)
                        ignored = {
                            sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs, sympy.floor,
                            sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan, sympy.sinh,
                            sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                        }
                        #print("A:", assignment)

                        # Collect only user-defined or nonstandard functions - in intersate edge this means array accees
                        funcs = {f.name for f in assignment_expr.atoms(sympy.Function) if f.func not in ignored}
                        # Any array on the right-hand-side -> big problem
                        # Check for scalar / array accesses like this too
                        scalars = {str(s)
                                   for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars
                        # If scalar is invariant it should be ok?
                        #print("Invariant", invariant_scalars)
                        #print("Non-invariant scalars",
                        #      {s
                        #       for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars)
                        if len(funcs) != 0 or len(scalars) != 0:
                            #print(f"Indirect access detected: ({funcs}, {scalars}) for {arr_name}, is not vectorizable")
                            array_is_vectorizable[arr_name] = False

            # Go through non unit stride dimensions in case it those dimensions have unstructuredness
            for i, (b, e, s) in enumerate(access_subset):
                #print(i, ",", (b,e,s), "|", access_subset)
                if i == stride_one_dim:
                    continue
                #print(b, type(b),)
                free_syms = set()
                if hasattr(b, "free_syms"):
                    free_syms = {str(s) for s in b.free_syms}
                if hasattr(b, "free_symbols"):
                    free_syms = {str(s) for s in b.free_symbols}

                if free_syms != set():
                    #print(free_syms)
                    for free_sym in free_syms:
                        # Accessing map param is ok
                        #print("FS", free_syms)
                        if str(free_sym) == map_param:
                            continue
                        else:
                            # Other free symbols should not have indirect accesses
                            # Analysis tries find the first assignment in the CFG
                            assignment = find_symbol_assignment(sdfg, str(free_sym))

                            # If assignment is None, it is probably coming from parent map
                            parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)
                            if assignment is None:
                                assert str(
                                    free_sym
                                ) in parent_syms_defined, f"Could not find an iedge assignment for {free_sym} it is also not defined in symbols defined in nsdfg entry {parent_syms_defined}"
                                continue

                            assignment_expr = dace.symbolic.SymExpr(assignment)
                            # Define functions to ignore (common arithmetic + piecewise + rounding)
                            ignored = {
                                sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs,
                                sympy.floor, sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan,
                                sympy.sinh, sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                            }
                            all_atoms = _all_atoms(assignment_expr, ignored)
                            all_atoms_str = {str(s) for s in all_atoms}
                            #print(all_atoms_str)

                            # Map parameter appears in inddirect access, array is not vectorizable
                            if map_param in all_atoms_str:
                                array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_non_unit_stride_accesses_in_map(sdfg: dace.SDFG, state: dace.SDFGState,
                                            map_entry: dace.nodes.MapEntry) -> Set[str]:
    """
    Determines which arrays can be vectorized based on their access patterns and symbol usage.
    The symbols used for accessing should not have any indirectness, meaning that they should
    not be accessing other Arrays on interstate assignemnts, this is expressed as a free function
    in sympy.

    The map parameter involve in vectorization should not appear in a multiplicaiton expression.
    E.g. loop (int i = 0; i < N; i ++) and access A[i] is ok but, A[i*2] means it is strided and it
    needs to be packed

    Args:
        sdfg: The SDFG to analyze.
        parent_nsdfg_node: NestedSDFG node.
        parent_state: State containing the NestedSDFG.
        invariant_scalars: Set of scalar names that are invariant across lanes (means these
            scalars to do not prevent vectorization)

    Returns:
        Dictionary mapping array names to a boolean indicating vectorizability.
    """
    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = map_entry
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]

    # Collect all subsets
    all_nodes = state.all_nodes_between(parent_map, state.exit_node(parent_map))
    all_accesses_to_arrays = {e.data.data: list() for e in state.all_edges(*all_nodes) if e.data.data is not None}
    for e in state.all_edges(*all_nodes):
        if e.data.data is not None:
            all_accesses_to_arrays[e.data.data].append(e.data.subset)

    for edge in state.all_edges(*all_nodes):
        if edge.data.other_subset is not None:
            raise NotImplementedError("other subset support not implemented")

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    # Since no nestedSDFG, no indirectness may occur just check stridedness

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # Get the stride 1 dimension
            stride_one_dim = {i for i, stride in enumerate(sdfg.arrays[arr_name].strides) if stride == 1}.pop()
            b, e, s = access_subset[stride_one_dim]
            assert b == e
            assert s == 1

            # Evaluate the expression (b == e)
            access_expr = b  # use b since b==e
            #print(access_expr, type(access_expr))
            #print(isinstance(access_expr, (dace.symbolic.SymExpr, dace.symbolic.symbol, sympy.Expr)))
            if isinstance(access_expr, (dace.symbolic.SymExpr, sympy.Expr)):
                # Check for multipliers
                # If map_param appears multiplied in the expression, it is strided
                free_syms = {str(s) for s in access_expr.free_symbols}
                #print(free_syms)
                #if map_param in free_syms:
                #    print(f"Map param {map_param} in free syms {free_syms} of {access_expr}")
                # If map param appears in a multiplication expression then it is not vectorizable
                if len({
                        term
                        for term in access_expr.atoms(sympy.Mul)
                        if isinstance(term, sympy.Mul) and map_param in free_syms
                }) > 0:
                    array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_accesses_to_array_name(sdfg: dace.SDFG) -> Dict[Tuple[str, dace.subsets.Range], str]:
    """
    Collects all access subsets for each array in the SDFG.

    Args:
        sdfg: The SDFG to analyze.

    Returns:
        Dictionary mapping array names to a set of accessed subsets.
    """
    d = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")
            if edge.data.data is not None:
                if edge.data.data not in d:
                    d[edge.data.data] = set()
                d[edge.data.data].add(edge.data.subset)
    return d


# Allowed standard and math-like functions
STANDARD_FUNCS = {
    # Builtins and math intrinsics
    "abs",
    "exp",
    "log",
    "ln",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "pow",
    "min",
    "max",
    # DaCe-like or symbolic math intrinsics
    "int_floor",
    "int_ceil",
    "div_floor",
    "div_ceil",
} | set(dir(math))


class FuncToSubscript(ast.NodeTransformer):

    def visit_Call(self, node):
        # Recurse into child nodes
        self.generic_visit(node)

        func = node.func
        func_name = None

        # Handle simple functions like foo(...)
        if isinstance(func, ast.Name):
            func_name = func.id

        # Handle qualified names like math.sin or np.exp
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                # Keep math.*, np.*, numpy.* untouched
                if func.value.id in ("math", "np", "numpy"):
                    func_name = f"{func.value.id}.{func.attr}"
                else:
                    func_name = func.attr

        # Skip conversion if function is standard
        if func_name and func_name.split(".")[-1] in STANDARD_FUNCS:
            return node

        # Otherwise, convert func(a, b) → func[a, b]
        if len(node.args) == 1:
            new_slice = node.args[0]
        else:
            new_slice = ast.Tuple(elts=node.args, ctx=ast.Load())

        return ast.Subscript(value=func, slice=new_slice, ctx=ast.Load())


def convert_nonstandard_calls(expr: str) -> str:
    tree = ast.parse(expr, mode="eval")
    new_tree = FuncToSubscript().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def expand_interstate_assignments_to_lanes(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                           state: dace.SDFGState, vector_width: int, invariant_data: Set[str],
                                           vector_dtype: typeclass,
                                           vector_map_param: str):
    # `sym = 0`
    # Would become
    # `sym_laneid_0 = 0, sym=sym_laneid_0, sym_laneid_1 = 0, sym_laneid_2 = 0, ....`
    # Assume:
    # `sym = A[_for_it] + 1`
    # Would become:
    # `sym_laneid_0 = A[_for_it + 0] + 1`, `sym = sym_laneid_0`, `sym_laneid_1 = A[_for_it + 1] + 1`, ...

    # Invariant data means that the data is constant across iterators
    # If all free symbols are from invariant data then duplication is not necessar

    # Pre-condition last dimension is the dimension we vectorize
    parent_map_entry = state.scope_dict()[nsdfg_node]
    assert parent_map_entry is not None and isinstance(parent_map_entry, dace.nodes.MapEntry)
    vectorized_param = vector_map_param
    #print(vector_map_param)

    for edge in inner_sdfg.all_interstate_edges():
        new_assignments = dict()
        assignments = edge.data.assignments
        #print(f"Work expanding interstate edge {edge.src} -> {edge.dst} ({edge.data.assignments})")
        for k, v in assignments.items():
            # Add a new symbol for each lane
            original_v_expr = dace.symbolic.SymExpr(v)
            #print(f"Update {k} assignments")
            for i in range(vector_width):
                # I do not know why is it happening but:
                # This function tries to apply expansion on something such as 'jo_laneid_3'
                # If 2 symbols need to be expanded then we get 'jo_laneid_3_laneid_0' (for vector length 8 we get 8x8 symbols that are wrong)
                # The update should be done only when both laneid's match and
                # in this case the anme should not be updated
                if "laneid" in k:
                    m = re.search(r"_(\d+)$", k)
                    assert m
                    idx = int(m.group(1))
                    if i != idx:
                        continue
                    new_k = k
                else:
                    new_k = f"{k}_laneid_{i}"
                #print(f"{new_k} is from {k}")
                v_expr = dace.symbolic.SymExpr(v)

                funcs = {str(f) for f in v_expr.atoms(sympy.Function)}
                non_func_free_syms = {str(s) for s in v_expr.free_symbols if str(s) not in funcs}
                # Collect functions that are accesses
                array_accesses = {f for f in funcs if f in inner_sdfg.arrays}
                variant_array_accesses = (array_accesses.union(non_func_free_syms)) - invariant_data

                if len(variant_array_accesses) == 0:
                    # Copy the old self
                    new_assignments[k] = v
                    continue

                if new_k not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(new_k, inner_sdfg.symbols.get(k, vector_dtype))

                # Replace map param `_for_it` with `_for_it + laneid`
                v_expr_old = copy.deepcopy(v_expr)
                v_expr = v_expr.subs(vectorized_param, f"({vectorized_param} + {i})")
                #print(v_expr_old, "->", v_expr)

                # All other symbols are replaced with `s` -> `s{laneid}`, except free symbols
                # Those are kept as they are
                non_map_free_syms = {str(s)
                                     for s in original_v_expr.free_symbols} - ({vectorized_param}.union(
                                         inner_sdfg.free_symbols))

                assert vectorized_param not in non_map_free_syms

                for free_sym in non_map_free_syms:
                    # The free_sym can be an array, scalar or symbol
                    free_sym_str = str(free_sym)
                    assert free_sym_str in inner_sdfg.arrays or free_sym_str in inner_sdfg.symbols

                    # If it is a symbol replace with laneid
                    if free_sym_str in inner_sdfg.symbols:
                        #print(f"Subs {free_sym} with {free_sym}_laneid_{i}")
                        if free_sym_str == vector_map_param:
                            assert False
                        else:
                            v_expr = v_expr.subs(free_sym, f"{free_sym}_laneid_{i}")

                            # Add the new symbol to the symbols
                            if f"{free_sym}_laneid_{i}" not in inner_sdfg.symbols:
                                inner_sdfg.add_symbol(f"{free_sym}_laneid_{i}",
                                                      inner_sdfg.symbols.get(str(free_sym), vector_dtype))
                    else:
                        if isinstance(inner_sdfg.arrays[free_sym_str], dace.data.Scalar):
                            #print(f"Subs {free_sym} with {free_sym}")
                            v_expr = v_expr.subs(free_sym, f"{free_sym}")
                        else:
                            assert inner_sdfg.arrays[free_sym_str].shape != (1, )
                            #print(f"Subs {free_sym} with {free_sym}({i})")
                            v_expr = v_expr.subs(free_sym, f"{free_sym}({i})")

                new_v = sympy.pycode(v_expr, allow_unknown_functions=True)
                new_v = tutil.rewrite_boolean_functions_to_boolean_ops(new_v)
                # Convert non-standard function calls back to array accesses before assigning
                new_assignments[new_k] = convert_nonstandard_calls(new_v)

                if i == 0:
                    # In case the symbol is used for something vectorizable - then this duplication was necessary
                    # Let's hope compiler will prune the unused data, have laneid 0 one for the original symbol too
                    new_assignments[k] = convert_nonstandard_calls(new_v)

        edge.data.assignments = new_assignments


def try_demoting_vectorizable_symbols(inner_sdfg: dace.SDFG) -> Set[str]:
    assigned_symbols = dict()
    for edge in inner_sdfg.all_interstate_edges():
        for k, v in edge.data.assignments.items():
            if k not in assigned_symbols:
                assigned_symbols[k] = set()
            assigned_symbols[k].add(v)

    demotable_symbols = set()
    for sym, sym_assignments in assigned_symbols.items():
        # Check that the access is to arrays and map param is involved
        all_function_args = set()
        #print(sym_assignments)
        for sym_assignment in sym_assignments:
            sym_assign_expr = dace.symbolic.SymExpr(sym_assignment)
            # Collect all array accesses (they are functions that are present in the sdfg)
            funcs = {(f.name, f) for f in sym_assign_expr.atoms(sympy.Function)}
            #print(funcs)
            for fname, f in funcs:
                #print(f"Check function: {fname} ({str(fname) in inner_sdfg.arrays})")
                if fname in inner_sdfg.arrays:
                    for arg in f.args:
                        all_function_args = all_function_args.union({str(s) for s in arg.free_symbols})

        # If all function args are s
        #print(f"{sym} <-(depends)- {all_function_args}")
        # if the depend set has no arrays or scalars we can do it
        data_in_dependence_set = {d for d in all_function_args if d in inner_sdfg.arrays}
        if len(data_in_dependence_set) == 0:
            demotable_symbols.add(sym)

    for demotable_symbol in demotable_symbols:
        stype = inner_sdfg.symbols[demotable_symbol]
        sdutil.demote_symbol_to_scalar(inner_sdfg, demotable_symbol, stype)

    return demotable_symbols


def collect_all_memlets_to_dataname(sdfg: dace.SDFG) -> Dict[str, Set[dace.subsets.Range]]:
    """
    Collect all unique memlet subsets for each data array in the SDFG.

    This function traverses all states and edges in the SDFG and groups memlet subsets
    by the data array they access. Does not check interstate edges or conditionals.

    Args:
        sdfg: The SDFG to analyze

    Returns:
        A dictionary mapping data array names to sets of their accessed subsets
    """
    dataname_to_memlets = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None:
                if edge.data.data not in dataname_to_memlets:
                    dataname_to_memlets[edge.data.data] = set()
                dataname_to_memlets[edge.data.data].add(edge.data.subset)

    return dataname_to_memlets


def is_vector_assign_tasklet(t: dace.nodes.Tasklet) -> bool:
    """
    Check if a tasklet performs a vector copy operation.

    Args:
        t: The tasklet to check

    Returns:
        True if the tasklet's code contains "vector_copy(", False otherwise
    """
    return "vector_copy(" in t.code.as_string


def insert_assignment_tasklet_from_src(state: dace.SDFGState, edge: Edge[Memlet],
                                       vector_storage_type: dace.dtypes.StorageType,
                                       vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """
    Insert a vector assignment tasklet after the source node of an edge.

    This function transforms:
        src --[memlet]--> dst
    Into:
        src --[memlet]--> copy_tasklet --> access_node[vector] --[memlet2]--> dst

    The tasklet performs a vector_copy operation, and a new transient vector array
    is created with the specified storage type and length.

    Args:
        state: The SDFG state containing the edge
        edge: The edge to transform
        vector_storage_type: Storage type for the new vector array (e.g., Register, FPGA_Local)
        vector_width: Length of the vector array

    Returns:
        A tuple of three new edges: (src->tasklet, tasklet->access, access->dst)

    Side effects:
        - Removes the original edge
        - Creates a new transient vector array if it doesn't exist
        - Adds a tasklet, access node, and three new edges
    """
    src = edge.src
    src_conn = edge.src_conn
    dst = edge.dst
    dst_conn = edge.dst_conn

    # Create or reuse vector array
    vector_dataname = edge.data.data + "_vec"
    if vector_dataname not in state.sdfg.arrays:
        orig_arr = state.sdfg.arrays[edge.data.data]
        arr_name, arr = state.sdfg.add_array(name=vector_dataname,
                                             shape=(vector_width, ),
                                             dtype=orig_arr.dtype,
                                             location=orig_arr.location,
                                             transient=True,
                                             find_new_name=False,
                                             storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(
        name="_AssignT3",
        inputs={"_in"},
        outputs={"_out"},
        code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vector_data.dtype]}, {vector_width}>(_out, _in);",
        language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    #an.setzero = True
    e1 = state.add_edge(src, src_conn, t, "_in", copy.deepcopy(edge.data))
    e2 = state.add_edge(t, "_out", an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(an, None, dst, dst_conn, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    state.remove_edge(edge)

    return (e1, e2, e3)


def insert_assignment_tasklet_to_dst(state: dace.SDFGState, edge: Edge[Memlet],
                                     vector_storage_type: dace.dtypes.StorageType,
                                     vector_width: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
    """
    Insert a vector assignment tasklet before the destination node of an edge.


    This function transforms:
        src --[memlet]--> dst
    Into:
        src --[memlet2]--> access_node[vector] --[memlet2]--> copy_tasklet --[memlet]--> dst


    The tasklet performs a vector_copy operation, and a new transient vector array
    is created with the specified storage type and length.

    Args:
        state: The SDFG state containing the edge
        edge: The edge to transform
        vector_storage_type: Storage type for the new vector array
        vector_width: Length of the vector array

    Returns:
        A tuple of three new edges: (src->access, access->tasklet, tasklet->dst)

    Side effects:
        - Removes the original edge
        - Creates a new transient vector array if it doesn't exist
        - Adds a tasklet, access node, and three new edges
    """
    src = edge.src
    src_conn = edge.src_conn
    dst = edge.dst
    dst_conn = edge.dst_conn

    # Create or reuse vector array
    vector_dataname = edge.data.data + "_vec"
    if vector_dataname not in state.sdfg.arrays:
        orig_arr = state.sdfg.arrays[edge.data.data]
        _, arr = state.sdfg.add_array(name=vector_dataname,
                                      shape=(vector_width, ),
                                      dtype=orig_arr.dtype,
                                      location=orig_arr.location,
                                      transient=True,
                                      find_new_name=False,
                                      storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(
        name="_AssignT4",
        inputs={"_in"},
        outputs={"_out"},
        code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vector_data.dtype]}, {vector_width}>(_out, _in);",
        language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    an.setzero = True
    e1 = state.add_edge(src, src_conn, an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e2 = state.add_edge(an, None, t, "_in", dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(t, "_out", dst, dst_conn, copy.deepcopy(edge.data))
    state.remove_edge(edge)

    return (e1, e2, e3)


def add_copies_before_and_after_nsdfg(state: SDFGState,
                                      nsdfg_node: dace.nodes.NestedSDFG,
                                      vector_width: int,
                                      vector_storage: dace.dtypes.StorageType,
                                      skip: Set[str],
                                      no_copy_out: bool = False) -> Set[str]:
    """
    Add vector copy operations before and after a nested SDFG node.
    If the copy can't be inserted before, then it is done inside as a fallback,

    This function analyzes data access patterns in a nested SDFG and determines which
    arrays can have their copies moved outside the nested SDFG for optimization. It
    handles two types of arrays:

    Skip set will result in the dataname to be not copied no matter what, it should be
    used for unstructured loads.

    1. Movable arrays: Arrays with uniform access patterns (structured and vectorizable)
       can be copied before/after the nested SDFG execution.

    2. Unmovable arrays: Arrays with unstructured access patterns or symbol dependencies
       that require copies to remain inside the nested SDFG.

    ----------------
    For movable arrays:
        MapEntry -(Array[0:N])-> NSDFG
        becomes:
        MapEntry -(Array[0:N])-> VecArray -(VecArray[0:vector_width])-> NSDFG

        or:

        MapEntry -(Array[i:i+vector_width])-> NSDFG
        becomes:
        MapEntry -(Array[i:i+vector_width])-> VecArray -(VecArray[0:vector_width])-> NSDFG

    For unmovable arrays:
        Assignment tasklets are inserted at each read/write point inside the nested SDFG.

    Args:
        state: The SDFG state containing the nested SDFG node
        nsdfg_node: The nested SDFG node to process
        vector_width: The width of vector operations
        vector_storage: Storage type for vector arrays (e.g., Register, FPGA_Local)

    Side effects:
        - Modifies the nested SDFG by adding assignment tasklets for unmovable arrays
        - Saves intermediate SDFG to "b.sdfg" for debugging
        - Calls process_in_edges and process_out_edges (which must be defined elsewhere)
    """

    # Fix offset bug here, test_snippet_from_cloudsc_three -> incorrect offests
    # Collect all arrays that are accessed in the nested SDFG
    inner_sdfg = nsdfg_node.sdfg
    dataname_to_subsets = collect_all_memlets_to_dataname(inner_sdfg)

    # Get read and write sets
    read_set, write_set = inner_sdfg.read_and_write_sets()

    # Filter to only non-transient arrays (inputs/outputs of the nested SDFG)
    dataname_to_subsets = {
        k: v
        for k, v in dataname_to_subsets.items() if k in inner_sdfg.arrays and inner_sdfg.arrays[k].transient is False
        and isinstance(inner_sdfg.arrays[k], dace.data.Array)
    }

    movable_arrays = set()
    unmovable_arrays = dict()

    # Classify arrays as movable or unmovable
    for dataname, memlets in dataname_to_subsets.items():
        if len(memlets) > 1:
            # Multiple distinct access patterns - can't safely move outside
            if dataname not in skip:
                unmovable_arrays[dataname] = set(memlets)
        else:
            # Single access pattern - check if symbols are available outside
            memlet = next(iter(memlets))
            memlet_syms = {str(s) for s in memlet.free_symbols}
            avaialble_syms = {str(s) for s in state.symbols_defined_at(nsdfg_node)}

            if all({s in avaialble_syms for s in memlet_syms}):
                if dataname not in skip:
                    movable_arrays.add((dataname, memlet))
            else:
                if dataname not in skip:
                    unmovable_arrays[dataname] = set(memlets)

    # Better analysis for this might be necessary (to make sure write happens through multiple laneid_* symbols)
    unstructured_load_arrays = set()
    for dataname, memlets in dataname_to_subsets.items():
        if len(memlets) == vector_width:
            unstructured_load_arrays.add(dataname)
        # Remove them from unmovable arrays (they are not in movable arrays either), as there is no need for a second copy
        for k in unstructured_load_arrays:
            if k in unmovable_arrays:
                del unmovable_arrays[k]

    # Generate name mappings
    subset_to_name_map = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
        # Insert copy-ins
        desc = inner_sdfg.arrays[unmovable_arr_name]
        for i, subset in enumerate(subsets):
            vec_arr_name = f"{unmovable_arr_name}_vec_{i}"
            if vec_arr_name not in inner_sdfg.arrays:
                inner_sdfg.add_array(
                    name=vec_arr_name,
                    shape=(vector_width, ),
                    dtype=desc.dtype,
                    location=desc.location,
                    transient=True,
                    strides=(1, ),
                )
            subset_to_name_map[(unmovable_arr_name, subset)] = vec_arr_name

    # For every memlet, replace the subset and
    # First replace all memlets, then access nodes

    # If there is discrepancy between in and out data names, then duplicate access nodes and add a dependency edge

    # First work on interstate edges
    for inner_state in inner_sdfg.all_states():
        for edge in inner_state.edges():
            # Skip packed arrays, it means either data name ends with packed or it is a gather-store to an array of length vector width
            if edge.data.data is not None and (edge.data.data.endswith("_packed")
                                               or inner_state.in_degree(edge.dst) == vector_width):
                continue
            if (edge.data.data, edge.data.subset) in subset_to_name_map:
                vec_name = subset_to_name_map[(edge.data.data, edge.data.subset)]
                # Then we need to get the new nae
                vec_subset = dace.subsets.Range([(0, vector_width - 1, 1)])
                edge.data = dace.memlet.Memlet(data=vec_name, subset=vec_subset)

    for inner_state in inner_sdfg.all_states():
        for node in inner_state.data_nodes():
            # Do not check packed storage
            if node.data.endswith("_packed"):
                continue

            ies = {ie for ie in inner_state.in_edges(node) if ie.data.data is not None}
            oes = {oe for oe in inner_state.out_edges(node) if oe.data.data is not None}

            # Do not check packed storage
            for e in ies.union(oes):
                if isinstance(e.src, dace.nodes.AccessNode) and e.src.data.endswith("_packed"):
                    continue
                if isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data.endswith("_packed"):
                    continue

            # Gather-store to a storage will have an in degree equal to vector length
            if len(ies) == vector_width:
                continue

            ie_datanames = {ie.data.data for ie in ies}
            oe_datanames = {oe.data.data for oe in oes}
            assert len(ie_datanames) in {
                0, 1, vector_width
            }, f"Input datanames more than one {ie_datanames}, and not equal to {vector_width} in state {state}, sdfg {state.sdfg.label}."

            assert len(ie_datanames) + len(oe_datanames) > 0

            #print(len(ie_datanames), ie_datanames)
            #print(len(oe_datanames), oe_datanames)

            if len(oe_datanames) == 0:
                ie_dataname = ie_datanames.pop()
                node.data = ie_dataname
            else:
                if len(oe_datanames) == 1:
                    oe_dataname = oe_datanames.pop()
                    node.data = oe_dataname

                    # If there is discrepancy between in and out data names, then duplicate access nodes and add a dependency edge
                    if len(ie_datanames) == 1:
                        ie_dataname = ie_datanames.pop()
                        if ie_dataname != oe_dataname:
                            # Need to duplicate the access node
                            an_in = inner_state.add_access(ie_dataname)
                            for ie in ies:
                                inner_state.remove_edge(ie)
                                inner_state.add_edge(ie.src, ie.src_conn, an_in, None, copy.deepcopy(ie.data))
                            # Add dependency edge
                            inner_state.add_edge(an_in, None, node, None, dace.memlet.Memlet(None))
                else:
                    assert len(
                        ie_datanames
                    ) == 0, f"If multiple out edges, no in edges allowed, found {ie_datanames} for {oe_datanames} in {inner_state}"
                    assert inner_state.in_degree(
                        node
                    ) == 0, f"If multiple out edges, no in edges allowed, found {ie_datanames} for {oe_datanames} in {inner_state}"
                    inner_state.remove_node(node)
                    for oe in oes:
                        an = inner_state.add_access(oe.data.data)
                        inner_state.add_edge(an, oe.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

    # Handle unmovable arrays by adding copies at the beginning and at the end of the inner SDFG
    # Copy in can't be always the first state, we need to traverse the SDFG to find it
    # Traverse using BFS, for the vectorization we assume that the inner nSDFG is a line-graph
    # And only consist of SDFGStates
    last_nodes = {n for n in inner_sdfg.nodes() if inner_sdfg.out_degree(n) == 0}
    assert len(last_nodes) == 1
    last_node = last_nodes.pop()
    if len(unmovable_arrays) > 0:
        copy_out_state = inner_sdfg.add_state_after(last_node, "copy_out")

    # Insert copy-ins and outs
    name_to_subset_map = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
        # If a packed stored, then continue
        # Add a unique vector array for each unique subset
        desc = inner_sdfg.arrays[unmovable_arr_name]

        if unmovable_arr_name in read_set:
            for i, subset in enumerate(subsets):
                vec_arr_name = f"{unmovable_arr_name}_vec_{i}"
                name_to_subset_map[vec_arr_name] = subset

                # We have the symbol mapping available in the beginning
                copy_in_state = find_copy_in_state(inner_sdfg, nsdfg_node, {str(s)
                                                                            for s in subset.free_symbols},
                                                   unmovable_arr_name)

                # Insert copy-ins
                # Need to find the copy in state
                orig_access = copy_in_state.add_access(unmovable_arr_name)
                #orig_access.setzero = True
                v_access = copy_in_state.add_access(vec_arr_name)
                #v_access.setzero = True
                vec_arr = copy_in_state.sdfg.arrays[vec_arr_name]
                assign_tasklet = copy_in_state.add_tasklet(
                    name="_AssignT1",
                    inputs={"_in"},
                    outputs={"_out"},
                    code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vec_arr.dtype]}, {vector_width}>(_out, _in);",
                    language=dace.dtypes.Language.CPP)
                copy_in_state.add_edge(orig_access, None, assign_tasklet, "_in",
                                       dace.memlet.Memlet(data=unmovable_arr_name, subset=copy.deepcopy(subset)))
                copy_in_state.add_edge(assign_tasklet, "_out", v_access, None,
                                       dace.memlet.Memlet.from_array(vec_arr_name, inner_sdfg.arrays[vec_arr_name]))

        if not no_copy_out:
            # Insert corresponding copy-out
            if unmovable_arr_name in write_set:
                for i, subset in enumerate(subsets):
                    vec_arr_name = f"{unmovable_arr_name}_vec_{i}"
                    name_to_subset_map[vec_arr_name] = subset
                    orig_access2 = copy_out_state.add_access(unmovable_arr_name)
                    #orig_access2.setzero = True
                    v_access2 = copy_out_state.add_access(vec_arr_name)
                    #v_access2.setzero = True
                    vec_arr = copy_out_state.sdfg.arrays[vec_arr_name]
                    assign_tasklet2 = copy_out_state.add_tasklet(
                        name="_AssignT2",
                        inputs={"_in"},
                        outputs={"_out"},
                        code=
                        f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vec_arr.dtype]}, {vector_width}>(_out, _in);",
                        language=dace.dtypes.Language.CPP)
                    copy_out_state.add_edge(
                        v_access2, None, assign_tasklet2, "_in",
                        dace.memlet.Memlet.from_array(vec_arr_name, inner_sdfg.arrays[vec_arr_name]))
                    copy_out_state.add_edge(assign_tasklet2, "_out", orig_access2, None,
                                            dace.memlet.Memlet(data=unmovable_arr_name, subset=copy.deepcopy(subset)))

    # Save intermediate SDFG for debugging
    # Process movable arrays at the nested SDFG boundary
    inserted_array_names = process_in_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage)

    if not no_copy_out:
        process_out_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage)

    for inner_state in inner_sdfg.all_states():
        for (dataname, subset) in movable_arrays:
            for edge in inner_state.edges():
                if edge.data.data == dataname and edge.data.subset == subset:
                    # Change the name later
                    edge.data = dace.memlet.Memlet(data=edge.data.data,
                                                   subset=dace.subsets.Range([(0, vector_width - 1, 1)]))

    for (dataname, subset) in movable_arrays:
        inner_sdfg.replace_dict({dataname: dataname + "_vec"})

    movable_datas = {t[0] for t in movable_arrays}

    nsdfg_in_conns = list(nsdfg_node.in_connectors.keys())
    nsdfg_out_conns = list(nsdfg_node.out_connectors.keys())

    for inc in nsdfg_in_conns:
        if inc in movable_datas:
            nsdfg_node.remove_in_connector(inc)
    for outc in nsdfg_out_conns:
        if outc in movable_datas:
            nsdfg_node.remove_out_connector(outc)

    for inc in nsdfg_in_conns:
        if inc in movable_datas:
            nsdfg_node.add_in_connector(inc + "_vec", force=True)

    for outc in nsdfg_out_conns:
        if outc in movable_datas:
            nsdfg_node.add_out_connector(outc + "_vec", force=True)

    # Update connector names
    # Remove movable datanames from connectors and replace with "_vec" variant
    # Some scalars / arrays will be not vectorized and thus not have `_vec` suffix
    # Make sure that we only connect the arrays that have `_vec` suffix.
    # For this: check if an edge's connector is in movable data (moved outside of the nested SDFG)
    # and top of that check if the vector-suffixed data is in the in connectors
    for movable_data in movable_datas:
        for ie in state.in_edges(nsdfg_node):
            if ie.dst_conn is not None and ie.dst_conn == movable_data and ie.dst_conn + "_vec" in nsdfg_node.in_connectors:
                assert movable_data + "_vec" in nsdfg_node.in_connectors, f"{movable_data}_vec not in {nsdfg_node.in_connectors}"
                assert len(
                    set(state.in_edges_by_connector(nsdfg_node, movable_data + "_vec"))
                ) == 0, f"There are edges connected to {movable_data}_vec: {set(state.in_edges_by_connector(nsdfg_node, movable_data + '_vec'))}"
                ie.dst_conn = movable_data + "_vec"
        for oe in state.out_edges(nsdfg_node):
            if oe.src_conn is not None and oe.src_conn == movable_data and oe.src_conn + "_vec" in nsdfg_node.out_connectors:
                assert movable_data + "_vec" in nsdfg_node.out_connectors, f"{movable_data}_vec not in {nsdfg_node.out_connectors}"
                assert len(
                    set(state.out_edges_by_connector(nsdfg_node, movable_data + "_vec"))
                ) == 0, f"There are edges connected to {movable_data}_vec: {set(state.out_edges_by_connector(nsdfg_node, movable_data + '_vec'))}"
                oe.src_conn = movable_data + "_vec"

    # Move vector data above the vector map, it makes merging overlapping accesses easier
    sdict = state.scope_dict()
    for ie in state.in_edges(nsdfg_node):
        if isinstance(ie.src, dace.nodes.AccessNode) and ie.data.data in inserted_array_names:
            sift_access_node_up(state, ie.src, sdict[ie.src])

    return inserted_array_names


def find_copy_in_state(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG, free_syms: Set[str],
                       name: str) -> dace.SDFGState:
    assert all({isinstance(n, dace.SDFGState) for n in inner_sdfg.nodes()})

    syms_available = set(nsdfg_node.symbol_mapping.keys())
    nodes_to_check = [inner_sdfg.start_block]
    # Stop when all symbols ara available
    while nodes_to_check:
        node_to_check = nodes_to_check.pop()
        cur_node = node_to_check

        if all({free_sym in syms_available for free_sym in free_syms}):
            # Add a state after cur_node
            # Check next node
            if cur_node.label.startswith("copy_in"):
                cur_node.label += f"_{name}"
                return cur_node
            return inner_sdfg.add_state_before(cur_node, f"copy_in_{name}")

        assert len(inner_sdfg.out_edges(cur_node)) <= 1
        oe = inner_sdfg.out_edges(cur_node).pop()
        nodes_to_check.append(oe.dst)
        syms_available = syms_available.union({str(s) for s in oe.data.assignments.keys()})

    raise Exception("Find copy_in state called, it could not find a state for copy-in,"
                    "this should not occur as this array already exist,"
                    "there needs to be a state where all symbols have been defined")


def map_has_branching_memlets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for out_conn in map_entry.out_connectors:
        out_egdges_of_out_conn = set(state.out_edges_by_connector(map_entry, out_conn))
        if len(out_egdges_of_out_conn) > 1:
            return True
    return False


def parse_int_or_default(value, default=8):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def sift_access_node_up(state: dace.SDFGState, node: dace.nodes.AccessNode, map_entry: dace.nodes.MapEntry):
    # We have MapEntry -> AccessNode -> DstNode
    # We move it up to be: AccessNode -> MapEntry -> DstNode
    # If access node's size is multiplied with the loop's dimensions

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)
    src_nodes = {ie.src for ie in in_edges}
    assert map_entry in src_nodes
    assert len(in_edges) == 1
    assert len(out_edges) == 1

    desc = state.sdfg.arrays[node.data]
    assert len(desc.shape) == len(map_entry.map.params)
    map_lengths = tuple([(e + 1 - b) // s for (b, e, s) in map_entry.map.range])
    # Vector map is one dimensional and has length 1 due to step size
    assert len(map_entry.map.params) == 1
    assert map_lengths[0] == 1

    ie = in_edges[0]
    oe = out_edges[0]
    # Rm access node's connection
    state.remove_edge(ie)
    state.remove_edge(oe)
    state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

    ies_from_connector = state.in_edges_by_connector(map_entry, ie.src_conn.replace("OUT_", "IN_"))
    for s_ie in ies_from_connector:
        state.remove_edge(s_ie)

        # Expand oe.data.subset
        new_subset_list = []
        p, (mb, me, ms) = map_entry.map.params[0], map_entry.map.range[0]
        for (b, e, s) in ie.data.subset:
            nb = b.subs(p, mb)
            ne = e.subs(p, mb)
            ns = s
            new_subset_list.append((nb, ne, ns))
        s_ie_subset = dace.subsets.Range(new_subset_list)

        state.add_edge(s_ie.src, s_ie.src_conn, node, None, dace.memlet.Memlet(data=s_ie.data.data, subset=s_ie_subset))
        state.add_edge(node, None, s_ie.dst, s_ie.dst_conn, copy.deepcopy(oe.data))


def sdfg_has_nested_sdfgs(sdfg: dace.SDFG):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                return True
    return False


def map_has_nested_sdfgs(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            return True
    return False


def has_nsdfg_depth_more_than_one(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            if sdfg_has_nested_sdfgs(node.sdfg):
                return True
    return False


def resolve_missing_laneid_symbols(inner_sdfg, nsdfg, state, vector_map_param):
    """
    Resolve missing expanded loop symbols of the form ``loop_var_laneid_ID`` inside
    an SDFG nested in a vectorized map.

    During vectorized code generation, additional symbol variants such as
    ``i_laneid_0``, ``i_laneid_1`` may appear, but these are often not present in
    ``nsdfg.symbol_mapping``. This function reconstructs such missing symbols and
    inserts appropriate symbol assignments before the start block of the inner SDFG.

    Parameters
    ----------
    inner_sdfg : dace.SDFG
        The inner SDFG in which missing free symbols appear.

    nsdfg : dace.nodes.NestedSDFG
        The nested SDFG node that contains the symbol mapping to the outer SDFG.

    state : dace.SDFGState
        The state containing the NestedSDFG node. Used to look up parent map symbols.

    vector_map_param : str
        The name of the map iterator corresponding to the vector lane dimension.
        Symbols derived from this parameter will be rewritten as `vector_map_param + laneid`.

    Notes
    -----
    - Missing symbols must contain ``"_laneid_"``. Symbols not matching this pattern
      trigger an assertion.
    - Symbols belonging to the parent map (returned by
      ``assert_symbols_in_parent_map_symbols``) are *not* rewritten.
    - All rewritten symbols are assigned immediately before the start block of
      ``inner_sdfg`` via ``add_state_before``.

    Raises
    ------
    AssertionError
        If unexpected missing symbols remain after processing, or if symbols do not
        conform to the expected ``*_laneid_*`` pattern.

    """
    # Find missing symbols
    missing_symbols = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))
    #print(missing_symbols)

    # Determine which of the missing symbols correspond to parent map symbols
    map_symbols = assert_symbols_in_parent_map_symbols(missing_symbols, state, nsdfg)

    # Any symbol not in map_symbols must be auto-constructed
    unresolved = missing_symbols - map_symbols
    if len(unresolved) != 0:
        assignments = {}

        for missing_sym in unresolved:
            assert "_laneid_" in missing_sym, \
                f"Unexpected symbol without '_laneid_': {missing_sym}"

            base, laneid = missing_sym.split("_laneid_")

            if base == vector_map_param:
                # vector iterator -> add lane offset
                assignments[missing_sym] = f"{base} + {laneid}"
            else:
                # other iterators -> simply alias
                assignments[missing_sym] = base

        # Insert assignment state before the start block
        inner_sdfg.add_state_before(
            inner_sdfg.start_block,
            "pre_missing_assignment",
            is_start_state=True,
            assignments=assignments,
        )

    # Ensure no missing symbols remain
    remaining = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))
    assert len(remaining) == 0, \
        f"Remaining missing symbols after fix: {remaining}"


def squeeze_memlets_of_packed_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                     array_accesses_to_be_packed: Set[str]):
    all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    all_edges = state.all_edges(*all_nodes)
    for edge in all_edges:
        if edge.data.data in array_accesses_to_be_packed:
            new_range_list = [(b, b, 1) for (b, e, s) in edge.data.subset]
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list))


def use_previous_subsets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, vector_width: int,
                         vectorizable_arrays: Set[str]):
    """
    Rewrite memlet subsets on edges leaving a single-parameter inner map so that
    structured vector accesses correctly refer to the surrounding parent map.

    The function performs:
        1. Extract the inner map parameter (e.g., `i`).
        2. Extract the parent's map lower bound symbol (e.g., `tile_i`).
        3. For each outgoing memlet:
             - Identify its corresponding incoming memlet (IN_x -> OUT_x).
             - Clone its subset.
             - Substitute outer -> inner symbols for begin/end expressions.
             - Adjust end bound when the subset length matches `vector_width` (=structured access).
             - Compute the memlet volume.
             - Assign a new Memlet with the updated subset and volume.

    Parameters
    ----------
    state : dace.SDFGState
        The current SDFG state containing the map entry.
    map_entry : dace.nodes.MapEntry
        The map entry whose outgoing memlet subsets will be rewritten.
        Must have exactly one map parameter.
    vector_width : int
        Width of the structured vector access. If a subset dimension has length
        equal to `vector_width`, we shrink its end bound by `vector_width - 1`.
        As in this case we have an exact subset, otherwise we pass a complete dimension or something in that fay that we cant change.

    Notes
    -----
    We cast symbolic expressions to string and re-sympify them to force SymPy
    to reattach the same symbol objects used by DaCe.
    """

    # Inner map has exactly one parameter, e.g., `i`.
    assert len(map_entry.map.params) == 1
    inner_param = map_entry.map.params[0]

    # Extract parent-map lower bound symbol as string, e.g. `[tile_i : ...]`.
    outer_param = str(map_entry.map.range[0][0])

    for out_edge in state.out_edges(map_entry):
        if out_edge.src_conn is None:
            continue

        # Find the corresponding incoming edge with IN_<idx> for OUT_<idx>.
        in_edges = set(state.in_edges_by_connector(map_entry, out_edge.src_conn.replace("OUT_", "IN_")))
        if not in_edges:
            continue

        # Safe: at most one incoming edge per OUT connector.
        assert len(in_edges) == 1
        in_edge = next(iter(in_edges))

        if in_edge.data.data not in vectorizable_arrays:
            continue

        # Copy original subset.
        orig_subset = copy.deepcopy(in_edge.data.subset)
        print("orig subset", orig_subset)

        new_ranges = []
        volume = 1

        for (begin, end, stride) in orig_subset:
            # Rewrite begin expression
            if hasattr(begin, "subs"):
                begin_str = str(begin.subs({outer_param: inner_param}))
                new_begin = dace.symbolic.SymExpr(begin_str).simplify()
            else:
                new_begin = begin

            # Rewrite end expression
            if hasattr(end, "subs"):
                # Subset extent length
                extent = (end + 1 - begin) // stride
                print(begin, end, stride, extent)
                # If exact vector access, shrink by vector_width - 1
                #tail_adjust = vector_width - 1 if extent == vector_width  else 0
                tail_adjust = 0
                end_str = f"{end.subs({outer_param: inner_param})} - {tail_adjust}"
                print(end_str)
                new_end = dace.symbolic.SymExpr(end_str).simplify()
                volume *= extent
            else:
                new_end = end
            
            print(new_begin, new_end, stride)
            new_ranges.append((new_begin, new_end, stride))

        # Assign new memlet with updated subset and volume
        out_edge.data = dace.memlet.Memlet(
            data=out_edge.data.data,
            subset=dace.subsets.Range(new_ranges),
            volume=volume,
        )


def reset_connectors(inner_sdfg: dace.SDFG, nsdfg: dace.nodes.NestedSDFG):
    for in_conn in nsdfg.in_connectors:
        #in_arr = inner_sdfg.arrays[in_conn]
        #print(in_arr, type(in_arr), nsdfg.in_connectors[in_conn], type(nsdfg.in_connectors[in_conn]))
        nsdfg.in_connectors[in_conn] = dace.dtypes.typeclass(None)
    for out_conn in nsdfg.out_connectors:
        #out_arr = inner_sdfg.arrays[out_conn]
        nsdfg.out_connectors[out_conn] = dace.dtypes.typeclass(None)

    for state in inner_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                for in_conn in node.in_connectors:
                    node.in_connectors[in_conn] = dace.dtypes.typeclass(None)
                for out_conn in node.out_connectors:
                    node.out_connectors[out_conn] = dace.dtypes.typeclass(None)


def remove_map(map_entry: dace.nodes.MapEntry, state: dace.SDFGState):
    assert map_entry in state.nodes()
    map_exit = state.exit_node(map_entry)

    # Replace symbol dictionary
    repldict = {str(p): str(r[0]) for p, r in zip(map_entry.map.params, map_entry.map.range)}

    # Redirect map entry's out edges
    write_only_map = True
    for edge in state.out_edges(map_entry):
        if edge.data.is_empty():
            continue
        # Add an edge directly from the previous source connector to the destination
        path = state.memlet_path(edge)
        index = path.index(edge)
        state.add_edge(path[index - 1].src, path[index - 1].src_conn, edge.dst, edge.dst_conn, edge.data)
        write_only_map = False

    # Redirect map exit's in edges.
    for edge in state.in_edges(map_exit):
        path = state.memlet_path(edge)
        index = path.index(edge)

        # Add an edge directly from the source to the next destination connector
        if len(path) > index + 1:
            state.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)

            if write_only_map:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)
                if outer_entry is not None:
                    if any({e.src == map_entry for e in state.in_edges(edge.src)}):
                        state.add_edge(outer_entry, None, edge.src, None, Memlet(None))
                    else:
                        for src in {e.src for e in state.in_edges(edge.src)}:
                            state.add_edge(outer_entry, None, src, None, Memlet(None))

            else:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)

    state.remove_node(map_entry)
    state.remove_node(map_exit)

    # Replace symbols
    all_nodes = state.all_nodes_between(outer_entry, outer_exit)
    all_edges = state.all_edges(*all_nodes)
    for n in all_nodes:
        if isinstance(n, dace.nodes.Tasklet):
            tutil.tasklet_replace_code(n, repldict)
        if isinstance(n, dace.nodes.NestedSDFG):
            for k, v in repldict.items():
                if k in n.symbol_mapping:
                    sym_expr = dace.symbolic.SymExpr(n.symbol_mapping[k])
                    if k in {str(s) for s in sym_expr.free_symbols}:
                        printer = DaceSympyPrinter(arrays=state.sdfg.arrays)
                        n.symbol_mapping[v] = printer.doprint(sym_expr.subs(k, v))
                    else:
                        n.symbol_mapping[v] = n.symbol_mapping[k]
                    del n.symbol_mapping[k]
            n.sdfg.replace_dict(repldict)
            for k, v in repldict.items():
                assert k not in n.sdfg.symbols
                assert k not in n.sdfg.free_symbols
            # SDFG repldict does not change edge subsets
            for _is in n.sdfg.all_states():
                for _se in _is.edges():
                    if _se.data.data is not None:
                        _se.data.subset.replace(repldict)
    for e in all_edges:
        if e.data.data is None:
            continue
        e.data.subset.replace(repldict)
