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


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    """ Convenience wrapper to make the .replace not in-place """
    new_subset = copy.deepcopy(subset)
    new_subset.replace(repl_dict)
    return new_subset


def repl_subset_to_symbol_offset(sdfg: dace.SDFG,
                                 subset: dace.subsets.Range,
                                 symbol_offset: str,
                                 add_missing_symbols=False) -> dace.subsets.Range:
    """
    Apply a symbolic offset to all free symbols in a subset.

    This function replaces each free symbol in the subset with a new symbol
    that has the offset appended to its name (e.g., 'i' becomes 'i_{offset}' where offset is an integer).
    New symbols are automatically added to the SDFG if they don't exist.

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

    free_syms = subset.free_symbols
    print("Free symbols in subset:", free_syms)

    repl_dict = {str(free_sym): str(free_sym) + symbol_offset for free_sym in free_syms}
    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        offset_symbol_name = str(free_sym) + symbol_offset
        if offset_symbol_name not in sdfg.symbols:
            sdfg.add_symbol(offset_symbol_name, stype)
    print("Generated replacement dictionary with offset:", repl_dict)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    print("Subset after symbol offset replacement:", new_subset)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols:
            print(free_sym, "is in free symbols of the sdfg")
            if not add_missing_symbols:
                raise Exception(
                    "This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue")

    if add_missing_symbols is True:
        for free_sym in free_syms:
            if str(free_sym) in sdfg.free_symbols:
                # The free symbol -> add to symbol mapping,
                # Assignments go to the first interstatedge
                offset_symbol_name = str(free_sym) + symbol_offset
                assign_dict = {offset_symbol_name: str(free_sym) + " + " + symbol_offset}
                sdfg.parent_nsdfg_node.symbol_mapping[str(free_sym)] = str(free_sym)
                print(free_sym, "is in free symbols of the sdfg")
                first_state = sdfg.start_block
                if len(sdfg.out_edges(first_state)) == 0:
                    sdfg.add_state_before(sdfg.start_block,
                                          label=f"pre_assign_{free_sym}",
                                          is_start_block=True,
                                          assignments=assign_dict)
                else:
                    first_edge = sdfg.out_edges(first_state)[0]
                    first_edge.data.assignments.update(assign_dict)

    return new_subset


def repl_subset_to_use_laneid_offset(sdfg: dace.SDFG, subset: dace.subsets.Range,
                                     symbol_offset: str) -> dace.subsets.Range:
    """
    Apply a symbolic offset to all free symbols in a subset.

    This function replaces each free symbol in the subset with a new symbol
    that has the offset appended to its name (e.g., 'i' becomes 'i_{offset}' where offset is an integer).
    New symbols are automatically added to the SDFG if they don't exist.

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

    free_syms = subset.free_symbols
    print(f"Symbol offset: {symbol_offset}")
    print("Free symbols in subset:", free_syms)

    repl_dict = {str(free_sym): str(free_sym) + "_laneid_" + str(symbol_offset) for free_sym in free_syms}
    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        offset_symbol_name = str(free_sym) + "_laneid_" + str(symbol_offset)
        if offset_symbol_name not in sdfg.symbols:
            sdfg.add_symbol(offset_symbol_name, stype)
    print("Generated replacement dictionary with offset:", repl_dict)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    print("Subset before symbol offset replacement:", subset)
    print("Subset after symbol offset replacement:", new_subset)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols:
            print(free_sym, "is in free symbols of the sdfg")
            raise Exception(
                "This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue")

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
                             vector_length: int) -> Set[Edge[Memlet]]:
    """
    Expand single-element memlet subsets along stride-1 dimensions to a given vector length.
    Pre-condition: all subset dimensions need to be 1

    For each memlet edge, this function modifies subsets that represent a single element
    and extend them to cover `vector_length` elements when the corresponding array stride is 1.
    Trying to modify an edge listed in `edges_to_skip` raises an error as it indicates a
    bug in the auto-vectorization logic.

    Args:
        state (SDFGState): The SDFG state containing the edges.
        edges (Iterable[Edge[Memlet]]): The memlet edges to inspect and possibly modify.
        edges_to_skip (Set[Edge[Memlet]]): Edges that should not be expanded.
        vector_length (int): The number of elements to expand contiguous subsets to.

    Returns:
        Set[Edge[Memlet]]: The set of edges whose memlets were modified.
    """
    modified_edges = set()
    for edge in edges:
        if edge.data is not None:
            state.sdfg.save("e.sdfg")
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
                    new_subset_list.append((b, b + vector_length - 1, s))
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


def is_innermost_map(map_entry: dace.nodes.MapEntry, state: SDFGState) -> bool:
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
            all_nodes = {
                k
                for k in g.all_nodes_between(n, g.exit_node(n))
                if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
            }
            assert (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)) or not any(
                isinstance(_n, dace.nodes.NestedSDFG) for _n in all_nodes)


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


def prepare_vectorized_array(state, inner_sdfg, inner_arr_name, orig_dataname, orig_arr, subset, vector_width,
                             vector_storage):
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

    Returns:
        tuple: (vector_dataname, inner_offset or 0)
    """
    # Create vector array in outer SDFG
    vector_dataname = orig_dataname + "_vec"
    if vector_dataname not in state.sdfg.arrays:
        state.sdfg.add_array(name=vector_dataname,
                             shape=(vector_width, ),
                             dtype=orig_arr.dtype,
                             location=orig_arr.location,
                             transient=True,
                             find_new_name=False,
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
        keep_mask[-1] = 1
        drop_dims(inner_sdfg, inner_arr_name, keep_mask)

        # Calculate offset for inner array
        inner_offset = [b for (b, e, s), stride in zip(subset, prev_inner_arr.strides) if stride == 1][0]
        offset_memlets(inner_sdfg, inner_arr_name, inner_offset)

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

        #print(edge_subset)
        #print(stride_one_subset)
        #print(stride_one_begin)
        #print(stride_one_begin == 0)
        #print(subset)
        #print((inner_offset, inner_offset + vector_width - 1, 1))

        # If the inner subset starts from 0, then to the SDFG just the subset accessed is passed
        # In that case we copy the edge as it is
        # Otherwise we need to generate the mapping (using the subst (and not edge subset))
        if stride_one_begin != 0:
            #print("R1", dace.subsets.Range([*subset[:-1], (subset[-1][0] + inner_offset, subset[-1][0] + inner_offset + vector_width - 1, 1)]))
            return dace.subsets.Range(
                [*subset[:-1], (subset[-1][0] + inner_offset, subset[-1][0] + inner_offset + vector_width - 1, 1)])
        else:
            #print("R2", edge_subset)
            return copy.deepcopy(edge_subset)
    else:
        # Definitely a smaller subset has ben taken due to the dimension change
        return copy.deepcopy(edge_subset)


def process_in_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                     vector_width: int, vector_storage: dace.dtypes.StorageType):
    """
    Process input edges for movable arrays.

    Args:
        state: The SDFG state
        nsdfg_node: The nested SDFG node
        movable_arrays: List of (array_name, memlet) tuples
        vector_width: Width of the vector
        vector_storage: Storage type for the vector
    """
    assert isinstance(nsdfg_node, dace.nodes.NestedSDFG)
    inner_sdfg = nsdfg_node.sdfg

    for movable_arr_name, subset in movable_arrays:
        print(movable_arr_name, subset, type(subset))
        in_edges = list(state.in_edges_by_connector(nsdfg_node, movable_arr_name))
        assert len(in_edges) <= 1

        for ie in in_edges:
            orig_arr = state.sdfg.arrays[ie.data.data]
            inner_arr_name = ie.dst_conn

            # Prepare vectorized arrays
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_arr_name, ie.data.data,
                                                                     orig_arr, subset, vector_width, vector_storage)

            # Compute copy subset
            copy_subset = compute_edge_subset(ie.data.subset, subset, orig_arr, inner_offset, vector_width)
            #raise Exception(copy_subset, subset, movable_arr_name)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            state.remove_edge(ie)
            state.add_edge(ie.src, ie.src_conn, an, None, dace.memlet.Memlet(data=ie.data.data, subset=copy_subset))
            state.add_edge(an, None, ie.dst, ie.dst_conn,
                           dace.memlet.Memlet.from_array(vector_dataname, state.sdfg.arrays[vector_dataname]))


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

    for movable_arr_name, subset in movable_arrays:
        out_edges = list(state.out_edges_by_connector(nsdfg_node, movable_arr_name))
        assert len(out_edges) <= 1

        for oe in out_edges:
            orig_arr = state.sdfg.arrays[oe.data.data]
            inner_arr_name = oe.src_conn

            # Prepare vectorized arrays
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_arr_name, oe.data.data,
                                                                     orig_arr, subset, vector_width, vector_storage)

            # Compute copy subset
            copy_subset = compute_edge_subset(oe.data.subset, subset, orig_arr, inner_offset, vector_width)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            state.remove_edge(oe)
            state.add_edge(oe.src, oe.src_conn, an, None,
                           dace.memlet.Memlet.from_array(vector_dataname, state.sdfg.arrays[vector_dataname]))
            state.add_edge(an, None, oe.dst, oe.dst_conn, dace.memlet.Memlet(data=oe.data.data, subset=copy_subset))


def offset_memlets(sdfg: dace.SDFG, dataname: str, offset: dace.symbolic.SymExpr):
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None and edge.data.data == dataname:
                edge.data.subset.offset(dace.subsets.Range([(offset, offset, 1)]), negative=True)


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
                nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
                edges = state.all_edges(*nodes)
                # Currently this enforced the map parameter to be involved in the memlet access
                # This will fail in a case such as:
                # _s2 = map_param + 1
                # A[_s2] as `map_param` will not be detected anymore.
                # This can be fixed by improving the analysis, as is an open TODO task.
                for edge in edges:
                    memlet: dace.memlet.Memlet = edge.data
                    free_symbols = memlet.subset.free_symbols
                    last_param = list(map_entry.map.params)[-1]
                    if last_param not in free_symbols:
                        raise ValueError(f"Last map parameter {last_param} must be in the memlet {memlet}, "
                                         f"not in this case - edge: {edge}, state: {state}")


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
        if in_edge.data is None:
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
        if in_edge.data is None:
            continue

        subset = in_edge.data.subset
        connector_name = in_edge.dst_conn
        connector_array = nsdfg_node.sdfg.arrays[connector_name]
        original_shape = connector_array.shape

        # Calculate all possible expected shapes
        expected_shape_full = tuple([(end + 1 - begin) for begin, end, step in subset])

        expected_shape_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([(end + 1 - begin) for begin, end, step in subset
                                               if (end + 1 - begin) != 1])

        expected_shape_collapsed_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset
                                                  if (end + 1 - begin) // step != 1])

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
        expected_shape_full = tuple([(end + 1 - begin) for begin, end, step in subset])

        expected_shape_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([(end + 1 - begin) for begin, end, step in subset
                                               if (end + 1 - begin) != 1])

        expected_shape_collapsed_strided = tuple([(end + 1 - begin) // step for begin, end, step in subset
                                                  if (end + 1 - begin) // step != 1])

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
        print(f"Output connector '{connector_name}' dimension analysis:\n"
              f"  Dims to keep: {dims_to_keep}\n"
              f"  Collapsed shape: {expected_shape_collapsed_full}\n"
              f"  Original shape: {original_shape}\n"
              f"  Subset: {subset}")

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
                                  templates: Dict[str, str], vector_map_param: str) -> None:
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

    def _str_to_float_or_str(s: Union[int, float, str, None]):
        """Convert string constants to float if possible."""
        if s is None:
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def _generate_code(rhs1_, rhs2_, const1_, const2_, lhs_, op_):
        """
        Generate the C++ vectorized code string using templates or fallbacks.

        Handles:
        - Array-array, array-scalar, scalar-array
        - Commutative and non-commutative ops
        - Single constant + array/scalar (or array/scalar + constant)
        - Fallback loops if operator not supported (hope compiler will do it)
        """
        # Use template if available
        if op_ in templates:
            # One array + optional constant
            if rhs1_ is None or rhs2_ is None:
                rhs = rhs1_ if rhs1_ is not None else rhs2_
                constant = const1_ if const1_ is not None else const2_
                if constant is None:
                    # Single array or repeated array case
                    if is_commutative:
                        return templates[op_].format(rhs1=rhs, rhs2=rhs, lhs=lhs_, op=op_, vector_width=vw)
                    return templates[op_].format(rhs1=rhs, rhs2=rhs, lhs=lhs_, op=op_, vector_width=vw)
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
                    return templates[cop_].format(rhs1=rhs,
                                                  constant=_str_to_float_or_str(constant),
                                                  lhs=lhs_,
                                                  op=op_,
                                                  vector_width=vw)

            else:
                # Two arrays
                return templates[op_].format(rhs1=rhs1_, rhs2=rhs2_, lhs=lhs_, op=op_, vector_width=vw)

        # Fallback: unsupported operator
        comparison_suffix = "? 1.0 : 0.0" if op_ in {">", ">=", "<", "<=", "==", "!="} else ""
        code_lines = [f"#pragma _dace_vectorize_hint", f"#pragma _dace_vectorize"]
        code_lines.append(f"for (int _vi = 0; _vi < {vw}; _vi += 1) {{")

        # Determine operand order
        lhs_expr = lhs_ + "[_vi]"
        rhs_left = rhs1_ if rhs1_ is not None else const1_
        rhs_right = rhs2_ if rhs2_ is not None else const2_

        if rhs_left is None or rhs_right is None:
            raise Exception("Invalid operand configuration for fallback vectorization")

        if rhs_left == const1_:
            code_lines.append(f"{lhs_expr} = ({rhs_left} {op_} {rhs_right}[_vi]){comparison_suffix};")
        else:
            code_lines.append(f"{lhs_expr} = ({rhs_left}[_vi] {op_} {rhs_right}){comparison_suffix};")

        code_lines.append("}")
        return "\n".join(code_lines)

    def _set_template(rhs1_, rhs2_, const1_, const2_, lhs_, op_, ttype):
        """Helper to set tasklet code from template/fallback."""
        node.code = dace.properties.CodeBlock(
            code=f"//{ttype}\n" +
            _generate_code(rhs1_, rhs2_, _str_to_float_or_str(const1_), _str_to_float_or_str(const2_), lhs_, op_),
            language=dace.Language.CPP)

    # Dispatch based on tasklet type
    if ttype == tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        _set_template(rhs1, rhs2, c1, c2, lhs, "=", ttype)
    elif ttype == tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT:
        node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c2};" for i in range(vw)]) + "\n",
                                              language=dace.Language.CPP)
    elif ttype == tutil.TaskletType.ARRAY_SYMBOL_ASSIGNMENT:
        node.code = dace.properties.CodeBlock(code="\n".join([f"{lhs}[{i}] = {c1}{i};" for i in range(vw)]) + "\n",
                                              language=dace.Language.CPP)
    elif ttype in {tutil.TaskletType.ARRAY_SYMBOL, tutil.TaskletType.ARRAY_ARRAY}:
        _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype)
    elif ttype in {tutil.TaskletType.UNARY_ARRAY}:
        arr_name = rhs1 if rhs1 is not None else rhs2
        occurences = tutil.count_name_occurrences(node.code.as_string.split(" = ")[1].strip(), arr_name)
        assert occurences == 1
        if op == "-":
            # Implement (-A) as (0 - A)
            _set_template(None, arr_name, "0.0", None, lhs, op, tutil.TaskletType.ARRAY_SYMBOL)
        elif op == "+":
            raise Exception("Unary + operator is not supported")
        else:
            _set_template(rhs1, rhs2, c1, c2, lhs, op, ttype)
    elif ttype in {
            tutil.TaskletType.SCALAR_ARRAY,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(None, rhs2, rhs1, None, lhs, op, ttype)
    elif ttype in {
            tutil.TaskletType.ARRAY_SCALAR,
    }:
        # The tasklet-info treads scalars as arrays and only symbols as constants
        # For the vector-code scalar is the same as a constant
        _set_template(rhs1, None, None, rhs2, lhs, op, ttype)
    elif ttype == tutil.TaskletType.SCALAR_SYMBOL:
        code_lines = [f"//{ttype}"]
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
            code_lines.append(f"{lhs}[{i}] = {expr};")
        node.code = dace.properties.CodeBlock(code="\n".join(code_lines) + "\n", language=dace.Language.CPP)
    elif ttype == tutil.TaskletType.SCALAR_SCALAR:
        raise Exception("Unhandled: SCALAR_SCALAR tasklets")
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
                code="\n".join([f"//{ttype}"] +
                               [f"{lhs}[{i}] = {use_laneid_symbol_in_expression(expr, c, i)};"
                                for i in range(vw)]) + "\n",
                language=dace.Language.CPP)
        else:
            node.code = dace.properties.CodeBlock(code=f"{lhs} = {expr};\n", language=dace.Language.CPP)
    else:
        raise NotImplementedError(f"Unhandled TaskletType: {ttype}, from: {node.code.as_string} ({node})")


def duplicate_access(state: dace.SDFGState, node: dace.nodes.AccessNode,
                     vector_width: int) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]]]:
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

        new_subset = repl_subset_to_symbol_offset(state.sdfg, ie.data.subset, str(i))

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


def expand_assignment_tasklets(state: dace.SDFGState, name: str, vector_length: int):
    """
    Expands assignment tasklets writing to an array at a to be over the vector length
    over the unit stride dimension a[0] = ..., a[1] = ..., ...
    For assignment tasklets the dataname given as name.

    Args:
        state: The SDFG state to modify.
        name: The array being written.
        vector_length: Length of the vector to expand to.
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
            ncode_str = "\n".join([f"{out_conn}[{i}] = {rhs}" for i in range(vector_length)])
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
    print(is_inout_accumulator, num_flops, accumulator_name)
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
        missing_symbols: Symbols to validate (e.g., {"i0", "j1"}).
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
            assert match
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


def collect_vectorizable_arrays(sdfg: dace.SDFG, parent_nsdfg_node: dace.nodes.NestedSDFG, parent_state: SDFGState,
                                invariant_scalars: Set[str]) -> Set[str]:
    """
    Determines which arrays can be vectorized based on their access patterns and symbol usage.
    The symbols used for accessing should not have any indirectness, meaning that they should
    not be accessing other Arrays on interstate assignemnts, this is expressed as a free function
    in sympy.

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
    print(all_accesses_to_arrays)

    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # Get the stride 1 dimension
            stride_one_dim = {i for i, stride in enumerate(sdfg.arrays[arr_name].strides) if stride == 1}.pop()
            b, e, s = access_subset[stride_one_dim]
            assert b == e
            assert s == 1

            # Evaluate the expression (b == e)
            if isinstance(b, (dace.symbolic.SymExpr, dace.symbolic.symbol)):
                if isinstance(b, dace.symbolic.SymExpr):
                    free_syms = {str(s) for s in b.free_syms}.union({f.func for f in b.atoms(sympy.Function)})
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
                        assert assignment is not None
                        assignment_expr = dace.symbolic.SymExpr(assignment)
                        # Define functions to ignore (common arithmetic + piecewise + rounding)
                        ignored = {
                            sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs, sympy.floor,
                            sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan, sympy.sinh,
                            sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                        }
                        print("A:", assignment)

                        # Collect only user-defined or nonstandard functions - in intersate edge this means array accees
                        funcs = {f.name for f in assignment_expr.atoms(sympy.Function) if f.func not in ignored}
                        # Any array on the right-hand-side -> big problem
                        # Check for scalar / array accesses like this too
                        scalars = {str(s)
                                   for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars
                        # If scalar is invariant it should be ok?
                        print("Invariant", invariant_scalars)
                        print("Non-invariant scalars",
                              {s
                               for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars)
                        if len(funcs) != 0 or len(scalars) != 0:
                            print(f"Indirect access detected: ({funcs}, {scalars}) for {arr_name}, is not vectorizable")
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
                                           state: dace.SDFGState, vector_width: int, invariant_data: Set[str]):
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
    vectorized_param = parent_map_entry.map.params[-1]
    #print("VVVVVVV", vectorized_param)

    for edge in inner_sdfg.all_interstate_edges():
        new_assignments = dict()
        assignments = edge.data.assignments
        for k, v in assignments.items():
            # Add a new symbol for each lane
            original_v_expr = dace.symbolic.SymExpr(v)
            for i in range(vector_width):
                v_expr = dace.symbolic.SymExpr(v)

                funcs = {str(f) for f in v_expr.atoms(sympy.Function)}
                non_func_free_syms = {str(s) for s in v_expr.free_symbols if str(s) not in funcs}
                # Collect functions that are accesses
                array_accesses = {f for f in funcs if f in inner_sdfg.arrays}
                variant_array_accesses = (array_accesses.union(non_func_free_syms)) - invariant_data

                if len(variant_array_accesses) == 0:
                    print(f"For `{k} = {v_expr}` no vector-lane variant data is found, skipping duplication")
                    # Copy the old self
                    new_assignments[k] = v
                    continue

                new_k = f"{k}_laneid_{i}"
                if new_k not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(new_k, inner_sdfg.symbols.get(k, dace.float64))

                # Replace map param `_for_it` with `_for_it + laneid`
                v_expr = v_expr.subs(vectorized_param, f"({vectorized_param} + {i})")

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
                        v_expr = v_expr.subs(free_sym, f"{free_sym}_laneid_{i}")

                        # Add the new symbol to the symbols
                        if f"{free_sym}_laneid_{i}" not in inner_sdfg.symbols:
                            inner_sdfg.add_symbol(f"{free_sym}_laneid_{i}",
                                                  inner_sdfg.symbols.get(str(free_sym), dace.float64))
                    else:
                        if isinstance(inner_sdfg.arrays[free_sym_str], dace.data.Scalar):
                            #print(f"Subs {free_sym} with {free_sym}")
                            v_expr = v_expr.subs(free_sym, f"{free_sym}")
                        else:
                            assert inner_sdfg.arrays[free_sym_str].shape != (1, )
                            #print(f"Subs {free_sym} with {free_sym}({i})")
                            v_expr = v_expr.subs(free_sym, f"{free_sym}({i})")

                new_v = sympy.pycode(v_expr, allow_unknown_functions=True)
                #print("NEWV", new_v)
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
        print(sym_assignments)
        for sym_assignment in sym_assignments:
            sym_assign_expr = dace.symbolic.SymExpr(sym_assignment)
            # Collect all array accesses (they are functions that are present in the sdfg)
            funcs = {(f.name, f) for f in sym_assign_expr.atoms(sympy.Function)}
            print(funcs)
            for fname, f in funcs:
                print(f"Check function: {fname} ({str(fname) in inner_sdfg.arrays})")
                if fname in inner_sdfg.arrays:
                    for arg in f.args:
                        all_function_args = all_function_args.union({str(s) for s in arg.free_symbols})

        # If all function args are s
        print(f"{sym} <-(depends)- {all_function_args}")
        # if the depend set has no arrays or scalars we can do it
        data_in_dependence_set = {d for d in all_function_args if d in inner_sdfg.arrays}
        if len(data_in_dependence_set) == 0:
            demotable_symbols.add(sym)

    for demotable_symbol in demotable_symbols:
        stype = inner_sdfg.symbols[demotable_symbol]
        sdutil.demote_symbol_to_scalar(inner_sdfg, demotable_symbol, stype)

    return demotable_symbols


def assert_no_other_subset(sdfg: dace.SDFG, recursive: bool = True):
    """
    Assert that no edge in the SDFG has an 'other_subset' attribute set.

    This validation is needed because vectorization does not support other subsets.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs

    """
    for state in sdfg.all_states():
        for edge in state.edges():
            assert edge.data.other_subset is None
        if recursive:
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    assert_no_other_subset(node.sdfg, True)


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
    for state in sdfg.all_states():
        for edge in state.edges():
            assert edge.data.wcr is None
        if recursive:
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    assert_no_wcr(node.sdfg, True)  # BUG: Should call assert_no_wcr


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
                                       vector_length: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
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
        vector_length: Length of the vector array

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
                                             shape=(vector_length, ),
                                             dtype=orig_arr.dtype,
                                             location=orig_arr.location,
                                             transient=True,
                                             find_new_name=False,
                                             storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(name="_Assign",
                          inputs={"_in"},
                          outputs={"_out"},
                          code="vector_copy(_out, _in);",
                          language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    e1 = state.add_edge(src, src_conn, t, "_in", copy.deepcopy(edge.data))
    e2 = state.add_edge(t, "_out", an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(an, None, dst, dst_conn, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    state.remove_edge(edge)

    return (e1, e2, e3)


def insert_assignment_tasklet_to_dst(state: dace.SDFGState, edge: Edge[Memlet],
                                     vector_storage_type: dace.dtypes.StorageType,
                                     vector_length: int) -> Tuple[Edge[Memlet], Edge[Memlet], Edge[Memlet]]:
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
        vector_length: Length of the vector array

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
                                      shape=(vector_length, ),
                                      dtype=orig_arr.dtype,
                                      location=orig_arr.location,
                                      transient=True,
                                      find_new_name=False,
                                      storage=vector_storage_type)
        vector_data = arr
    else:
        vector_data = state.sdfg.arrays[vector_dataname]

    # Create assignment tasklet
    t = state.add_tasklet(name="_Assign",
                          inputs={"_in"},
                          outputs={"_out"},
                          code="vector_copy(_out, _in);",
                          language=dace.dtypes.Language.CPP)

    # Create access node and edges
    an = state.add_access(vector_dataname)
    e1 = state.add_edge(src, src_conn, an, None, dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e2 = state.add_edge(an, None, t, "_in", dace.memlet.Memlet.from_array(vector_dataname, vector_data))
    e3 = state.add_edge(t, "_out", dst, dst_conn, copy.deepcopy(edge.data))
    state.remove_edge(edge)

    return (e1, e2, e3)


def add_copies_before_and_after_nsdfg(
    state: SDFGState,
    nsdfg_node: dace.nodes.NestedSDFG,
    vector_width: int,
    vector_storage: dace.dtypes.StorageType,
    skip: Set[str],
):
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
                print("Copy that can't be moved (due to unstructured / conflicting accesses):", dataname, "->", memlets)
                unmovable_arrays[dataname] = set(memlets)
            else:
                print(f"Copy is not moved: {dataname} in skip (packed store/load)")
        else:
            # Single access pattern - check if symbols are available outside
            memlet = next(iter(memlets))
            memlet_syms = {str(s) for s in memlet.free_symbols}
            avaialble_syms = {str(s) for s in state.symbols_defined_at(nsdfg_node)}

            if all({s in avaialble_syms for s in memlet_syms}):
                if dataname not in skip:
                    print("Copy can be moved", dataname, "->", memlets)
                    movable_arrays.add((dataname, memlet))
                else:
                    print(f"Copy is not moved: {dataname} in skip (packed store/load)")
            else:
                sym_ok = {(s, s in avaialble_syms) for s in memlet_syms}
                if dataname not in skip:
                    print("Copy can't be moved (due to symbols)", dataname, "->", memlets, "Syms:", sym_ok)
                    unmovable_arrays[dataname] = set(memlets)
                else:
                    print(f"Copy is not moved: {dataname} in skip (packed store/load)")

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
            if (edge.data.data, edge.data.subset) in subset_to_name_map:
                vec_name = subset_to_name_map[(edge.data.data, edge.data.subset)]
                # Then we need to get the new nae
                vec_subset = dace.subsets.Range([(0, vector_width - 1, 1)])
                edge.data = dace.memlet.Memlet(data=vec_name, subset=vec_subset)

    for inner_state in inner_sdfg.all_states():
        for node in inner_state.data_nodes():
            ies = {ie for ie in inner_state.in_edges(node) if ie.data.data is not None}
            oes = {oe for oe in inner_state.out_edges(node) if oe.data.data is not None}
            ie_datanames = {ie.data.data for ie in ies}
            oe_datanames = {oe.data.data for oe in oes}
            assert len(ie_datanames) <= 1 or len(oe_datanames) == vector_width
            assert len(oe_datanames) <= 1 or len(oe_datanames) == vector_width
            assert len(ie_datanames) + len(oe_datanames) > 0

            if len(ie_datanames) == 0:
                oe_dataname = oe_datanames.pop()
                node.data = oe_dataname
            elif len(oe_datanames) == 0:
                ie_dataname = ie_datanames.pop()
                node.data = ie_dataname
            else:
                ie_dataname = ie_datanames.pop()
                oe_dataname = oe_datanames.pop()

                if ie_dataname == oe_dataname:
                    node.data = ie_dataname
                else:
                    node.data = ie_dataname
                    # Add new access node
                    a2 = inner_state.add_access(oe_dataname)
                    for oe in oes:
                        inner_state.remove_edge(oe)
                        inner_state.add_edge(a2, oe.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
                    # Add the dependency edge
                    inner_state.add_edge(node, None, a2, None, dace.memlet.Memlet(None))

    # Handle unmovable arrays by adding copies at the beginning and at the end of the inner SDFG
    # Copy in can't be always the first state, we need to traverse the SDFG to find it
    # Traverse using BFS, for the vectorization we assume that the inner nSDFG is a line-graph
    # And only consist of SDFGStates
    last_nodes = {n for n in inner_sdfg.nodes() if inner_sdfg.out_degree(n) == 0}
    assert len(last_nodes) == 1
    last_node = last_nodes.pop()
    copy_out_state = inner_sdfg.add_state_after(last_node, "copy_out")

    # Insert copy-ins and outs
    name_to_subset_map = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
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
                v_access = copy_in_state.add_access(vec_arr_name)
                assign_tasklet = copy_in_state.add_tasklet(name="_Assign",
                                                           inputs={"_in"},
                                                           outputs={"_out"},
                                                           code="vector_copy(_out, _in);",
                                                           language=dace.dtypes.Language.CPP)
                copy_in_state.add_edge(orig_access, None, assign_tasklet, "_in",
                                       dace.memlet.Memlet(data=unmovable_arr_name, subset=copy.deepcopy(subset)))
                copy_in_state.add_edge(assign_tasklet, "_out", v_access, None,
                                       dace.memlet.Memlet.from_array(vec_arr_name, inner_sdfg.arrays[vec_arr_name]))

        # Insert corresponding copy-out
        if unmovable_arr_name in write_set:
            for i, subset in enumerate(subsets):
                vec_arr_name = f"{unmovable_arr_name}_vec_{i}"
                name_to_subset_map[vec_arr_name] = subset
                orig_access2 = copy_out_state.add_access(unmovable_arr_name)
                v_access2 = copy_out_state.add_access(vec_arr_name)
                assign_tasklet2 = copy_out_state.add_tasklet(name="_Assign",
                                                             inputs={"_in"},
                                                             outputs={"_out"},
                                                             code="vector_copy(_out, _in);",
                                                             language=dace.dtypes.Language.CPP)
                copy_out_state.add_edge(v_access2, None, assign_tasklet2, "_in",
                                        dace.memlet.Memlet.from_array(vec_arr_name, inner_sdfg.arrays[vec_arr_name]))
                copy_out_state.add_edge(assign_tasklet2, "_out", orig_access2, None,
                                        dace.memlet.Memlet(data=unmovable_arr_name, subset=copy.deepcopy(subset)))

    # Save intermediate SDFG for debugging
    # Process movable arrays at the nested SDFG boundary
    process_in_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage)
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

    # Update connector names
    # Remove movable datanames from connectors and replace with "_vec" variant
    for movable_data in movable_datas:
        for ie in state.in_edges(nsdfg_node):
            if ie.dst_conn in movable_data:
                ie.dst_conn = movable_data + "_vec"
        for oe in state.out_edges(nsdfg_node):
            if oe.src_conn in movable_data:
                oe.src_conn = movable_data + "_vec"

    nsdfg_in_conns = list(nsdfg_node.in_connectors.keys())
    nsdfg_out_conns = list(nsdfg_node.out_connectors.keys())

    for inc in nsdfg_in_conns:
        if inc in movable_datas:
            nsdfg_node.remove_in_connector(inc)
            nsdfg_node.add_in_connector(inc + "_vec", force=True)
    for outc in nsdfg_out_conns:
        if outc in movable_datas:
            nsdfg_node.remove_out_connector(outc)
            nsdfg_node.add_out_connector(outc + "_vec", force=True)


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
