# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
NestedSDFG connector-array shape / reshape helpers.

These helpers manage the shape contract at the NSDFG boundary: when a
parent state passes a slice of an outer array through a connector, the
inner SDFG's array descriptor must match the slice shape. DaCe collapses
length-1 dims at the boundary by convention, so the inner shape is
typically a strict subset of the outer dims.

Each helper here is a pure relocation of the corresponding function
from ``vectorization_utils.py``. Defensive checks (the validation
``assert`` inside ``check_nsdfg_connector_array_shapes_match``, the
``original.shape`` rebuild contract inside
``fix_nsdfg_connector_array_shapes_mismatch``) are kept as-is per the
locked policy.

``prepare_vectorized_array``, ``process_in_edges``, ``process_out_edges``,
``add_copies_before_and_after_nsdfg``, and ``find_copy_in_state`` are
larger and migrate in follow-up slices (S4b, S4c).
"""
import copy
from typing import Dict, List, Set

import dace
from dace import SDFGState

from dace.transformation.passes.vectorization.utils.code_rewrite import drop_dims


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
        # Cloudsc-class kernels pass the FULL outer-array shape as the
        # connector (e.g. ``(klon, klev)``) with a smaller memlet subset
        # (e.g. ``arr[8*i, 0:j+1]``); the rebuild to ``collapsed_full``
        # narrows the connector to the actual slice and is legitimate.
        # A stricter raise here breaks those callers — the planned
        # pass-through-subsets redesign will replace this whole function.

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
        # See input-edge branch above for the rationale.

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

        # Update all accesses inside nested SDFG if:
        # 1. Not a 1D array (len > 1)
        # 2. Original shape matches the subset dimensionality
        # 3. Original shape had more dimensions than the collapsed shape
        should_drop_dims = (len(dims_to_keep) != 1 and len(original_shape) == len(dims_to_keep)
                            and len(original_shape) > len(expected_shape_collapsed_full))

        if should_drop_dims:
            drop_dims(nsdfg_node.sdfg, dims_to_keep, connector_name)


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
    vector_dataname_candidate = orig_dataname + "_vec_k" if use_name is None else use_name
    if reuse_name_if_existing:
        assert use_name is not None
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
        # NSDFG semantics collapse every length-1 subset dim at the boundary;
        # the surviving dim is the one whose subset length is not 1. Drive the
        # keep-mask off the subset rather than a layout-specific guess (the
        # previous ``keep_mask[-1] = 1`` was C-layout only and the
        # ``drop_dims`` call itself had swapped args, so it had never actually
        # rewritten the inner memlet — landing the dim-collapse here for the
        # first time means the inner accesses now match the (vector_width,)
        # connector shape).
        keep_mask = [0 for _ in orig_arr.shape]
        for i, (b, e, s) in enumerate(subset):
            length = e - b + 1
            try:
                if dace.symbolic.simplify(length) != 1:
                    keep_mask[i] = 1
            except Exception:
                keep_mask[i] = 1
        if sum(keep_mask) != 1:
            raise NotImplementedError(
                f"prepare_vectorized_array: subset {subset} has {sum(keep_mask)} non-length-1 dims "
                f"on a {len(orig_arr.shape)}-D array, exactly one is required by the NSDFG collapse")
        # Note: contig-vs-surviving-dim alignment is NOT enforced here; the
        # vectorizer also handles non-unit-stride packs via gather paths
        # elsewhere, and the existing test corpus exercises those.
        drop_dims(inner_sdfg, tuple(keep_mask), inner_arr_name)

        # Offset the surviving dim by the outer subset's start on that dim,
        # so an inner access like ``arr[start]`` becomes the first vector
        # lane ``arr[0]``. Don't route through ``offset_memlets`` here: it
        # post-collapses length-1 dims which would silently turn the
        # vector-lane memlet into a 0-D ``arr[]`` access.
        if not (reuse_name_if_existing is True and use_name is not None):
            from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
            surviving_offsets = [(b, b, 1) for (b, e, s), keep in zip(subset, keep_mask) if keep]
            offset_range = dace.subsets.Range(surviving_offsets)
            for _inner_state, inner_edge in walk_memlets_of(inner_sdfg, inner_arr_name):
                inner_edge.data.subset = inner_edge.data.subset.offset_new(offset_range, negative=True)

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
            assert vector_dataname not in vectorized_datanames
            vectorized_datanames.add(vector_dataname)

            # Compute copy subset
            copy_subset = compute_edge_subset(ie.data.subset, prev_subset, orig_arr, inner_offset, vector_width)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            an.setzero = True
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
            # We should reuse the name if we have an inout connectors.
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_arr_name, oe.data.data,
                                                                     orig_arr, subset, vector_width, vector_storage,
                                                                     inout_data_name is not None, inout_data_name)

            # Compute copy subset
            copy_subset = compute_edge_subset(oe.data.subset, prev_subset, orig_arr, inner_offset, vector_width)

            # Add access node and rewire edges
            an = state.add_access(vector_dataname)
            an.setzero = True
            state.remove_edge(oe)
            assert oe.src == nsdfg_node
            assert oe.src_conn is not None
            assert len(set(state.out_edges_by_connector(nsdfg_node, oe.src_conn))) == 0
            state.add_edge(oe.src, oe.src_conn, an, None,
                           dace.memlet.Memlet.from_array(vector_dataname, state.sdfg.arrays[vector_dataname]))
            state.add_edge(an, None, oe.dst, oe.dst_conn, dace.memlet.Memlet(data=oe.data.data, subset=copy_subset))

    state.sdfg.validate()


def reset_connectors(inner_sdfg: dace.SDFG, nsdfg: dace.nodes.NestedSDFG):
    for in_conn in nsdfg.in_connectors:
        nsdfg.in_connectors[in_conn] = dace.dtypes.typeclass(None)
    for out_conn in nsdfg.out_connectors:
        nsdfg.out_connectors[out_conn] = dace.dtypes.typeclass(None)

    for state in inner_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                for in_conn in node.in_connectors:
                    node.in_connectors[in_conn] = dace.dtypes.typeclass(None)
                for out_conn in node.out_connectors:
                    node.out_connectors[out_conn] = dace.dtypes.typeclass(None)
