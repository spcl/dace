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
from typing import Dict

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
