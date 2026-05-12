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
from typing import Dict, Set

import dace
from dace import SDFGState

from dace.transformation.passes.vectorization.utils.code_rewrite import drop_dims

# Suffix used for the per-NSDFG vector-array name allocated by
# ``prepare_vectorized_array``. Single module-level constant rather than a
# hardcoded f-string at the callsite. See ``NameScheme`` cleanup TODO in
# the plan's "Known issues" section for the broader centralisation.
_VEC_K_SUFFIX = "_vec_k"


def get_vector_max_access_ranges(state: SDFGState, node: dace.nodes.NestedSDFG) -> Dict[str, str]:
    """
    For each vector-map parameter, return the END bound of the outer
    data-parallel map that supplies its BEGIN expression.

    Walks the two-level scope hierarchy ``nsdfg -> vector_map -> data_map``,
    matches each vector-map ``begin`` expression to a data-map parameter
    (so that ``vector_map`` was constructed as ``i_v = i : i + W : W`` over
    the data-map's ``i``), and returns the data-map's end bound.

    Args:
        state: The SDFG state containing the nested SDFG node.
        node: The nested SDFG node whose vector access ranges to determine.

    Returns:
        Dictionary mapping vector-map parameter names (e.g. ``'i_v'``) to
        the outer data-map's end bound (e.g. the string repr of ``'N - 1'``
        if the data map iterates ``0:N``). The semantic name "max access
        range" is loose — this is the upper iteration bound, not the
        per-memlet access range.

    Example:
        For ``map i=0:N -> map i_v=i:i+4:4 -> NestedSDFG``, returns ``{'i_v': 'N - 1'}``.

    Note:
        Lookup uses ``dace.symbolic.simplify(begin)`` for canonical
        equality so structurally-equal exprs (``i + 0`` vs ``i``) match.
    """
    scope_dict = state.scope_dict()
    vector_map = scope_dict[node]
    data_map = scope_dict[vector_map]

    # Simplify-keyed mapping: data-map ``begin`` -> data-map ``end``.
    d_simplified_begin_to_end = {dace.symbolic.simplify(begin): end for begin, end, _ in data_map.map.range}

    param_max_ranges = {}
    for v_param, (v_begin, _, _) in zip(vector_map.map.params, vector_map.map.range):
        canonical_begin = dace.symbolic.simplify(v_begin)
        # Bare lookup: matches when the vector-map ``begin`` simplifies
        # to the same sympy expression as one of the data-map begins.
        param_max_ranges[v_param] = str(d_simplified_begin_to_end[canonical_begin])

    return param_max_ranges


def find_state_of_nsdfg_node(root_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG) -> dace.SDFGState:
    """Return the ``SDFGState`` that contains ``nsdfg_node``.

    Callers that need the containing ``SDFG`` should read it off the
    returned state via ``state.sdfg``. The previous body returned the
    root SDFG (not the state) despite the name and return annotation; it
    was a name-vs-behaviour mismatch. Fixed to honour the signature.
    """
    for n, g in root_sdfg.all_nodes_recursive():
        if n == nsdfg_node:
            if not isinstance(g, dace.SDFGState):
                raise Exception(f"Expected a SDFGState container for {nsdfg_node}, got {type(g).__name__} ({g})")
            return g
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

        # Calculate expected shapes based on subset.
        # Apply ``.simplify()`` so that the canonicalisation matches the
        # sibling ``fix_nsdfg_connector_array_shapes_mismatch`` — previously
        # the check used raw ``(end + 1 - begin)`` and the fix used
        # ``.simplify()``, so the same input could be flagged as a mismatch
        # by ``check_*`` but accepted by ``fix_*``.
        expected_shape_full = tuple([(end + 1 - begin).simplify() for begin, end, step in subset])

        expected_shape_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([((end + 1 - begin) // step).simplify() for begin, end, step in subset
                                                  if ((end + 1 - begin) // step).simplify() != 1])

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
        tuple: (vector_dataname, inner_offset). ``inner_offset`` is currently
        always ``0`` because the multi-dim path rewrites inner memlets directly
        via ``walk_memlets_of`` instead of returning an offset for the caller
        to apply. It is kept in the return tuple for backwards-compatibility
        with ``compute_edge_subset``, which still accepts an offset arg.
    """
    vector_dataname_candidate = orig_dataname + _VEC_K_SUFFIX if use_name is None else use_name
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
            except (TypeError, ValueError, AttributeError):
                # Non-numeric length symbolic that ``simplify`` cannot
                # canonicalize to ``1``. Conservatively keep the dim
                # (it is provably not length-1).
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
        if not (reuse_name_if_existing and use_name is not None):
            from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
            surviving_offsets = [(b, b, 1) for (b, e, s), keep in zip(subset, keep_mask) if keep]
            offset_range = dace.subsets.Range(surviving_offsets)
            for _inner_state, inner_edge in walk_memlets_of(inner_sdfg, inner_arr_name):
                inner_edge.data.subset = inner_edge.data.subset.offset_new(offset_range, negative=True)

    assert inner_offset == 0, (f"prepare_vectorized_array contract: inner_offset must remain 0 (the multi-dim path "
                               f"rewrites memlets in-place via walk_memlets_of); got {inner_offset}")
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
        stride_one_indices = [i for i, stride in enumerate(orig_arr.strides) if stride == 1]
        assert len(stride_one_indices) == 1, f"{stride_one_indices} != 1: {orig_arr.strides}, {subset}"
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
        # ``subset`` has fewer dims than the original array. The NSDFG
        # collapse already dropped the length-1 dims, so the edge_subset
        # carries the post-collapse access window — pass it through.
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

    vectorized_datanames: Set[str] = set()
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

            # Symmetric with ``process_in_edges``: a newly minted vector
            # array must not already be in ``vectorized_datanames`` unless
            # we asked to reuse the inout-connector name. Catches collisions
            # between two unrelated out-edges that picked the same suffix.
            if inout_data_name is None:
                assert vector_dataname not in vectorized_datanames
            vectorized_datanames.add(vector_dataname)

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


def add_copies_before_and_after_nsdfg(
    state: SDFGState,
    nsdfg_node: dace.nodes.NestedSDFG,
    vector_width: int,
    vector_storage: dace.dtypes.StorageType,
    skip: Set[str],
) -> Set[str]:
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
    # ``collect_all_memlets_to_dataname`` lives in ``utils.queries`` (S1b);
    # imported lazily to keep this module's top-level import surface narrow.
    # ``sift_access_node_up`` is defined in this module (moved in S6d-d) —
    # used directly without re-import.
    from dace.transformation.passes.vectorization.utils.queries import collect_all_memlets_to_dataname

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

    # Unstructured-load heuristic: producers of unstructured-loads
    # (the existing ``_generate_loads_to_packed_storage`` path) always
    # emit exactly ``vector_width`` distinct memlets per data name, so
    # we recognise the shape by count. False-positive risk: any unrelated
    # data with the same memlet count would be misclassified — the proper
    # fix is for the producer to mark its access nodes with an
    # ``is_unstructured_load`` sentinel and have us read that here instead
    # of pattern-matching by count.
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
                    find_new_name=False,
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
                            an_in.setzero = True
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
                        an.setzero = True
                        inner_state.add_edge(an, oe.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

    # Handle unmovable arrays by adding copies at the beginning and at the end of the inner SDFG
    # Copy in can't be always the first state, we need to traverse the SDFG to find it.
    # The walk is single-successor only (line graph asserted in ``find_copy_in_state``),
    # so BFS/DFS distinction does not apply.
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
                orig_access.setzero = True
                v_access = copy_in_state.add_access(vec_arr_name)
                v_access.setzero = True
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

        # Insert corresponding copy-out
        if unmovable_arr_name in write_set:
            for i, subset in enumerate(subsets):
                vec_arr_name = f"{unmovable_arr_name}_vec_{i}"
                name_to_subset_map[vec_arr_name] = subset
                orig_access2 = copy_out_state.add_access(unmovable_arr_name)
                orig_access2.setzero = True
                v_access2 = copy_out_state.add_access(vec_arr_name)
                v_access2.setzero = True
                vec_arr = copy_out_state.sdfg.arrays[vec_arr_name]
                assign_tasklet2 = copy_out_state.add_tasklet(
                    name="_AssignT2",
                    inputs={"_in"},
                    outputs={"_out"},
                    code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vec_arr.dtype]}, {vector_width}>(_out, _in);",
                    language=dace.dtypes.Language.CPP)
                copy_out_state.add_edge(v_access2, None, assign_tasklet2, "_in",
                                        dace.memlet.Memlet.from_array(vec_arr_name, inner_sdfg.arrays[vec_arr_name]))
                copy_out_state.add_edge(assign_tasklet2, "_out", orig_access2, None,
                                        dace.memlet.Memlet(data=unmovable_arr_name, subset=copy.deepcopy(subset)))

    # Save intermediate SDFG for debugging
    # Process movable arrays at the nested SDFG boundary
    inserted_array_names = process_in_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage)
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
            # If this state was created by a prior call to
            # ``find_copy_in_state`` (marked via the side attribute below),
            # reuse it — the array name gets appended to its label so the
            # SDFG dump shows every reuse hit.
            if getattr(cur_node, "_vec_copy_in_state", False):
                cur_node.label += f"_{name}"
                return cur_node
            new_state = inner_sdfg.add_state_before(cur_node, f"copy_in_{name}")
            new_state._vec_copy_in_state = True
            return new_state

        assert len(inner_sdfg.out_edges(cur_node)) <= 1
        oe = inner_sdfg.out_edges(cur_node).pop()
        nodes_to_check.append(oe.dst)
        syms_available = syms_available.union({str(s) for s in oe.data.assignments.keys()})

    raise RuntimeError(f"find_copy_in_state: no state in {inner_sdfg.label} defines every symbol in "
                       f"{free_syms} (have only {syms_available}); the array {name!r} must already exist "
                       f"by the time some state has all its defining symbols in scope")


def reset_connectors(inner_sdfg: dace.SDFG, nsdfg: dace.nodes.NestedSDFG):
    # TODO(upstream-correctness): this helper erases all connector dtypes
    # because earlier passes in the vectorization pipeline leave wrong
    # types behind. The proper fix is to make those upstream passes set
    # the right type at the source; this helper is a band-aid.
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
            elif isinstance(node, dace.nodes.NestedSDFG):
                # Recurse: NSDFG nested deeper than one level also needs
                # the type reset. Previously this branch was missing,
                # leaving deep hierarchies untouched.
                reset_connectors(node.sdfg, node)


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
