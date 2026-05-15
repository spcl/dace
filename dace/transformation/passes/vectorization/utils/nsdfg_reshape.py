# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
NestedSDFG connector-array shape / reshape helpers.

These helpers manage the shape contract at the NSDFG boundary: when a
parent state passes a slice of an outer array through a connector, the
inner SDFG's array descriptor must match the slice shape. DaCe collapses
length-1 dims at the boundary by convention, so the inner shape is
typically a strict subset of the outer dims.

Defensive checks (the validation ``assert`` inside
``check_nsdfg_connector_array_shapes_match``, the ``original.shape``
rebuild contract inside ``fix_nsdfg_connector_array_shapes_mismatch``)
are intentional — loud failures are preferred over silent shape
corruption at the NSDFG boundary.
"""
import copy
from typing import Dict, Optional, Set

import dace
from dace import SDFGState

from dace.transformation.passes.vectorization.utils.code_rewrite import drop_dims
from dace.transformation.passes.vectorization.utils.name_schemes import PackedNameScheme, VecNameScheme


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


def find_state_containing_node(root_sdfg: dace.SDFG, node: dace.nodes.Node) -> dace.SDFGState:
    """Return the ``SDFGState`` that contains ``node``.

    Works for any node type — Tasklet, NestedSDFG, AccessNode, MapEntry,
    etc. The only caller today (``map_predicates``) passes a Tasklet, so
    the function is intentionally typed as general ``dace.nodes.Node``.

    Callers that need the containing ``SDFG`` should read it off the
    returned state via ``state.sdfg``.

    History: previously named ``find_state_of_nsdfg_node`` with a
    return-type annotation of ``SDFGState`` but a body that returned
    the root SDFG, then later fixed to return the state. Renamed now
    so the name + param type reflect the actual contract.
    """
    for n, g in root_sdfg.all_nodes_recursive():
        if n == node:
            if not isinstance(g, dace.SDFGState):
                raise Exception(f"Expected a SDFGState container for {node}, got {type(g).__name__} ({g})")
            return g
    raise Exception(f"State containing the node ({node}) not found in the root SDFG ({root_sdfg.label})")


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


def _raise_on_expansion_rebuild_mismatch(connector_name: str, original_shape: tuple, new_shape: tuple,
                                         expected_full: tuple, expected_strided: tuple,
                                         expected_collapsed_strided: tuple, *, direction: str,
                                         vector_width: Optional[int] = None) -> None:
    """Guard for ``fix_nsdfg_connector_array_shapes_mismatch``.

    Four rebuild patterns are legitimate and must be allowed:
    - **Narrowing**: rank-equal with every new dim ≤ original (cloudsc
      case: ``(klon, klev) → (5, 9)`` slice).
    - **Drop dims**: new rank < original (collapsed-full).
    - **Vector-widening**: rank-equal, every expanding dim grows by
      exactly ``vector_width - 1`` while the rest are unchanged.  This
      is a stencil halo being widened for a W-wide vector body — e.g. a
      5-point jacobi stencil window ``(3, 3) → (3, 10)`` for ``W=8``
      (column halo ``3`` → ``3 + (W-1) = 10``).  The inner SDFG's
      vectorized accesses *do* legitimately address the wider range, so
      this is NOT corrupting.
    - **Placeholder expansion**: original has every dim == 1 — the
      connector was the post-P1 / post-prepare_vectorized_array
      single-element placeholder being expanded to the actual bbox
      (strided 1D / multi-dim diagonal paths rely on this).

    Reject only the genuinely-corrupting shape: rank-equal with at
    least one new dim STRICTLY larger than a non-1 original dim. That
    is the test-fixture case ``(3, 4) → (5, 9)`` where the inner SDFG
    was sized for an actual range that the rebuild can't legitimately
    grow past.

    Raises:
        ValueError: if the proposed rebuild expands a non-1 original
            dim past its size, or grows the rank.
    """
    # Drop-dims case (new has fewer dims than original) — always
    # allowed; the helper computes ``dims_to_keep`` separately.
    if len(new_shape) < len(original_shape):
        return

    # Placeholder expansion (original is all-1s) — always allowed.
    try:
        all_ones = all(int(d) == 1 for d in original_shape)
    except Exception:
        all_ones = False
    if all_ones:
        return

    def _int_or_none(x):
        try:
            return int(x)
        except Exception:
            return None

    expands_real_dim = False
    if len(new_shape) > len(original_shape):
        # Rank-growth on a non-all-1s original — genuinely corrupting.
        expands_real_dim = True
    else:
        # Rank-equal: flag if any non-1 original dim gets STRICTLY larger.
        per_dim_growth = []
        for orig_d, new_d in zip(original_shape, new_shape):
            orig_int = _int_or_none(orig_d)
            diff = _int_or_none(new_d - orig_d) if hasattr(new_d, "__sub__") else None
            per_dim_growth.append(diff)
            if diff is not None and diff > 0 and (orig_int is None or orig_int > 1):
                expands_real_dim = True

        # Vector-widening exception: every dim either unchanged (diff 0)
        # or grown by exactly ``vector_width - 1`` (a stencil halo being
        # widened for the W-wide vector body, e.g. jacobi ``(3,3)`` →
        # ``(3,10)`` at W=8).  The inner vectorized accesses legitimately
        # address the wider range — allow it.
        if (expands_real_dim and vector_width is not None
                and all(d == 0 or d == vector_width - 1 for d in per_dim_growth if d is not None)
                and all(d is not None for d in per_dim_growth)
                and any(d == vector_width - 1 for d in per_dim_growth)):
            expands_real_dim = False

    if expands_real_dim:
        raise ValueError(
            f"fix_nsdfg_connector_array_shapes_mismatch ({direction}): connector "
            f"{connector_name!r} has original shape {original_shape}; none of the four "
            f"expected interpretations match and the candidate rebuild "
            f"({new_shape}) would EXPAND a non-placeholder dim. Inner SDFG accesses "
            f"can't legitimately address the larger range. Expected shapes considered:\n"
            f"    Full:              {expected_full}\n"
            f"    Strided:           {expected_strided}\n"
            f"    Collapsed full:    {new_shape}\n"
            f"    Collapsed strided: {expected_collapsed_strided}\n"
            f"Fix the caller's connector shape or memlet subset to be consistent."
        )


def fix_nsdfg_connector_array_shapes_mismatch(parent_state: dace.SDFGState,
                                              nsdfg_node: dace.nodes.NestedSDFG,
                                              vector_width: Optional[int] = None) -> None:
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

    :param parent_state: State in the parent SDFG containing the NSDFG node.
    :param nsdfg_node: NSDFG node whose connector shapes to fix.
    :param vector_width: When set, lets the expansion guard recognise a
        legitimate vector-widening rebuild (a halo dim grown by exactly
        ``vector_width - 1``) instead of raising — needed for stencil
        bodies (jacobi) on the remainder path.
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

        # ===== Mismatch detected - decide rebuild vs raise =====
        # Cloudsc-class kernels pass the FULL outer-array shape as the
        # connector (e.g. ``(klon, klev)``) with a smaller memlet subset
        # (e.g. ``arr[8*i, 0:j+1]``); the rebuild to ``collapsed_full``
        # narrows the connector to the actual slice and is legitimate.
        #
        # The rebuild is ONLY safe when it NARROWS (drops dims or
        # shrinks each surviving dim). When the rebuild would EXPAND
        # the connector (any new dim is larger than the corresponding
        # original dim), the inner SDFG's existing accesses can't
        # legitimately address the larger range — raise loudly so the
        # caller fixes its inputs, rather than silently corrupting
        # downstream codegen.
        _raise_on_expansion_rebuild_mismatch(connector_name, original_shape,
                                             expected_shape_collapsed_full,
                                             expected_shape_full,
                                             expected_shape_strided,
                                             expected_shape_collapsed_strided,
                                             direction="in",
                                             vector_width=vector_width)

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

        # ===== Mismatch detected - decide rebuild vs raise =====
        # See input-edge branch above for the rationale.
        _raise_on_expansion_rebuild_mismatch(connector_name, original_shape,
                                             expected_shape_collapsed_full,
                                             expected_shape_full,
                                             expected_shape_strided,
                                             expected_shape_collapsed_strided,
                                             direction="out",
                                             vector_width=vector_width)

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
    vector_dataname_candidate = VecNameScheme.make_k(orig_dataname) if use_name is None else use_name
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


def _setup_strided_inside_nsdfg(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                inner_sdfg: dace.SDFG, edge, inner_conn: str, orig_data: str,
                                orig_arr, vector_width: int, stride: int, *, direction: str,
                                multi_dim_param_dims: tuple = ()) -> None:
    """Wire a strided boundary edge into the NSDFG so the strided load /
    store happens INSIDE the NSDFG body.

    Direction "in": outer edge passes the full bbox to the NSDFG connector
    (reshaped to (bbox_size,)). Inside the NSDFG, a new prep state runs
    ``strided_load<T>`` from the connector array into a fresh W-wide
    transient ``<inner_conn>_vec``. Body memlets that referenced the
    connector are rewritten to reference ``<inner_conn>_vec``.

    Direction "out": symmetric — the body writes to ``<inner_conn>_vec``;
    a finish state runs ``strided_store<T>`` from the transient into the
    bbox-shaped connector; the outer edge carries the bbox subset.

    ``multi_dim_param_dims`` (multi-dim diagonal / linear-combo case):
    when non-empty, the bbox is expanded across ALL listed dims (each
    set to ``vector_width``) instead of only the single stride-1 dim.
    The ``stride`` argument is interpreted as a LINEARISED inter-lane
    stride (e.g. ``N + 1`` for ``A[i, i]``) and the same
    ``strided_load`` / ``strided_store`` intrinsic walks the W elements
    through linear memory.
    """
    assert direction in ("in", "out")
    bbox_shape = list(orig_arr.shape)
    if multi_dim_param_dims:
        for d in multi_dim_param_dims:
            bbox_shape[d] = vector_width
    else:
        # Single-stride-1-dim path (1D strided / s127 / s1111 shape).
        stride_one_indices = [i for i, s in enumerate(orig_arr.strides) if s == 1]
        assert len(stride_one_indices) == 1, (
            f"Strided-inside requires a single stride-1 dim; got {orig_arr.strides}")
        stride_one_idx = stride_one_indices[0]
        edge_b, edge_e, _ = edge.data.subset[stride_one_idx]
        bbox_size = edge_e - edge_b + 1
        bbox_shape[stride_one_idx] = bbox_size

    # Reshape the inner connector array to the bbox shape.
    if inner_conn in inner_sdfg.arrays:
        prev_arr = inner_sdfg.arrays[inner_conn]
        inner_sdfg.remove_data(inner_conn, validate=False)
    else:
        prev_arr = orig_arr
    inner_sdfg.add_array(name=inner_conn,
                         shape=tuple(bbox_shape),
                         dtype=orig_arr.dtype,
                         storage=getattr(prev_arr, "storage", orig_arr.storage),
                         transient=False,
                         find_new_name=False,
                         may_alias=False)

    # Add the W-wide inner transient.
    vec_name = f"__strided_buf_{inner_conn}"
    if vec_name in inner_sdfg.arrays:
        inner_sdfg.remove_data(vec_name, validate=False)
    inner_sdfg.add_array(name=vec_name,
                         shape=(vector_width, ),
                         dtype=orig_arr.dtype,
                         storage=dace.dtypes.StorageType.Register,
                         transient=True,
                         find_new_name=False,
                         may_alias=False)

    dtype_ctype = orig_arr.dtype.ctype

    if direction == "in":
        # Rewrite body memlets: every reference to ``inner_conn`` in body
        # states becomes ``vec_name``. The connector's underlying array
        # (still named ``inner_conn``) stays bbox-shaped — only the prep
        # state reads it; the body sees ``vec_name`` instead.
        #
        # Multi-dim case: the inner connector array was N-D (bbox-shaped)
        # and the body's point-access memlets were N-D too (e.g. ``A[0, 0]``).
        # The W-wide transient ``__strided_buf_A`` is 1-D and holds the
        # W gathered elements (one per lane), so the rename collapses
        # the subset to 1-D ``[0 : W-1]`` — the downstream vector tasklet
        # consumes the full W-wide buffer.
        _flatten_subset = bool(multi_dim_param_dims)
        _flatten_range = dace.subsets.Range([(dace.symbolic.SymExpr(0),
                                              dace.symbolic.SymExpr(vector_width - 1),
                                              dace.symbolic.SymExpr(1))])
        for inner_state in list(inner_sdfg.states()):
            for inner_edge in inner_state.edges():
                if inner_edge.data is not None and inner_edge.data.data == inner_conn:
                    inner_edge.data.data = vec_name
                    if _flatten_subset and inner_edge.data.subset is not None:
                        inner_edge.data.subset = copy.deepcopy(_flatten_range)
            for node in list(inner_state.nodes()):
                if isinstance(node, dace.nodes.AccessNode) and node.data == inner_conn:
                    node.data = vec_name

        # Insert prep state at the start: strided_load(connector, vec).
        old_start = inner_sdfg.start_block
        prep = inner_sdfg.add_state("_strided_load_prep_" + inner_conn, is_start_block=True)
        bbox_an = prep.add_access(inner_conn)
        vec_an = prep.add_access(vec_name)
        tasklet = prep.add_tasklet(
            name=f"_strided_load_{inner_conn}",
            inputs={"_in"},
            outputs={"_out"},
            code=f"strided_load<{dtype_ctype}>(_in, _out, {vector_width}, {stride});",
            language=dace.dtypes.Language.CPP,
        )
        prep.add_edge(bbox_an, None, tasklet, "_in",
                      dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
        prep.add_edge(tasklet, "_out", vec_an, None,
                      dace.memlet.Memlet.from_array(vec_name, inner_sdfg.arrays[vec_name]))
        if old_start is not None and old_start is not prep:
            inner_sdfg.add_edge(prep, old_start, dace.InterstateEdge())

        # Outer edge: re-attach with the bbox subset directly to the NSDFG.
        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, nsdfg_node, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))
    else:
        # Direction "out": rewrite body's writes from inner_conn → vec_name;
        # add a finish state that strided_stores vec → connector.
        # Multi-dim case: also collapse the subset to 1-D ``[0:W-1]``
        # (same reason as the "in" direction above — the W-wide
        # transient gets a vector-store from the body).
        _flatten_subset = bool(multi_dim_param_dims)
        _flatten_range = dace.subsets.Range([(dace.symbolic.SymExpr(0),
                                              dace.symbolic.SymExpr(vector_width - 1),
                                              dace.symbolic.SymExpr(1))])
        for inner_state in list(inner_sdfg.states()):
            for inner_edge in inner_state.edges():
                if inner_edge.data is not None and inner_edge.data.data == inner_conn:
                    inner_edge.data.data = vec_name
                    if _flatten_subset and inner_edge.data.subset is not None:
                        inner_edge.data.subset = copy.deepcopy(_flatten_range)
            for node in list(inner_state.nodes()):
                if isinstance(node, dace.nodes.AccessNode) and node.data == inner_conn:
                    node.data = vec_name

        # Add finish state after every existing sink state.
        sink_states = [s for s in inner_sdfg.states() if len(inner_sdfg.out_edges(s)) == 0]
        finish = inner_sdfg.add_state("_strided_store_finish_" + inner_conn)
        vec_an = finish.add_access(vec_name)
        bbox_an = finish.add_access(inner_conn)
        tasklet = finish.add_tasklet(
            name=f"_strided_store_{inner_conn}",
            inputs={"_in"},
            outputs={"_out"},
            code=f"strided_store<{dtype_ctype}>(_in, _out, {vector_width}, {stride});",
            language=dace.dtypes.Language.CPP,
        )
        finish.add_edge(vec_an, None, tasklet, "_in",
                        dace.memlet.Memlet.from_array(vec_name, inner_sdfg.arrays[vec_name]))
        finish.add_edge(tasklet, "_out", bbox_an, None,
                        dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
        for sink in sink_states:
            if sink is finish:
                continue
            inner_sdfg.add_edge(sink, finish, dace.InterstateEdge())

        # Outer edge: re-attach with the bbox subset directly from the NSDFG.
        state.remove_edge(edge)
        state.add_edge(nsdfg_node, edge.src_conn, edge.dst, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))


def _setup_multi_element_strided_inside_nsdfg(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                               inner_sdfg: dace.SDFG, edge, inner_conn: str, orig_data: str,
                                               orig_arr, vector_width: int, *, elements_per_iter: int,
                                               stride: int, direction: str) -> None:
    """Multi-element-per-iteration strided / contiguous bbox case.

    Generalised K-elements-per-iter pattern: each iteration accesses
    ``K`` consecutive elements per lane; consecutive lanes are
    ``stride`` apart. Total bbox spans ``(W-1)*stride + K`` indices.

    - ``stride == K``: contiguous bbox (TSVC s127 — ``a[2*i], a[2*i+1]``).
    - ``stride > K``: multi-elem scatter/gather with gaps
      (e.g. ``a[4*i], a[4*i+1]`` writes 2 contiguous elements every 4).

    The body's inner connector is shape ``(K, ...)`` with K separate
    tasklet writes / reads (one per intra-iter position). Vectorize
    as K independent stride-``stride`` loads / stores into K W-wide
    transients, one per phase ``p in [0, K)``:

    - Direction "in":   ``strided_load<T>(_in + p, _out, W, stride)``
    - Direction "out":  ``strided_store<T>(_in, _out + p, W, stride)``
    """
    assert direction in ("in", "out")
    assert elements_per_iter >= 2, f"multi-element handler requires K >= 2, got {elements_per_iter}"
    K = int(elements_per_iter)
    W = int(vector_width)
    S = int(stride)
    assert S >= K, f"multi-element handler requires stride >= K; got stride={S}, K={K}"

    bbox_size = (W - 1) * S + K

    # Reshape inner connector to the bbox shape ``((W-1)*stride + K,)``
    # in its stride-1 dim. Other dims preserved from the original array.
    bbox_shape = list(orig_arr.shape)
    stride_one_indices = [i for i, s in enumerate(orig_arr.strides) if s == 1]
    assert len(stride_one_indices) == 1, (
        f"Multi-element-per-iter strided requires a single stride-1 dim; got {orig_arr.strides}")
    stride_one_idx = stride_one_indices[0]
    bbox_shape[stride_one_idx] = bbox_size

    if inner_conn in inner_sdfg.arrays:
        prev_arr = inner_sdfg.arrays[inner_conn]
        inner_sdfg.remove_data(inner_conn, validate=False)
    else:
        prev_arr = orig_arr
    inner_sdfg.add_array(name=inner_conn,
                         shape=tuple(bbox_shape),
                         dtype=orig_arr.dtype,
                         storage=getattr(prev_arr, "storage", orig_arr.storage),
                         transient=False,
                         find_new_name=False,
                         may_alias=False)

    # Allocate K W-wide phase transients.
    phase_names = []
    for p in range(K):
        name = f"__strided_buf_{inner_conn}_p{p}"
        if name in inner_sdfg.arrays:
            inner_sdfg.remove_data(name, validate=False)
        inner_sdfg.add_array(name=name,
                             shape=(W,),
                             dtype=orig_arr.dtype,
                             storage=dace.dtypes.StorageType.Register,
                             transient=True,
                             find_new_name=False,
                             may_alias=False)
        phase_names.append(name)

    dtype_ctype = orig_arr.dtype.ctype
    full_buf_range = dace.subsets.Range([(dace.symbolic.SymExpr(0),
                                          dace.symbolic.SymExpr(W - 1),
                                          dace.symbolic.SymExpr(1))])

    # Rewrite body memlets: each ``<conn>[p]`` reference (single-point
    # subset whose stride-1-dim begin == p) becomes a ``__strided_buf_<conn>_p{p}[0:W-1]``
    # reference. The K access nodes for ``<conn>`` get split by phase.
    for inner_state in list(inner_sdfg.states()):
        # Edges first — capture old data before nodes are renamed.
        for inner_edge in inner_state.edges():
            if inner_edge.data is None or inner_edge.data.data != inner_conn:
                continue
            if inner_edge.data.subset is None or len(inner_edge.data.subset) <= stride_one_idx:
                continue
            b, ee, _ = inner_edge.data.subset[stride_one_idx]
            try:
                # Phase is the integer value of the begin in the stride-1 dim.
                p = int(b)
            except Exception:
                continue
            if not (0 <= p < K) or int(ee) != p:
                continue
            inner_edge.data.data = phase_names[p]
            inner_edge.data.subset = copy.deepcopy(full_buf_range)
        # Access nodes: rename based on which edge uses them.  An access
        # node is connected to ONE edge in the body (write tasklet → AN
        # or AN → read tasklet); rename to the matching phase.
        for node in list(inner_state.nodes()):
            if not (isinstance(node, dace.nodes.AccessNode) and node.data == inner_conn):
                continue
            # Find the phase via the connected edge(s).
            for e2 in list(inner_state.in_edges(node)) + list(inner_state.out_edges(node)):
                if e2.data is not None and e2.data.data in phase_names:
                    node.data = e2.data.data
                    break

    if direction == "in":
        # Prep state: K ``strided_load`` tasklets, each at stride K with offset p.
        old_start = inner_sdfg.start_block
        prep = inner_sdfg.add_state("_multi_elem_load_prep_" + inner_conn, is_start_block=True)
        bbox_an = prep.add_access(inner_conn)
        for p in range(K):
            vec_an = prep.add_access(phase_names[p])
            tasklet = prep.add_tasklet(
                name=f"_multi_elem_load_{inner_conn}_p{p}",
                inputs={"_in"},
                outputs={"_out"},
                code=f"strided_load<{dtype_ctype}>(_in + {p}, _out, {W}, {S});",
                language=dace.dtypes.Language.CPP,
            )
            prep.add_edge(bbox_an, None, tasklet, "_in",
                          dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
            prep.add_edge(tasklet, "_out", vec_an, None,
                          dace.memlet.Memlet.from_array(phase_names[p], inner_sdfg.arrays[phase_names[p]]))
        if old_start is not None and old_start is not prep:
            inner_sdfg.add_edge(prep, old_start, dace.InterstateEdge())

        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, nsdfg_node, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))
    else:
        # Finish state: K ``strided_store`` tasklets.
        sink_states = [s for s in inner_sdfg.states() if len(inner_sdfg.out_edges(s)) == 0]
        finish = inner_sdfg.add_state("_multi_elem_store_finish_" + inner_conn)
        bbox_an = finish.add_access(inner_conn)
        for p in range(K):
            vec_an = finish.add_access(phase_names[p])
            tasklet = finish.add_tasklet(
                name=f"_multi_elem_store_{inner_conn}_p{p}",
                inputs={"_in"},
                outputs={"_out"},
                code=f"strided_store<{dtype_ctype}>(_in, _out + {p}, {W}, {S});",
                language=dace.dtypes.Language.CPP,
            )
            finish.add_edge(vec_an, None, tasklet, "_in",
                            dace.memlet.Memlet.from_array(phase_names[p], inner_sdfg.arrays[phase_names[p]]))
            finish.add_edge(tasklet, "_out", bbox_an, None,
                            dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
        for sink in sink_states:
            if sink is finish:
                continue
            inner_sdfg.add_edge(sink, finish, dace.InterstateEdge())

        state.remove_edge(edge)
        state.add_edge(nsdfg_node, edge.src_conn, edge.dst, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))


def _process_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                   vector_width: int, vector_storage: dace.dtypes.StorageType, *, direction: str) -> Set[str]:
    """
    Rewire one direction of NSDFG-boundary edges to flow through a freshly
    allocated vector access node. Single implementation for both
    ``process_in_edges`` and ``process_out_edges``.

    direction="in": for each movable arr, rewire ``src --[orig]--> nsdfg`` into
        ``src --[orig, copy_subset]--> vec_an --[vec_from_array]--> nsdfg``.
    direction="out": for each movable arr, rewire ``nsdfg --[orig]--> dst`` into
        ``nsdfg --[vec_from_array]--> vec_an --[orig, copy_subset]--> dst``,
        and also detect inout connectors (already created on the in-side by
        a prior call with ``direction="in"``) so the existing vector name
        is reused instead of clashing.
    """
    assert direction in ("in", "out"), direction
    assert isinstance(nsdfg_node, dace.nodes.NestedSDFG)
    inner_sdfg = nsdfg_node.sdfg
    edges_by_connector = state.in_edges_by_connector if direction == "in" else state.out_edges_by_connector

    vectorized_datanames: Set[str] = set()
    for movable_arr_name, subset in movable_arrays:
        edges = list(edges_by_connector(nsdfg_node, movable_arr_name))
        assert len(edges) <= 1

        for e in edges:
            orig_data = e.data.data
            orig_arr = state.sdfg.arrays[orig_data]
            inner_conn = e.dst_conn if direction == "in" else e.src_conn

            # Strided detection: BEFORE prepare_vectorized_array, check the
            # OUTER edge's subset. Two patterns:
            #
            # - 1D strided: a single stride-1 dim has bbox > W. Stride
            #   is ``(bbox - 1) / (W - 1)``. Catches s127 / s1111 shape.
            #
            # - Multi-dim strided (e.g. diagonal ``A[i, i]``): two or
            #   more dims have a W-wide bbox AND the original subset's
            #   begin in those dims is per-lane-incrementing. The
            #   linearised stride is computed from the array's per-dim
            #   strides. Catches ``A[i, i]``, ``A[2*i, i]``, ``A[i, 2*i]``
            #   patterns under the NSDFG-wrapped (P1+P2) path.
            import sympy as _sp
            stride_one_indices = [i for i, s in enumerate(orig_arr.strides) if s == 1]
            is_strided = False
            stride_value = 1
            multi_dim_dims = ()

            # Wide-bbox dims (per-dim bbox > 1 and step 1).
            wide_dims = []
            for d, (b, ee, s) in enumerate(e.data.subset):
                if s != 1:
                    continue
                try:
                    bw = int(ee - b + 1)
                except (TypeError, ValueError):
                    bw = None
                if bw is not None and bw > 1:
                    wide_dims.append((d, bw))

            multi_elem_per_iter = 0  # >0 means K-elements-per-iter (with stride_value)
            if len(wide_dims) == 1 and len(stride_one_indices) == 1 and wide_dims[0][0] == stride_one_indices[0]:
                # Generalised K-elements-per-iter at inter-lane stride S:
                # bbox = (W-1)*S + K. ``K`` is the inner connector's
                # stride-1 dim size.
                bbox_vol = wide_dims[0][1]
                if bbox_vol > vector_width and vector_width > 1:
                    inner_arr = inner_sdfg.arrays.get(inner_conn)
                    inner_dim0 = None
                    if inner_arr is not None and len(inner_arr.shape) >= 1:
                        try:
                            inner_dim0 = int(inner_arr.shape[-1])
                        except Exception:
                            inner_dim0 = None
                    K_candidate = inner_dim0 if (inner_dim0 is not None and inner_dim0 >= 1) else 1
                    handled = False
                    if (bbox_vol - K_candidate) % (vector_width - 1) == 0:
                        S_value = (bbox_vol - K_candidate) // (vector_width - 1)
                        if S_value >= K_candidate:
                            is_strided = True
                            stride_value = S_value
                            if K_candidate > 1:
                                multi_elem_per_iter = K_candidate
                            handled = True
                    if not handled:
                        # Fall back to K=1.
                        if (bbox_vol - 1) % (vector_width - 1) == 0:
                            is_strided = True
                            stride_value = (bbox_vol - 1) // (vector_width - 1)
                            handled = True
                    if not handled:
                        raise NotImplementedError(
                            f"_process_edges (direction={direction!r}): outer subset on {orig_data} "
                            f"has bbox volume {bbox_vol}; doesn't match (W-1)*S+K for any K in "
                            f"[1, {inner_arr.shape if inner_arr else None}] for vector_width={vector_width}.")
            elif len(wide_dims) >= 2 and vector_width > 1:
                # Multi-dim strided. Each wide dim must be W-bbox; the
                # inter-lane stride is the sum of per-dim
                # ``arr.strides[d] * coeff_of_inner_map_param`` across
                # the wide dims. The inner map param isn't directly
                # known to ``_process_edges``, but it's the symbol that
                # is *common* across the wide dims' begins (every wide
                # dim begin has the same lane-increment coefficient).
                if all(bw == vector_width for _, bw in wide_dims):
                    # Identify candidate map-param symbol: pick the free
                    # symbol shared by every wide dim's begin expression.
                    # Critically: pick the ACTUAL sympy/dace symbol
                    # instance from the begins, not a fresh
                    # ``_sp.Symbol(name)`` — dace.symbolic.symbol is a
                    # subclass with its own identity, and
                    # ``beg.coeff(_sp.Symbol(name))`` returns 0 even
                    # when the begin contains the same-named dace
                    # symbol.
                    shared_syms_by_name = None
                    begins = [e.data.subset[d][0] for d, _ in wide_dims]
                    sym_by_name = {}
                    for beg in begins:
                        if not hasattr(beg, "free_symbols"):
                            shared_syms_by_name = set()
                            break
                        bnames = set()
                        for sym in beg.free_symbols:
                            sym_by_name.setdefault(str(sym), sym)
                            bnames.add(str(sym))
                        shared_syms_by_name = bnames if shared_syms_by_name is None else shared_syms_by_name & bnames
                    if shared_syms_by_name:
                        map_sym_name = sorted(shared_syms_by_name)[0]
                        map_sym = sym_by_name[map_sym_name]
                        linear_stride = 0
                        try:
                            for d, _bw in wide_dims:
                                beg = e.data.subset[d][0]
                                coeff = beg.coeff(map_sym)
                                linear_stride = linear_stride + coeff * orig_arr.strides[d]
                            is_strided = True
                            stride_value = linear_stride
                            multi_dim_dims = tuple(d for d, _ in wide_dims)
                        except Exception:
                            pass

            if is_strided:
                # Strided boundary (inside the NSDFG): the FULL bbox passes
                # through to the NSDFG connector (kept at its original name,
                # array reshaped to bbox shape). A prep / finish state inside
                # the NSDFG performs ``strided_load<T>`` (in) or
                # ``strided_store<T>`` (out) into / from a new W-wide
                # transient. The body's memlets are rewritten to reference
                # the W-wide transient. Note: strided arrays are NOT added
                # to ``vectorized_datanames`` so the downstream connector
                # rename (``movable_data`` → ``VecNameScheme.make(movable_data)``)
                # skips them — the connector keeps the original name.
                if multi_elem_per_iter > 0:
                    _setup_multi_element_strided_inside_nsdfg(state, nsdfg_node, inner_sdfg, e,
                                                              inner_conn, orig_data, orig_arr,
                                                              vector_width,
                                                              elements_per_iter=multi_elem_per_iter,
                                                              stride=stride_value,
                                                              direction=direction)
                else:
                    _setup_strided_inside_nsdfg(state, nsdfg_node, inner_sdfg, e, inner_conn, orig_data,
                                                orig_arr, vector_width, stride_value, direction=direction,
                                                multi_dim_param_dims=multi_dim_dims)
                continue

            inout_data_name = None
            if direction == "out" and isinstance(e.src, dace.nodes.NestedSDFG) and e.src_conn in e.src.in_connectors:
                # Inout connector: a sibling ``process_in_edges`` call already
                # allocated the vector array; reuse its name rather than
                # minting a fresh one.
                ie_datas = {ie.data.data for ie in state.in_edges_by_connector(nsdfg_node, e.src_conn)}
                assert len(ie_datas) == 1
                ie_data = ie_datas.pop()
                _vec_prefix = VecNameScheme.make(orig_data)
                assert orig_data == ie_data or ie_data.startswith(_vec_prefix), (
                    f"{orig_data} != {ie_data} and {ie_data} not startswith {_vec_prefix} "
                    f"(from {inner_conn}) not in {state.sdfg.arrays}")
                inout_data_name = ie_data

            prev_subset = copy.deepcopy(subset)
            vector_dataname, inner_offset = prepare_vectorized_array(state, inner_sdfg, inner_conn, orig_data, orig_arr,
                                                                     subset, vector_width, vector_storage,
                                                                     inout_data_name is not None, inout_data_name)

            # Catch collisions: two unrelated edges picking the same vector
            # name. Skipped when ``inout_data_name`` deliberately reused it.
            if inout_data_name is None:
                assert vector_dataname not in vectorized_datanames
            vectorized_datanames.add(vector_dataname)

            copy_subset = compute_edge_subset(e.data.subset, prev_subset, orig_arr, inner_offset, vector_width)

            an = state.add_access(vector_dataname)
            an.setzero = True
            state.remove_edge(e)

            vec_arr_desc = state.sdfg.arrays[vector_dataname]
            if direction == "in":
                # src --[orig, copy_subset]--> an --[vec_from_array]--> nsdfg
                state.add_edge(e.src, e.src_conn, an, None, dace.memlet.Memlet(data=orig_data, subset=copy_subset))
                state.add_edge(an, None, e.dst, e.dst_conn,
                               dace.memlet.Memlet.from_array(vector_dataname, vec_arr_desc))
            else:
                # nsdfg --[vec_from_array]--> an --[orig, copy_subset]--> dst
                assert e.src == nsdfg_node
                assert e.src_conn is not None
                assert len(set(state.out_edges_by_connector(nsdfg_node, e.src_conn))) == 0
                state.add_edge(e.src, e.src_conn, an, None,
                               dace.memlet.Memlet.from_array(vector_dataname, vec_arr_desc))
                state.add_edge(an, None, e.dst, e.dst_conn, dace.memlet.Memlet(data=orig_data, subset=copy_subset))

    return vectorized_datanames


def process_in_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                     vector_width: int, vector_storage: dace.dtypes.StorageType) -> Set[str]:
    """Rewire ``src -> nsdfg`` boundary edges through a freshly allocated vector access."""
    return _process_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage, direction="in")


def process_out_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                      vector_width: int, vector_storage: dace.dtypes.StorageType) -> Set[str]:
    """Rewire ``nsdfg -> dst`` boundary edges through a freshly allocated vector access."""
    return _process_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage, direction="out")


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
            vec_arr_name = VecNameScheme.make_indexed(unmovable_arr_name, i)
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
            if edge.data.data is not None and (PackedNameScheme.is_packed(edge.data.data)
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
            if PackedNameScheme.is_packed(node.data):
                continue

            ies = {ie for ie in inner_state.in_edges(node) if ie.data.data is not None}
            oes = {oe for oe in inner_state.out_edges(node) if oe.data.data is not None}

            # Do not check packed storage
            for e in ies.union(oes):
                if isinstance(e.src, dace.nodes.AccessNode) and PackedNameScheme.is_packed(e.src.data):
                    continue
                if isinstance(e.dst, dace.nodes.AccessNode) and PackedNameScheme.is_packed(e.dst.data):
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

    def _emit_unmovable_copy(target_state, unmovable_name, vec_name, subset, direction):
        """Splice a ``vector_copy`` tasklet inside ``target_state``.

        direction="in":  orig_access --[orig, subset]--> tasklet --[vec_from_array]--> vec_access
        direction="out": vec_access --[vec_from_array]--> tasklet --[orig, subset]--> orig_access
        """
        assert direction in ("in", "out"), direction
        orig_access = target_state.add_access(unmovable_name)
        orig_access.setzero = True
        v_access = target_state.add_access(vec_name)
        v_access.setzero = True
        vec_arr = target_state.sdfg.arrays[vec_name]
        tasklet_name = "_unmovable_copy_in" if direction == "in" else "_unmovable_copy_out"
        t = target_state.add_tasklet(
            name=tasklet_name,
            inputs={"_in"},
            outputs={"_out"},
            code=f"vector_copy<{dace.dtypes.TYPECLASS_TO_STRING[vec_arr.dtype]}, {vector_width}>(_out, _in);",
            language=dace.dtypes.Language.CPP)
        orig_memlet = dace.memlet.Memlet(data=unmovable_name, subset=copy.deepcopy(subset))
        vec_memlet = dace.memlet.Memlet.from_array(vec_name, vec_arr)
        if direction == "in":
            target_state.add_edge(orig_access, None, t, "_in", orig_memlet)
            target_state.add_edge(t, "_out", v_access, None, vec_memlet)
        else:
            target_state.add_edge(v_access, None, t, "_in", vec_memlet)
            target_state.add_edge(t, "_out", orig_access, None, orig_memlet)

    # Insert copy-ins and outs
    name_to_subset_map = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
        # If a packed stored, then continue
        # Add a unique vector array for each unique subset
        desc = inner_sdfg.arrays[unmovable_arr_name]

        if unmovable_arr_name in read_set:
            for i, subset in enumerate(subsets):
                vec_arr_name = VecNameScheme.make_indexed(unmovable_arr_name, i)
                name_to_subset_map[vec_arr_name] = subset

                # The copy-in state has to defer to after every symbol the
                # subset reads is in scope; ``find_copy_in_state`` walks the
                # NSDFG forward from start until that's true.
                copy_in_state = find_copy_in_state(inner_sdfg, nsdfg_node, {str(s)
                                                                            for s in subset.free_symbols},
                                                   unmovable_arr_name)
                _emit_unmovable_copy(copy_in_state, unmovable_arr_name, vec_arr_name, subset, "in")

        if unmovable_arr_name in write_set:
            for i, subset in enumerate(subsets):
                vec_arr_name = VecNameScheme.make_indexed(unmovable_arr_name, i)
                name_to_subset_map[vec_arr_name] = subset
                _emit_unmovable_copy(copy_out_state, unmovable_arr_name, vec_arr_name, subset, "out")

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
        inner_sdfg.replace_dict({dataname: VecNameScheme.make(dataname)})

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
            nsdfg_node.add_in_connector(VecNameScheme.make(inc), force=True)

    for outc in nsdfg_out_conns:
        if outc in movable_datas:
            nsdfg_node.add_out_connector(VecNameScheme.make(outc), force=True)

    # Update connector names
    # Remove movable datanames from connectors and replace with vec variant.
    # Some scalars / arrays will be not vectorized and thus not have ``_vec``
    # suffix; make sure we only connect the arrays that DO have the suffix.
    # Inout connectors get the same suffix on both directions (per the
    # NameScheme directive); the route here is symmetric for ``in_edges``
    # and ``out_edges``.
    for movable_data in movable_datas:
        vec_data = VecNameScheme.make(movable_data)
        for ie in state.in_edges(nsdfg_node):
            if ie.dst_conn is not None and ie.dst_conn == movable_data and vec_data in nsdfg_node.in_connectors:
                assert vec_data in nsdfg_node.in_connectors, f"{vec_data} not in {nsdfg_node.in_connectors}"
                assert len(set(state.in_edges_by_connector(nsdfg_node, vec_data))) == 0, (
                    f"There are edges connected to {vec_data}: "
                    f"{set(state.in_edges_by_connector(nsdfg_node, vec_data))}")
                ie.dst_conn = vec_data
        for oe in state.out_edges(nsdfg_node):
            if oe.src_conn is not None and oe.src_conn == movable_data and vec_data in nsdfg_node.out_connectors:
                assert vec_data in nsdfg_node.out_connectors, f"{vec_data} not in {nsdfg_node.out_connectors}"
                assert len(set(state.out_edges_by_connector(nsdfg_node, vec_data))) == 0, (
                    f"There are edges connected to {vec_data}: "
                    f"{set(state.out_edges_by_connector(nsdfg_node, vec_data))}")
                oe.src_conn = vec_data

    # Move vector data above the vector map, it makes merging overlapping accesses easier.
    # Skip when the AccessNode's in-edge does NOT come directly from a MapEntry:
    # that's the strided-load boundary, where ``_process_edges`` inserts a
    # ``strided_load<T>`` CPP tasklet between map_entry and the vec AccessNode.
    # The tasklet is correctly placed inside the map scope; sifting the vec
    # AccessNode above the map would break the tasklet wiring.
    sdict = state.scope_dict()
    for ie in state.in_edges(nsdfg_node):
        if isinstance(ie.src, dace.nodes.AccessNode) and ie.data.data in inserted_array_names:
            an = ie.src
            an_in_edges = state.in_edges(an)
            if len(an_in_edges) == 1 and isinstance(an_in_edges[0].src, dace.nodes.MapEntry):
                sift_access_node_up(state, an, sdict[an])

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
