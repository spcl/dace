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
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbols

_ITER_MASK_PREFIX = "_iter_mask"


def _iter_mask_name(sdfg: dace.SDFG) -> Optional[str]:
    """Return the ``_iter_mask`` array name in ``sdfg.arrays``, or ``None``.

    When a strided load / store runs inside a masked-remainder NSDFG
    (``GenerateIterationMask`` attached ``_iter_mask: bool[W]`` to it),
    the boundary ``strided_{load,store}`` must use the ``_masked``
    runtime variant: at the tail only ``R < W`` lanes are in bounds, so
    an unmasked W-wide store writes ``arr[N .. base + (W-1)*stride]``
    past the real array (the diagonal-scatter OOB write that corrupts
    the heap and aborts later innocent tests with ``free(): invalid
    size``). Returns ``None`` for the unmasked main body.

    :param sdfg: The inner SDFG receiving the prep / finish state.
    :returns: The mask array name, or ``None`` if absent.
    """
    for name in sdfg.arrays:
        if name == _ITER_MASK_PREFIX or name.startswith(_ITER_MASK_PREFIX + "_"):
            return name
    return None


_STRIDED_LOAD_PREP_PREFIX = "_strided_load_prep_"
_STRIDED_STORE_FINISH_PREFIX = "_strided_store_finish_"
_MULTI_ELEM_LOAD_PREP_PREFIX = "_multi_elem_load_prep_"
_MULTI_ELEM_STORE_FINISH_PREFIX = "_multi_elem_store_finish_"
_STRIDED_AUX_STATE_PREFIXES = (_STRIDED_LOAD_PREP_PREFIX, _STRIDED_STORE_FINISH_PREFIX,
                               _MULTI_ELEM_LOAD_PREP_PREFIX, _MULTI_ELEM_STORE_FINISH_PREFIX)


def _is_strided_aux_state(state: SDFGState) -> bool:
    """Return whether ``state`` is a strided load/store prep or finish state.

    These auxiliary states are minted by ``_setup_strided_inside_nsdfg`` /
    ``_setup_multi_element_strided_inside_nsdfg`` to run the boundary
    ``strided_{load,store}`` against the bbox connector. The body-memlet
    rename loops must skip them: an inout (RMW) connector is processed
    twice — ``in`` then ``out`` — and the ``out`` pass would otherwise
    rename the bbox-connector read inside the ``in`` pass's load-prep
    state to the W-wide buffer, so the strided load reads the
    uninitialised buffer instead of the connector (the s2101 diagonal /
    s2275 column RMW corruption).

    :param state: An inner-SDFG state.
    :returns: ``True`` if it is a strided prep / finish state.
    """
    return any(state.label.startswith(p) for p in _STRIDED_AUX_STATE_PREFIXES)


def _compute_subset_union(subsets):
    """Bounding-box union of overlapping subsets and its per-dim shape.

    Reuses the exact union / extent math of the standalone
    :class:`~dace.transformation.passes.vectorization.fuse_overlapping_loads.FuseOverlappingLoads`
    pass, so the baked-in fusion (knob ``fuse_overlapping_loads``) and the
    legacy standalone pass stay numerically identical: the bounding box is
    the left fold of :meth:`dace.subsets.Range.union` over ``subsets`` and
    each dimension extent is ``int_floor((end + 1) - begin, step)``.
    ``int_floor`` (not sympy ``//``) is used so a strided window emits a
    correct C++ integer extent rather than a broken ``floor((x - 1) / s)``.

    :param subsets: Iterable of :class:`dace.subsets.Range` accesses to the
        same array (the multiple per-lane / per-stencil-arm views that the
        unmovable-array classification gathered).
    :returns: ``(union_subset, union_shape)`` — the bounding-box
        :class:`dace.subsets.Range` and its integer per-dim shape list.
    """
    union_subset = None
    for s in subsets:
        union_subset = s if union_subset is None else union_subset.union(s)
    union_shape = [dace.symbolic.int_floor((e + 1) - b, s) for (b, e, s) in union_subset]
    return union_subset, union_shape


def _is_trivial_exact_vlen_copy(subset, arr: dace.data.Array, vector_width: int, masked: bool) -> bool:
    """Decide whether a single-subset NSDFG-boundary access may stay *outside*.

    The R4/R5 copy-inside rule: a boundary access is materialised as a
    plain ``vector_copy<T, W>`` access node *outside* the NSDFG (the
    "movable" path) **only** when it is a trivial exact-VLEN, step-1,
    contiguous window — i.e. exactly ``vector_width`` consecutive
    elements on the array's stride-1 dimension with every other
    dimension collapsed to length 1. Anything else must be staged
    *inside* the NSDFG body, where :func:`emit_staging_copy` is in scope
    and gates every lane against the in-scope ``_iter_mask`` / the real
    array's extent. The outside path has no such gate, so a non-trivial
    access placed there silently produces wrong results:

    - **Strided** (step ``> 1`` on the contiguous dim, e.g. ``b[4*i]``):
      an outside ``vector_copy`` reads ``W`` *contiguous* elements, not
      the strided lanes — wrong numerics. Belongs inside via the
      ``strided_{load,store}`` handlers.
    - **Non-exact-W extent** (e.g. a masked-remainder tail with only
      ``R < W`` valid elements, or an un-collapsed multi-dim window):
      an outside W-wide copy OOB-reads / -writes the real array.
    The ``masked`` flag is accepted for call-site clarity but does *not*
    by itself force the inside path: a trivial exact-W contiguous copy
    in a masked-remainder body is already correctly per-lane gated by
    :func:`emit_staging_copy` (``mask_name`` / ``gate_extent``) on the
    movable path, and strided masked accesses are already routed inside
    by the ``strided_{load,store}`` handlers. Forcing *every* masked
    single-subset access inside instead reclassifies those
    already-correct accesses into the per-subset ``_emit_unmovable_copy``
    path, which does not handle masked strided / diagonal shapes —
    regressing them. The genuine correctness gain here is therefore the
    *shape* gate (strided / partial / non-collapsed → inside), not a
    blanket masked rule.

    Multi-subset arrays are handled by the union / per-subset
    classifier before this predicate is consulted; this only refines the
    single-subset decision (previously: movable iff its free symbols are
    available outside — which silently kept strided / partial accesses
    outside and corrupted them).

    :param subset: The single boundary access :class:`dace.subsets.Range`.
    :param arr: The inner-SDFG array descriptor for the accessed data.
    :param vector_width: The SIMD lane count ``W``.
    :param masked: ``True`` if the enclosing NSDFG is a masked-remainder
        body (an ``_iter_mask`` is in scope). Accepted for call-site
        clarity; does not by itself force the inside path (see above).
    :returns: ``True`` iff the access may stay outside as a plain
        ``vector_copy<T, W>``; ``False`` to force inside staging.
    """
    strides = list(arr.strides)
    contig = strides.index(1) if 1 in strides else len(strides) - 1
    for d, (b, e, s) in enumerate(subset):
        extent = e - b + 1
        if d == contig:
            if s != 1:
                return False
            if str(dace.symbolic.simplify(extent - vector_width)) != "0":
                return False
        else:
            if str(dace.symbolic.simplify(extent - 1)) != "0":
                return False
    return True


def emit_staging_copy(state: SDFGState, src: dace.nodes.Node, src_conn, dst: dace.nodes.Node, dst_conn,
                      real_memlet: dace.memlet.Memlet, buf_name: str, vector_width: int, direction: str, *,
                      mask_name: Optional[str] = None, gate_extent: bool = False) -> None:
    """Splice one staging copy between a real array and a W-wide buffer.

    The single emitter behind every vector-staging boundary copy
    (``_emit_unmovable_copy``, ``_process_edges`` movable in/out, and
    ``Vectorize._copy_in_and_copy_out``). In a masked remainder only
    ``R < W`` lanes are in bounds, so an unconditional W-wide copy
    OOB-reads / -writes the real (caller, non-W-padded) array at the
    tail — an OOB write corrupts the heap. Each lane is gated:

    - ``mask_name`` set: the copy lives inside the masked-remainder body
      NSDFG; gate by the in-scope ``_iter_mask`` and wire a ``_mask``
      connector.
    - ``gate_extent`` set: the copy lives at the outer NSDFG boundary
      where ``_iter_mask`` is not in scope; gate by the real array's own
      extent (``base + l < extent``) — the algebraic equivalent of the
      iteration mask, correct for every access shape.
    - neither: plain ``vector_copy<T, W>`` (main loop / scalar remainder;
      unchanged, no per-lane overhead).

    A length-1 array or ``Scalar`` source on the ``"in"`` side is a
    loop-invariant broadcast (e.g. TSVC s293 ``a0 = a[0]; a[i] = a0``):
    it must replicate element 0 to every lane, never read ``W`` elements
    from a 1-element source (that OOB-reads ``W-1`` elements past it and
    corrupts the heap). This case ignores the mask/extent guards — a
    broadcast is in-bounds and valid for every lane.

    :param state: State to splice into.
    :param src: Edge source node.
    :param src_conn: Edge source connector.
    :param dst: Edge destination node.
    :param dst_conn: Edge destination connector.
    :param real_memlet: Memlet on the real array (its W-wide subset).
    :param buf_name: The W-wide staging-buffer array name.
    :param vector_width: Lane count.
    :param direction: ``"in"`` (real -> buffer) or ``"out"`` (buffer -> real).
    :param mask_name: In-scope ``_iter_mask`` name, or ``None``.
    :param gate_extent: Gate by the real array's extent instead of a mask.
    """
    assert direction in ("in", "out"), direction
    W = int(vector_width)
    buf_arr = state.sdfg.arrays[buf_name]
    ctype = dace.dtypes.TYPECLASS_TO_STRING[buf_arr.dtype]

    src_desc = state.sdfg.arrays.get(real_memlet.data)
    is_broadcast = (direction == "in" and src_desc is not None
                    and (isinstance(src_desc, dace.data.Scalar) or str(src_desc.total_size) == "1"))
    if is_broadcast:
        # Read element 0 once (single-element memlet -> scalar connector)
        # and replicate to every lane. Never a W-wide read of a 1-element
        # source.
        t = state.add_tasklet(name=f"_stage_in_bcast_{buf_name}",
                               inputs={"_in"},
                               outputs={"_out"},
                               code=f"for (int _l = 0; _l < {W}; ++_l) {{ _out[_l] = _in; }}",
                               language=dace.dtypes.Language.CPP)
        state.add_edge(src, src_conn, t, "_in", dace.memlet.Memlet(f"{real_memlet.data}[0]"))
        state.add_edge(t, "_out", dst, dst_conn, dace.memlet.Memlet(f"{buf_name}[0:{W}]"))
        return

    if mask_name is not None:
        guard = "_mask[_l]"
    elif gate_extent:
        arr = state.sdfg.arrays[real_memlet.data]
        strides = list(arr.strides)
        contig = strides.index(1) if 1 in strides else len(strides) - 1
        base = real_memlet.subset[contig][0]
        extent = arr.shape[contig]
        guard = f"(({base}) + _l) < ({extent})"
    else:
        guard = None

    if guard is None:
        code = f"vector_copy<{ctype}, {W}>(_out, _in);"
    elif direction == "in":
        code = (f"for (int _l = 0; _l < {W}; ++_l) {{\n"
                f"    if ({guard}) _out[_l] = _in[_l]; else _out[_l] = 0;\n"
                f"}}")
    else:
        code = (f"for (int _l = 0; _l < {W}; ++_l) {{\n"
                f"    if ({guard}) _out[_l] = _in[_l];\n"
                f"}}")

    t = state.add_tasklet(name=f"_stage_{direction}_{buf_name}",
                          inputs={"_in"},
                          outputs={"_out"},
                          code=code,
                          language=dace.dtypes.Language.CPP)
    buf_memlet = dace.memlet.Memlet(f"{buf_name}[0:{W}]")
    if direction == "in":
        state.add_edge(src, src_conn, t, "_in", copy.deepcopy(real_memlet))
        state.add_edge(t, "_out", dst, dst_conn, buf_memlet)
    else:
        state.add_edge(src, src_conn, t, "_in", buf_memlet)
        state.add_edge(t, "_out", dst, dst_conn, copy.deepcopy(real_memlet))
    if mask_name is not None:
        t.add_in_connector("_mask", dtype=dace.dtypes.pointer(dace.bool_), force=True)
        mask_an = state.add_access(mask_name)
        state.add_edge(mask_an, None, t, "_mask", dace.memlet.Memlet(f"{mask_name}[0:{W}]"))


def get_vector_max_access_ranges(state: SDFGState, node: dace.nodes.NestedSDFG) -> Dict[str, str]:
    """Map each vector-map param to the end bound of the outer data-parallel map.

    Walks ``nsdfg -> vector_map -> data_map`` and matches each vector-map
    ``begin`` (canonicalised via ``dace.symbolic.simplify``) to a data-map
    ``begin``, returning that data-map's end bound. For
    ``map i=0:N -> map i_v=i:i+4:4 -> NestedSDFG`` the result is
    ``{'i_v': 'N - 1'}``.

    :param state: The SDFG state containing the nested SDFG node.
    :param node: The nested SDFG node whose vector access ranges to determine.
    :returns: Dictionary mapping vector-map param names to the outer data-map's
        end bound (string repr). This is the upper iteration bound, not the
        per-memlet access range.
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

    Works for any node type (Tasklet, NestedSDFG, AccessNode, MapEntry, ...).

    :param root_sdfg: SDFG to search recursively.
    :param node: Node to locate.
    :returns: The state containing ``node``.
    :raises Exception: if the node is found in a non-state container, or if
        it is not found at all.
    """
    for n, g in root_sdfg.all_nodes_recursive():
        if n == node:
            if not isinstance(g, dace.SDFGState):
                raise Exception(f"Expected a SDFGState container for {node}, got {type(g).__name__} ({g})")
            return g
    raise Exception(f"State containing the node ({node}) not found in the root SDFG ({root_sdfg.label})")


def check_nsdfg_connector_array_shapes_match(parent_state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG):
    """Validate that NSDFG connector arrays match their memlet subset shapes.

    Checks input and output edges against four expected shape interpretations
    (full / strided / collapsed-full / collapsed-strided). Validation-only;
    use ``fix_nsdfg_connector_array_shapes_mismatch`` to correct mismatches.

    :param parent_state: State in the parent SDFG containing the NSDFG node.
    :param nsdfg_node: The NSDFG node whose connector shapes to validate.
    :raises AssertionError: if a connector array shape matches none of the
        expected interpretations.
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
        # Analysis-only extent check (this shape is ONLY compared against
        # connector_array.shape, never emitted) — sympy ``//`` is the
        # blessed exception here; the codegen-reaching rebuild in
        # fix_nsdfg_connector_array_shapes_mismatch uses int_floor.
        expected_shape_full = tuple([(end + 1 - begin).simplify() for begin, end, step in subset])

        expected_shape_strided = tuple([(((end + 1 - begin) // step)).simplify() for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([(((end + 1 - begin) // step)).simplify()
                                                  for begin, end, step in subset
                                                  if (((end + 1 - begin) // step)).simplify() != 1])

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

        expected_shape_strided = tuple([((end + 1 - begin) // step) for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([(end + 1 - begin) for begin, end, step in subset
                                               if (end + 1 - begin) != 1])

        expected_shape_collapsed_strided = tuple([((end + 1 - begin) // step) for begin, end, step in subset
                                                  if ((end + 1 - begin) // step) != 1])

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


def _raise_on_expansion_rebuild_mismatch(connector_name: str,
                                         original_shape: tuple,
                                         new_shape: tuple,
                                         expected_full: tuple,
                                         expected_strided: tuple,
                                         expected_collapsed_strided: tuple,
                                         *,
                                         direction: str,
                                         vector_width: Optional[int] = None) -> None:
    """Guard ``fix_nsdfg_connector_array_shapes_mismatch`` against corrupting rebuilds.

    Four rebuilds are legitimate: narrowing (every new dim <= original),
    drop-dims (lower rank), vector-widening (a halo dim grown by exactly
    ``vector_width - 1``), and placeholder expansion (original all-1s, or a
    1-D ``(K,)`` widened to a strided bounding box). A rank-equal rebuild that
    strictly enlarges a non-1 original dim is corrupting and rejected.

    :param connector_name: Connector being rebuilt (for the error message).
    :param original_shape: Connector array shape before the rebuild.
    :param new_shape: Proposed (collapsed-full) shape.
    :param expected_full: Full expected shape (for the error message).
    :param expected_strided: Strided expected shape (for the error message).
    :param expected_collapsed_strided: Collapsed-strided expected shape.
    :param direction: ``"in"`` or ``"out"`` (for the error message).
    :param vector_width: When set, enables the vector-widening / strided-bbox
        exceptions.
    :raises ValueError: if the rebuild expands a non-placeholder dim or grows
        the rank.
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
            # ``shape`` entries are always int or sympy Expr -- both expose
            # ``__sub__``; no defensive guard required.
            diff = _int_or_none(new_d - orig_d)
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
                and all(d is not None for d in per_dim_growth) and any(d == vector_width - 1 for d in per_dim_growth)):
            expands_real_dim = False

        # Strided-bbox exception: a 1-D ``(K,)`` connector (K elements
        # per iter) widened to the strided bounding box ``(W-1)*S + K``
        # for some inter-lane stride ``S >= 1`` — the legitimate reshape
        # the strided / multi-element-strided handlers
        # (``_setup_strided_inside_nsdfg`` /
        # ``_setup_multi_element_strided_inside_nsdfg``) need, e.g. a K=2
        # stride-4 gather ``(2,) -> (30,)`` at W=8 (7*4 + 2). The inner
        # body still accesses it K-wise; only the boundary copy spans
        # the bbox.
        if (expands_real_dim and vector_width is not None and len(original_shape) == 1 and len(new_shape) == 1):
            K = _int_or_none(original_shape[0])
            bbox = _int_or_none(new_shape[0])
            if K is not None and bbox is not None and K >= 1 and bbox > K:
                num = bbox - K
                den = vector_width - 1
                # bbox == (W-1)*S + K  ⇔  (bbox-K) divisible by (W-1),
                # quotient S >= 1 (S >= K is the gather/scatter-with-gaps
                # case; S < K never happens for a real strided bbox).
                if den > 0 and num % den == 0 and (num // den) >= 1:
                    expands_real_dim = False

    if expands_real_dim:
        raise ValueError(f"fix_nsdfg_connector_array_shapes_mismatch ({direction}): connector "
                         f"{connector_name!r} has original shape {original_shape}; none of the four "
                         f"expected interpretations match and the candidate rebuild "
                         f"({new_shape}) would EXPAND a non-placeholder dim. Inner SDFG accesses "
                         f"can't legitimately address the larger range. Expected shapes considered:\n"
                         f"    Full:              {expected_full}\n"
                         f"    Strided:           {expected_strided}\n"
                         f"    Collapsed full:    {new_shape}\n"
                         f"    Collapsed strided: {expected_collapsed_strided}\n"
                         f"Fix the caller's connector shape or memlet subset to be consistent.")


def fix_nsdfg_connector_array_shapes_mismatch(parent_state: dace.SDFGState,
                                              nsdfg_node: dace.nodes.NestedSDFG,
                                              vector_width: Optional[int] = None) -> None:
    """Detect and fix shape mismatches in nested SDFG connector arrays.

    For each connector whose array shape matches none of the expected
    interpretations, the array is recreated at the collapsed-full shape (after
    the expansion guard accepts the rebuild) and inner accesses are updated via
    ``drop_dims``. See also ``check_nsdfg_connector_array_shapes_match``.

    :param parent_state: State in the parent SDFG containing the NSDFG node.
    :param nsdfg_node: NSDFG node whose connector shapes to fix.
    :param vector_width: When set, lets the expansion guard recognise a
        legitimate vector-widening rebuild (a halo dim grown by exactly
        ``vector_width - 1``) instead of raising.
    :raises ValueError: if a rebuild would corrupt the connector shape
        (via the expansion guard).
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

        expected_shape_strided = tuple([(dace.symbolic.int_floor(end + 1 - begin, step)).simplify()
                                        for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([(dace.symbolic.int_floor(end + 1 - begin, step)).simplify()
                                                  for begin, end, step in subset
                                                  if (dace.symbolic.int_floor(end + 1 - begin, step)).simplify() != 1])

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
        _raise_on_expansion_rebuild_mismatch(connector_name,
                                             original_shape,
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

        expected_shape_strided = tuple([(dace.symbolic.int_floor(end + 1 - begin, step)).simplify()
                                        for begin, end, step in subset])

        expected_shape_collapsed_full = tuple([((end + 1 - begin).simplify()) for begin, end, step in subset
                                               if ((end + 1 - begin).simplify()) != 1])

        expected_shape_collapsed_strided = tuple([(dace.symbolic.int_floor(end + 1 - begin, step)).simplify()
                                                  for begin, end, step in subset
                                                  if (dace.symbolic.int_floor(end + 1 - begin, step)).simplify() != 1])

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
        _raise_on_expansion_rebuild_mismatch(connector_name,
                                             original_shape,
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
    """Allocate the outer vector array and reshape the inner array to it.

    For multi-dimensional arrays the NSDFG length-1 dims are collapsed and the
    surviving dim is offset to the outer subset's start.

    :param state: The SDFG state.
    :param inner_sdfg: The inner SDFG containing the array.
    :param inner_arr_name: Name of the array to vectorize.
    :param orig_dataname: Original data array name.
    :param orig_arr: Original outer array descriptor.
    :param subset: Outer memlet subset, used to drive the dim collapse.
    :param vector_width: Width of the vector.
    :param vector_storage: Storage type for the vector.
    :param reuse_name_if_existing: Reuse ``use_name`` instead of finding a new
        name (requires ``use_name``).
    :param use_name: Explicit vector array name to use.
    :returns: ``(vector_dataname, inner_offset)``. ``inner_offset`` is always
        ``0`` (the multi-dim path rewrites memlets in place); it is kept for
        backwards-compatibility with ``compute_edge_subset``.
    :raises NotImplementedError: if the multi-dim subset does not have exactly
        one non-length-1 dimension.
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
    """Compute the boundary copy subset based on stride and offset.

    :param edge_subset: Subset from the edge.
    :param subset: Subset from the memlet.
    :param orig_arr: Original array descriptor.
    :param inner_offset: Offset value.
    :param vector_width: Width of the vector.
    :returns: The copy subset.
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


def _setup_strided_inside_nsdfg(state: dace.SDFGState,
                                nsdfg_node: dace.nodes.NestedSDFG,
                                inner_sdfg: dace.SDFG,
                                edge,
                                inner_conn: str,
                                orig_data: str,
                                orig_arr,
                                vector_width: int,
                                stride: int,
                                *,
                                direction: str,
                                multi_dim_param_dims: tuple = ()) -> None:
    """Wire a strided boundary edge so the strided load / store runs inside the NSDFG.

    Direction ``"in"``: the outer edge passes the full bounding box to the
    connector; a prep state runs ``strided_load<T>`` into a W-wide transient
    and body memlets are rewritten to it. Direction ``"out"`` is symmetric
    with a ``strided_store<T>`` finish state.

    :param state: Parent SDFG state.
    :param nsdfg_node: The NSDFG node.
    :param inner_sdfg: Inner SDFG of the NSDFG.
    :param edge: The boundary edge to rewire.
    :param inner_conn: Connector name inside the NSDFG.
    :param orig_data: Outer data name.
    :param orig_arr: Outer array descriptor.
    :param vector_width: Lane count.
    :param stride: Inter-lane stride (linearised when ``multi_dim_param_dims``).
    :param direction: ``"in"`` or ``"out"``.
    :param multi_dim_param_dims: When non-empty (diagonal / linear-combo
        case), the bbox is expanded across all listed dims instead of only
        the stride-1 dim.
    """
    assert direction in ("in", "out")
    bbox_shape = list(orig_arr.shape)
    if multi_dim_param_dims:
        for d in multi_dim_param_dims:
            bbox_shape[d] = vector_width
    else:
        # Single-stride-1-dim path (1D strided / s127 / s1111 shape).
        stride_one_indices = [i for i, s in enumerate(orig_arr.strides) if s == 1]
        assert len(stride_one_indices) == 1, (f"Strided-inside requires a single stride-1 dim; got {orig_arr.strides}")
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
                         storage=prev_arr.storage,
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
        _flatten_range = dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(vector_width - 1),
                                              dace.symbolic.SymExpr(1))])
        for inner_state in list(inner_sdfg.states()):
            if _is_strided_aux_state(inner_state):
                continue
            for inner_edge in inner_state.edges():
                if inner_edge.data is not None and inner_edge.data.data == inner_conn:
                    inner_edge.data.data = vec_name
                    if _flatten_subset and inner_edge.data.subset is not None:
                        inner_edge.data.subset = copy.deepcopy(_flatten_range)
            for node in list(inner_state.nodes()):
                if isinstance(node, dace.nodes.AccessNode) and node.data == inner_conn:
                    node.data = vec_name

        # Insert prep state at the start: strided_load(connector, vec).
        # Masked remainder: the ``_iter_mask`` fill state (the current
        # start block, prepended by ``GenerateIterationMask``) MUST run
        # before the masked load, otherwise the mask is still all-zero
        # and ``strided_load_masked`` skips every lane (an all-zero
        # result). Splice prep AFTER ``old_start`` instead of making it
        # the new start block.
        old_start = inner_sdfg.start_block
        mask_name = _iter_mask_name(inner_sdfg)
        prep = inner_sdfg.add_state(_STRIDED_LOAD_PREP_PREFIX + inner_conn, is_start_block=(mask_name is None))
        bbox_an = prep.add_access(inner_conn)
        vec_an = prep.add_access(vec_name)
        if mask_name is not None:
            code = f"strided_load_masked<{dtype_ctype}>(_in, _out, {vector_width}, {stride}, _mask);"
        else:
            code = f"strided_load<{dtype_ctype}>(_in, _out, {vector_width}, {stride});"
        tasklet = prep.add_tasklet(
            name=f"_strided_load_{inner_conn}",
            inputs={"_in"},
            outputs={"_out"},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        prep.add_edge(bbox_an, None, tasklet, "_in",
                      dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
        prep.add_edge(tasklet, "_out", vec_an, None, dace.memlet.Memlet.from_array(vec_name,
                                                                                   inner_sdfg.arrays[vec_name]))
        if mask_name is not None:
            tasklet.add_in_connector("_mask", dtype=dace.dtypes.pointer(dace.bool_), force=True)
            mask_an = prep.add_access(mask_name)
            prep.add_edge(mask_an, None, tasklet, "_mask", dace.memlet.Memlet(f"{mask_name}[0:{vector_width}]"))
            # old_start (mask fill) -> prep -> old_start's former successors.
            for se in list(inner_sdfg.out_edges(old_start)):
                inner_sdfg.add_edge(prep, se.dst, se.data)
                inner_sdfg.remove_edge(se)
            inner_sdfg.add_edge(old_start, prep, dace.InterstateEdge())
        elif old_start is not None and old_start is not prep:
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
        _flatten_range = dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(vector_width - 1),
                                              dace.symbolic.SymExpr(1))])
        for inner_state in list(inner_sdfg.states()):
            if _is_strided_aux_state(inner_state):
                continue
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
        finish = inner_sdfg.add_state(_STRIDED_STORE_FINISH_PREFIX + inner_conn)
        vec_an = finish.add_access(vec_name)
        bbox_an = finish.add_access(inner_conn)
        mask_name = _iter_mask_name(inner_sdfg)
        if mask_name is not None:
            code = f"strided_store_masked<{dtype_ctype}>(_in, _out, {vector_width}, {stride}, _mask);"
        else:
            code = f"strided_store<{dtype_ctype}>(_in, _out, {vector_width}, {stride});"
        tasklet = finish.add_tasklet(
            name=f"_strided_store_{inner_conn}",
            inputs={"_in"},
            outputs={"_out"},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        finish.add_edge(vec_an, None, tasklet, "_in",
                        dace.memlet.Memlet.from_array(vec_name, inner_sdfg.arrays[vec_name]))
        finish.add_edge(tasklet, "_out", bbox_an, None,
                        dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
        if mask_name is not None:
            tasklet.add_in_connector("_mask", dtype=dace.dtypes.pointer(dace.bool_), force=True)
            mask_an = finish.add_access(mask_name)
            finish.add_edge(mask_an, None, tasklet, "_mask", dace.memlet.Memlet(f"{mask_name}[0:{vector_width}]"))
        for sink in sink_states:
            if sink is finish:
                continue
            inner_sdfg.add_edge(sink, finish, dace.InterstateEdge())

        # Outer edge: re-attach with the bbox subset directly from the NSDFG.
        state.remove_edge(edge)
        state.add_edge(nsdfg_node, edge.src_conn, edge.dst, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))


def _setup_multi_element_strided_inside_nsdfg(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                              inner_sdfg: dace.SDFG, edge, inner_conn: str, orig_data: str, orig_arr,
                                              vector_width: int, *, elements_per_iter: int, stride: int,
                                              direction: str) -> None:
    """Wire a K-elements-per-iteration strided boundary into the NSDFG.

    Each iteration accesses ``K`` consecutive elements per lane; consecutive
    lanes are ``stride`` apart, so the bbox spans ``(W-1)*stride + K``. The
    pattern is split into K independent stride-``stride`` loads / stores into
    K W-wide phase transients.

    :param state: Parent SDFG state.
    :param nsdfg_node: The NSDFG node.
    :param inner_sdfg: Inner SDFG of the NSDFG.
    :param edge: The boundary edge to rewire.
    :param inner_conn: Connector name inside the NSDFG.
    :param orig_data: Outer data name.
    :param orig_arr: Outer array descriptor.
    :param vector_width: Lane count.
    :param elements_per_iter: K, elements accessed per iteration (>= 2).
    :param stride: Inter-lane stride (>= K).
    :param direction: ``"in"`` or ``"out"``.
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
                         storage=prev_arr.storage,
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
                             shape=(W, ),
                             dtype=orig_arr.dtype,
                             storage=dace.dtypes.StorageType.Register,
                             transient=True,
                             find_new_name=False,
                             may_alias=False)
        phase_names.append(name)

    dtype_ctype = orig_arr.dtype.ctype
    full_buf_range = dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(W - 1),
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
        # Masked remainder: splice prep AFTER the ``_iter_mask`` fill
        # state (see the single-element path) so the mask is filled
        # before the masked load runs.
        old_start = inner_sdfg.start_block
        mask_name = _iter_mask_name(inner_sdfg)
        prep = inner_sdfg.add_state(_MULTI_ELEM_LOAD_PREP_PREFIX + inner_conn, is_start_block=(mask_name is None))
        bbox_an = prep.add_access(inner_conn)
        for p in range(K):
            vec_an = prep.add_access(phase_names[p])
            if mask_name is not None:
                code = f"strided_load_masked<{dtype_ctype}>(_in + {p}, _out, {W}, {S}, _mask);"
            else:
                code = f"strided_load<{dtype_ctype}>(_in + {p}, _out, {W}, {S});"
            tasklet = prep.add_tasklet(
                name=f"_multi_elem_load_{inner_conn}_p{p}",
                inputs={"_in"},
                outputs={"_out"},
                code=code,
                language=dace.dtypes.Language.CPP,
            )
            prep.add_edge(bbox_an, None, tasklet, "_in",
                          dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
            prep.add_edge(tasklet, "_out", vec_an, None,
                          dace.memlet.Memlet.from_array(phase_names[p], inner_sdfg.arrays[phase_names[p]]))
            if mask_name is not None:
                tasklet.add_in_connector("_mask", dtype=dace.dtypes.pointer(dace.bool_), force=True)
                mask_an = prep.add_access(mask_name)
                prep.add_edge(mask_an, None, tasklet, "_mask", dace.memlet.Memlet(f"{mask_name}[0:{W}]"))
        if mask_name is not None:
            for se in list(inner_sdfg.out_edges(old_start)):
                inner_sdfg.add_edge(prep, se.dst, se.data)
                inner_sdfg.remove_edge(se)
            inner_sdfg.add_edge(old_start, prep, dace.InterstateEdge())
        elif old_start is not None and old_start is not prep:
            inner_sdfg.add_edge(prep, old_start, dace.InterstateEdge())

        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, nsdfg_node, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))
    else:
        # Finish state: K ``strided_store`` tasklets.
        sink_states = [s for s in inner_sdfg.states() if len(inner_sdfg.out_edges(s)) == 0]
        finish = inner_sdfg.add_state(_MULTI_ELEM_STORE_FINISH_PREFIX + inner_conn)
        bbox_an = finish.add_access(inner_conn)
        mask_name = _iter_mask_name(inner_sdfg)
        for p in range(K):
            vec_an = finish.add_access(phase_names[p])
            if mask_name is not None:
                code = f"strided_store_masked<{dtype_ctype}>(_in, _out + {p}, {W}, {S}, _mask);"
            else:
                code = f"strided_store<{dtype_ctype}>(_in, _out + {p}, {W}, {S});"
            tasklet = finish.add_tasklet(
                name=f"_multi_elem_store_{inner_conn}_p{p}",
                inputs={"_in"},
                outputs={"_out"},
                code=code,
                language=dace.dtypes.Language.CPP,
            )
            finish.add_edge(vec_an, None, tasklet, "_in",
                            dace.memlet.Memlet.from_array(phase_names[p], inner_sdfg.arrays[phase_names[p]]))
            finish.add_edge(tasklet, "_out", bbox_an, None,
                            dace.memlet.Memlet.from_array(inner_conn, inner_sdfg.arrays[inner_conn]))
            if mask_name is not None:
                tasklet.add_in_connector("_mask", dtype=dace.dtypes.pointer(dace.bool_), force=True)
                mask_an = finish.add_access(mask_name)
                finish.add_edge(mask_an, None, tasklet, "_mask", dace.memlet.Memlet(f"{mask_name}[0:{W}]"))
        for sink in sink_states:
            if sink is finish:
                continue
            inner_sdfg.add_edge(sink, finish, dace.InterstateEdge())

        state.remove_edge(edge)
        state.add_edge(nsdfg_node, edge.src_conn, edge.dst, edge.dst_conn,
                       dace.memlet.Memlet(data=orig_data, subset=edge.data.subset))


def _process_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                   vector_width: int, vector_storage: dace.dtypes.StorageType, *, direction: str) -> Set[str]:
    """Rewire one direction of NSDFG-boundary edges through a fresh vector access node.

    Backs ``process_in_edges`` / ``process_out_edges``. Strided edges are
    instead routed inside the NSDFG via the strided handlers; ``"out"`` reuses
    an inout connector's vector name minted by a prior ``"in"`` call.

    :param state: Parent SDFG state.
    :param nsdfg_node: The NSDFG node.
    :param movable_arrays: Set of ``(data_name, subset)`` pairs to rewire.
    :param vector_width: Lane count.
    :param vector_storage: Storage type for the vector arrays.
    :param direction: ``"in"`` or ``"out"``.
    :returns: The set of vector data names allocated.
    :raises NotImplementedError: if a strided outer subset's bbox volume does
        not match ``(W-1)*S + K`` for any valid K.
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

            # A stride-1 dim that spans more than one element is the SIMD
            # lane (a unit-stride window). When one is present, never treat
            # a *different* non-contiguous dim as the strided lane: the
            # ``wide_dims`` scan drops symbolic-bound dims (``int(ee - b + 1)``
            # raises on ``0:klon``), so a full passthrough dim with a
            # concrete size (e.g. cloudsc ``zqx0[0:klon, 0:klev, 0:5]`` —
            # the size-5 species dim) is the only survivor and would be
            # mistaken for a strided lane. The contiguous lane dim itself
            # is symbolic here, so it is detected directly from the subset.
            contiguous_lane_present = any(
                not bool(dace.symbolic.simplify(e.data.subset[i][1] - e.data.subset[i][0]) == 0)
                for i in stride_one_indices)

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
            elif (len(wide_dims) == 1 and vector_width > 1 and orig_arr.strides[wide_dims[0][0]] != 1
                  and not contiguous_lane_present):
                # Single NON-contiguous wide dim: the vectorized (innermost)
                # map param indexes a non-unit-stride array dim, so the W
                # lanes sit at memory stride ``orig_arr.strides[d]`` apart --
                # a strided gather, not a contiguous window. Continuity is
                # judged by the descriptor stride (not dim position): widen
                # the lane dim and let the strided-load path squeeze it into
                # a W-wide transient filled at that stride. (C-layout
                # ``bb[i, j]`` with ``i`` innermost, ``strides=(N, 1)`` ->
                # wide dim 0, stride N.)
                d, bw = wide_dims[0]
                if bw != vector_width:
                    raise NotImplementedError(
                        f"_process_edges (direction={direction!r}): non-contiguous vectorized dim {d} "
                        f"of {orig_data} has bbox {bw} != W={vector_width}; K-elements-per-iter on a "
                        f"strided dim is not yet supported.")
                # Guard: a strided access widens ONLY the lane dim; every
                # other dim must be a single element. A wide non-lane dim
                # (an inner loop's full range) is the mixed strided+gather
                # edge case -- refuse (-> clean skip) instead of a 2D access.
                for _d, (_b, _ee, _s) in enumerate(e.data.subset):
                    if _d == d:
                        continue
                    try:
                        _ln = int(_ee - _b + 1)
                    except (TypeError, ValueError):
                        _ln = None
                    if _ln is None or _ln > 1:
                        raise NotImplementedError(
                            f"_process_edges (direction={direction!r}): non-contiguous edge on {orig_data} "
                            f"has a wide non-lane dim {_d} (mixed strided+gather; lane dim {d}); not supported.")
                is_strided = True
                stride_value = orig_arr.strides[d]
                multi_dim_dims = (d, )
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
                        beg_syms = free_symbols(beg)
                        if not beg_syms:
                            shared_syms_by_name = set()
                            break
                        bnames = set()
                        for sym in beg_syms:
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
                    _setup_multi_element_strided_inside_nsdfg(state,
                                                              nsdfg_node,
                                                              inner_sdfg,
                                                              e,
                                                              inner_conn,
                                                              orig_data,
                                                              orig_arr,
                                                              vector_width,
                                                              elements_per_iter=multi_elem_per_iter,
                                                              stride=stride_value,
                                                              direction=direction)
                else:
                    _setup_strided_inside_nsdfg(state,
                                                nsdfg_node,
                                                inner_sdfg,
                                                e,
                                                inner_conn,
                                                orig_data,
                                                orig_arr,
                                                vector_width,
                                                stride_value,
                                                direction=direction,
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
            # Masked remainder: ``_iter_mask`` lives inside the NSDFG, not
            # at this outer boundary, so the real<->buffer staging copy is
            # gated by the real array's own extent. The main loop / scalar
            # remainder keep the raw W-wide memlet (codegen ``CopyND<W>``,
            # the fast path) unchanged.
            masked = _iter_mask_name(inner_sdfg) is not None
            orig_memlet = dace.memlet.Memlet(data=orig_data, subset=copy_subset)
            vec_memlet = dace.memlet.Memlet.from_array(vector_dataname, vec_arr_desc)
            if direction == "in":
                # src --[orig, copy_subset]--> an --[vec_from_array]--> nsdfg
                if masked:
                    emit_staging_copy(state, e.src, e.src_conn, an, None, orig_memlet, vector_dataname, vector_width,
                                      "in", gate_extent=True)
                else:
                    state.add_edge(e.src, e.src_conn, an, None, orig_memlet)
                state.add_edge(an, None, e.dst, e.dst_conn, vec_memlet)
            else:
                # nsdfg --[vec_from_array]--> an --[orig, copy_subset]--> dst
                assert e.src == nsdfg_node
                assert e.src_conn is not None
                assert len(set(state.out_edges_by_connector(nsdfg_node, e.src_conn))) == 0
                state.add_edge(e.src, e.src_conn, an, None, vec_memlet)
                if masked:
                    emit_staging_copy(state, an, None, e.dst, e.dst_conn, orig_memlet, vector_dataname, vector_width,
                                      "out", gate_extent=True)
                else:
                    state.add_edge(an, None, e.dst, e.dst_conn, orig_memlet)

    return vectorized_datanames


def process_in_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                     vector_width: int, vector_storage: dace.dtypes.StorageType) -> Set[str]:
    """Rewire ``src -> nsdfg`` boundary edges through a fresh vector access.

    :param state: Parent SDFG state.
    :param nsdfg_node: The NSDFG node.
    :param movable_arrays: Set of ``(data_name, subset)`` pairs to rewire.
    :param vector_width: Lane count.
    :param vector_storage: Storage type for the vector arrays.
    :returns: The set of vector data names allocated.
    """
    return _process_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage, direction="in")


def process_out_edges(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG, movable_arrays: Set[str],
                      vector_width: int, vector_storage: dace.dtypes.StorageType) -> Set[str]:
    """Rewire ``nsdfg -> dst`` boundary edges through a fresh vector access.

    :param state: Parent SDFG state.
    :param nsdfg_node: The NSDFG node.
    :param movable_arrays: Set of ``(data_name, subset)`` pairs to rewire.
    :param vector_width: Lane count.
    :param vector_storage: Storage type for the vector arrays.
    :returns: The set of vector data names allocated.
    """
    return _process_edges(state, nsdfg_node, movable_arrays, vector_width, vector_storage, direction="out")


def add_copies_before_and_after_nsdfg(
    state: SDFGState,
    nsdfg_node: dace.nodes.NestedSDFG,
    vector_width: int,
    vector_storage: dace.dtypes.StorageType,
    skip: Set[str],
    fuse_overlapping_loads: bool = False,
) -> Set[str]:
    """Add vector copy operations before and after a nested SDFG node.

    Movable arrays (single uniform, vectorizable access pattern) get a vector
    copy at the NSDFG boundary; unmovable arrays (multiple subsets or unbound
    symbols) get copy-in / copy-out tasklets inside the NSDFG instead.

    :param state: The SDFG state containing the nested SDFG node.
    :param nsdfg_node: The nested SDFG node to process.
    :param vector_width: The width of vector operations.
    :param vector_storage: Storage type for vector arrays.
    :param skip: Data names never copied (used for unstructured loads).
    :param fuse_overlapping_loads: when True, an array read at multiple
        overlapping subsets is fused into one shared union-window staging
        buffer (consumers become offset views into it) instead of one
        independent copy per subset (jacobi2d-style stencils). Replaces
        the former standalone post-vectorizer ``FuseOverlappingLoads``
        pass; baked here so it composes with the staging classification.
    :returns: The set of vector data names inserted at the boundary.
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
            # Single access pattern. R4/R5 copy-inside rule: it may stay
            # *outside* (the movable ``vector_copy<T, W>`` path) only when
            # (a) every free symbol is available outside the NSDFG AND
            # (b) it is a trivial exact-VLEN, step-1, contiguous window
            # (see ``_is_trivial_exact_vlen_copy``). A strided / partial /
            # masked-remainder single-subset access fails (b) and is
            # staged *inside* instead, where ``emit_staging_copy`` gates
            # each lane — the outside path has no such gate and would
            # silently corrupt those accesses.
            memlet = next(iter(memlets))
            memlet_syms = {str(s) for s in memlet.free_symbols}
            avaialble_syms = {str(s) for s in state.symbols_defined_at(nsdfg_node)}
            masked = _iter_mask_name(inner_sdfg) is not None

            syms_available = all({s in avaialble_syms for s in memlet_syms})
            if syms_available and _is_trivial_exact_vlen_copy(memlet, inner_sdfg.arrays[dataname], vector_width,
                                                              masked):
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

    # Generate name mappings.
    #
    # Default (knob off / single subset): one independent ``(W,)`` staging
    # buffer per subset (``A_vec_0``, ``A_vec_1``, ...), each filled by its
    # own ``vector_copy<T, W>``. For a jacobi2d-style stencil this emits 5
    # redundant copies of overlapping windows of the same inner array.
    #
    # Fused (``fuse_overlapping_loads`` and >= 2 subsets): one shared
    # N-D buffer ``A_vec`` shaped to the bounding-box union of all subsets,
    # staged once; the N consumers become offset-slice views into it
    # (``fused_targets``). This bakes in the former standalone
    # ``FuseOverlappingLoads`` pass at the staging boundary so it composes
    # with the movable / unmovable classification (knob default off keeps
    # the legacy per-subset behaviour byte-identical).
    subset_to_name_map = dict()
    fused_targets = dict()
    fused_arrays = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
        # Insert copy-ins
        desc = inner_sdfg.arrays[unmovable_arr_name]
        if fuse_overlapping_loads and len(subsets) >= 2:
            union_subset, union_shape = _compute_subset_union(subsets)
            vec_arr_name = VecNameScheme.make(unmovable_arr_name)
            if vec_arr_name not in inner_sdfg.arrays:
                inner_sdfg.add_array(
                    name=vec_arr_name,
                    shape=tuple(union_shape),
                    dtype=desc.dtype,
                    location=desc.location,
                    transient=True,
                    find_new_name=False,
                )
            fused_arrays[unmovable_arr_name] = (vec_arr_name, union_subset)
            union_begin = dace.subsets.Range([(b, b, 1) for (b, _, _) in union_subset])
            for subset in subsets:
                offsetted = copy.deepcopy(subset).offset_new(union_begin, True)
                subset_to_name_map[(unmovable_arr_name, subset)] = vec_arr_name
                fused_targets[(unmovable_arr_name, subset)] = offsetted
        else:
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
                fused_subset = fused_targets.get((edge.data.data, edge.data.subset))
                if fused_subset is not None:
                    # Fused: offset-slice view into the one shared buffer.
                    edge.data = dace.memlet.Memlet(data=vec_name, subset=copy.deepcopy(fused_subset))
                else:
                    # Per-subset: the whole ``(W,)`` independent buffer.
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

        :param target_state: State to splice into.
        :param unmovable_name: Original (unmovable) data name.
        :param vec_name: Vector buffer name.
        :param subset: Subset of the original access.
        :param direction: ``"in"`` (orig -> tasklet -> vec) or ``"out"``
            (vec -> tasklet -> orig).
        """
        assert direction in ("in", "out"), direction
        orig_access = target_state.add_access(unmovable_name)
        orig_access.setzero = True
        v_access = target_state.add_access(vec_name)
        v_access.setzero = True
        orig_memlet = dace.memlet.Memlet(data=unmovable_name, subset=copy.deepcopy(subset))
        # The copy lives inside the body NSDFG, so the masked-remainder
        # ``_iter_mask`` (if any) is in scope and gates each lane.
        mask_name = _iter_mask_name(target_state.sdfg)
        if direction == "in":
            emit_staging_copy(target_state, orig_access, None, v_access, None, orig_memlet, vec_name, vector_width,
                              "in", mask_name=mask_name)
        else:
            emit_staging_copy(target_state, v_access, None, orig_access, None, orig_memlet, vec_name, vector_width,
                              "out", mask_name=mask_name)

    def _emit_fused_union_copy(target_state, unmovable_name, vec_name, union_subset, direction):
        """Splice ONE shared-buffer staging copy for a fused unmovable array.

        Unlike :func:`_emit_unmovable_copy` (one ``vector_copy<T, W>`` per
        subset), this copies the whole bounding-box union window once into
        the single N-D shared buffer. The buffer is shaped to the union
        extent (not ``W``), so a plain AccessNode -> AccessNode memlet copy
        is emitted (lowered by DaCe codegen); the N consumers are already
        rewritten as offset-slice views into this one buffer (via
        ``fused_targets``). Mirrors the proven union-copy edge of the
        standalone ``FuseOverlappingLoads`` pass.

        :param target_state: State to splice into (copy-in / copy-out).
        :param unmovable_name: Original (unmovable) data name.
        :param vec_name: Shared union-window buffer name.
        :param union_subset: Bounding-box subset on the original array.
        :param direction: ``"in"`` (orig -> buffer) or ``"out"``
            (buffer -> orig).
        """
        assert direction in ("in", "out"), direction
        orig_access = target_state.add_access(unmovable_name)
        orig_access.setzero = True
        v_access = target_state.add_access(vec_name)
        v_access.setzero = True
        union_memlet = dace.memlet.Memlet(data=unmovable_name, subset=copy.deepcopy(union_subset))
        if direction == "in":
            target_state.add_edge(orig_access, None, v_access, None, union_memlet)
        else:
            target_state.add_edge(v_access, None, orig_access, None, union_memlet)

    # Insert copy-ins and outs
    name_to_subset_map = dict()
    for unmovable_arr_name, subsets in unmovable_arrays.items():
        # If a packed stored, then continue
        # Add a unique vector array for each unique subset
        desc = inner_sdfg.arrays[unmovable_arr_name]

        if unmovable_arr_name in fused_arrays:
            # Fused: one union-window copy in / out instead of one per
            # subset. The shared buffer is N-D (union extent, != W), so the
            # masked-remainder per-lane ``_iter_mask`` gate of
            # ``emit_staging_copy`` does not apply; fail loud rather than
            # silently OOB / un-gated when both are requested together.
            vec_arr_name, union_subset = fused_arrays[unmovable_arr_name]
            if unmovable_arr_name in read_set:
                copy_in_state = find_copy_in_state(inner_sdfg, nsdfg_node,
                                                   {str(s)
                                                    for s in union_subset.free_symbols}, unmovable_arr_name)
                if _iter_mask_name(copy_in_state.sdfg) is not None:
                    raise NotImplementedError(
                        "fuse_overlapping_loads with a masked remainder is not supported yet: the shared "
                        f"union-window staging copy for {unmovable_arr_name!r} is not iter_mask-gated. Use "
                        "fuse_overlapping_loads=False or a non-masked remainder strategy.")
                _emit_fused_union_copy(copy_in_state, unmovable_arr_name, vec_arr_name, union_subset, "in")
            if unmovable_arr_name in write_set:
                if _iter_mask_name(copy_out_state.sdfg) is not None:
                    raise NotImplementedError(
                        "fuse_overlapping_loads with a masked remainder is not supported yet: the shared "
                        f"union-window staging copy for {unmovable_arr_name!r} is not iter_mask-gated. Use "
                        "fuse_overlapping_loads=False or a non-masked remainder strategy.")
                _emit_fused_union_copy(copy_out_state, unmovable_arr_name, vec_arr_name, union_subset, "out")
            continue

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
                assert len(set(state.in_edges_by_connector(
                    nsdfg_node, vec_data))) == 0, (f"There are edges connected to {vec_data}: "
                                                   f"{set(state.in_edges_by_connector(nsdfg_node, vec_data))}")
                ie.dst_conn = vec_data
        for oe in state.out_edges(nsdfg_node):
            if oe.src_conn is not None and oe.src_conn == movable_data and vec_data in nsdfg_node.out_connectors:
                assert vec_data in nsdfg_node.out_connectors, f"{vec_data} not in {nsdfg_node.out_connectors}"
                assert len(set(state.out_edges_by_connector(
                    nsdfg_node, vec_data))) == 0, (f"There are edges connected to {vec_data}: "
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
    """Find / create the first NSDFG state where every symbol in ``free_syms`` is in scope.

    Walks the (single-successor) NSDFG forward from the start block,
    accumulating interstate-edge assignments, and inserts a new copy-in
    state before the first state that defines all required symbols. A state
    created by a prior call is reused.

    :param inner_sdfg: The inner SDFG to walk.
    :param nsdfg_node: The NSDFG node (source of the initial symbol mapping).
    :param free_syms: Symbol names that must be in scope.
    :param name: Array name being copied (used in the state label).
    :returns: The copy-in state.
    :raises RuntimeError: if no state defines every required symbol.
    """
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
    """Clear connector and tasklet dtypes on the NSDFG and recurse into nested ones.

    :param inner_sdfg: Inner SDFG whose tasklet / nested connectors are reset.
    :param nsdfg: NSDFG node whose in / out connectors are reset.
    """
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
    """Move an access node from below a one-param vector map to above it.

    Rewrites ``MapEntry -> AccessNode -> dst`` into
    ``AccessNode -> MapEntry -> dst`` so overlapping accesses merge more
    easily.

    :param state: The SDFG state to modify.
    :param node: The access node to lift.
    :param map_entry: The single-parameter vector map (length 1) it sits under.
    """
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
    map_lengths = tuple([dace.symbolic.int_floor(e + 1 - b, s) for (b, e, s) in map_entry.map.range])
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
