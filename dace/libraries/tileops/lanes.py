# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""C++ emission shared by the ``pure`` expansions of the tile nodes.

A tile is a K-fold nested loop over per-dim lane indices ``__l0, __l1, ...`` rather than a flattened index with a
decode. Register tiles are contiguous and row-major, so :func:`tile_offset` flattens the lane indices; a node supplies
the per-lane body to :func:`nested_loops`.
"""
import numbers
from collections.abc import Sequence

import sympy

import dace
from dace.symbolic import has_one_marker

# Legal ``_idx_<d>`` gather/scatter index dtypes (design section 10.4). Unsigned widths are
# accepted because CSR/COO index arrays are commonly ``uint32``; ``gather_lane_offset`` casts
# the read to ``long long`` so an unsigned index cannot wrap the signed address sum.
GATHER_INDEX_DTYPES = (dace.int32, dace.int64, dace.uint32, dace.uint64)


def constant_trip_count(width: int | sympy.Basic) -> bool:
    """True iff ``width`` is a compile-time-constant integer loop bound.

    A per-lane tile loop with a constant trip count (the register-tile /
    vector width, e.g. ``2`` for an ``fp16x2`` fill) is safe to force-unroll:
    the bound is known at code-gen time and never a runtime / symbolic value.
    This guard keeps the ``#pragma unroll`` (see :func:`nested_loops`) off any
    hypothetical symbolic-width loop, where an unroll pragma on a runtime bound
    is meaningless. Tile widths are ``ListProperty(element_type=int)`` today, so
    the common path is the plain ``int`` check; the sympy branch is defensive.

    :param width: A per-tile-dim width (``int`` in practice; a sympy expression
        is tolerated and accepted only when it is a concrete integer).
    :returns: ``True`` for a compile-time-constant integer width, else ``False``.
    """
    if isinstance(width, numbers.Integral):
        return True
    return isinstance(width, sympy.Basic) and bool(width.is_Integer)


def half_disambiguated(ref: str, own_ctype: str, meets_ctype: str) -> str:
    """Route a bare ``dace::float16`` (CUDA's ``__half``) operand through one
    explicit ``(float)`` hop before it meets a differently-typed value.

    ``dace/runtime/include/dace/types.h`` aliases the GPU ``dace::float16``
    straight to CUDA's ``__half`` (unlike the CPU-side ``half`` struct, which
    deliberately exposes exactly ONE implicit conversion, ``operator
    float()``, for the documented reason "declaring member binary overloads
    would tie with the built-ins ... and be ambiguous"). ``__half`` itself
    declares SEVERAL simultaneously non-``explicit`` conversions (to
    ``float``, ``short``, ``unsigned short``, ``int``, ``unsigned int``,
    ``long long``, ``unsigned long long``, ``bool`` -- all gated by the one
    ``__CUDA_NO_HALF_CONVERSIONS__`` macro, never individually). A bare half
    value handed directly to a mixed-type infix operator or to an overloaded
    function with no half overload (``std::sqrt``, ``std::fma``, ...) is
    ambiguous ("more than one conversion function ... applies" / "more than
    one operator ... matches" / "more than one instance of overloaded
    function ... matches" -- the same defect, worded differently per call
    shape). ``dace/runtime/include/dace/tile_ops/cuda.h`` already dodges
    this at the ISA/runtime layer (``_cuda_to_compute`` casts every operand
    to ``float`` before doing any arithmetic); this mirrors that at the
    per-kernel Python-codegen layer so the ``pure`` expansion agrees with
    the ISA one instead of handing the C++ compiler an unresolvable half.

    Every fp16 value is exactly representable in ``float32``, so the hop
    picks the one lossless conversion and changes no computed value -- the
    alternative was a compile error, not a different result.

    :param ref: The C++ expression for the operand.
    :param own_ctype: The operand's own element C++ type.
    :param meets_ctype: The C++ type of whatever this operand is about to
        meet (the other operand, or the function it is passed to). No hop is
        inserted when this already equals ``own_ctype`` -- the value never
        leaves ``dace::float16``, so native half arithmetic keeps its own
        precision.
    :returns: ``ref`` unchanged, or ``(float)(ref)`` when disambiguation is
        needed.
    """
    if own_ctype == dace.float16.ctype and meets_ctype != dace.float16.ctype:
        return f"(float)({ref})"
    return ref


def nested_loops(widths: Sequence[int], body: str, indent: str = "    ") -> str:
    """Wrap ``body`` in a K-fold nested for-loop iterating per-dim
    lane indices ``__l0, __l1, ...``.

    Each fixed-width lane loop is preceded by ``#pragma unroll`` (guarded by
    :func:`constant_trip_count`): the trip count is the compile-time
    register-tile width -- the vector width -- so a full unroll strips the loop
    overhead of a broadcast / scalar-fill (e.g. the ``fp16`` constant-multiply
    ``_c[__l] = float16(0.125)``) and lets the backend keep the tile in
    registers. This mirrors the CPU map-unroll pragma (``cpu.py`` ``#pragma
    unroll``) and the per-lane ``#pragma unroll`` the CUDA tile-op header emits;
    the same pure tasklet body is emitted verbatim by both the CPU and CUDA
    targets, so NVCC / Clang honour the pragma and GCC harmlessly ignores the
    unknown pragma.

    :param widths: Per-tile-dim widths, innermost-last.
    :param body: The per-lane C++ body (already trailing-``;`` if
        needed); may contain newlines (each line is indented).
    :param indent: One indent level (default 4 spaces).
    :returns: A C++ snippet with the nested loops + indented body.
    """
    K = len(widths)
    lines = []
    for d, w in enumerate(widths):
        if constant_trip_count(w):
            lines.append(f"{indent * d}#pragma unroll")
        lines.append(f"{indent * d}for (std::size_t __l{d} = 0; __l{d} < {w}; ++__l{d}) {{")
    for line in body.splitlines():
        lines.append(f"{indent * K}{line}")
    for d in reversed(range(K)):
        lines.append(f"{indent * d}}}")
    return "\n".join(lines)


def tile_offset(widths: Sequence[int]) -> str:
    """Return the row-major flat offset expression for a register tile.

    For ``widths = (W_0, W_1, W_2)`` returns
    ``__l0 * (W_1*W_2) + __l1 * W_2 + __l2``. Always row-major because
    register-storage tile transients are contiguous by construction.

    :param widths: Per-tile-dim widths, innermost-last.
    :returns: The C++ offset expression.
    """
    K = len(widths)
    if K == 0:
        return "0"
    stride = 1
    parts = []
    for d in reversed(range(K)):
        if stride == 1:
            parts.append(f"__l{d}")
        else:
            parts.append(f"(__l{d} * {stride})")
        stride *= widths[d]
    return " + ".join(reversed(parts))


def lane_invariant_assign(out_conn: str, rhs_expr: str, out_dtype: str, widths: Sequence[int],
                          mask_elements: int | sympy.Basic | None) -> str:
    """Body assigning a lane-invariant ``rhs_expr`` to the one-element output ``out_conn``.

    Every lane computes the same value, so under ``_mask`` it is kept when AT LEAST ONE lane is
    active and takes the masked fill ``out_dtype(0)`` otherwise, like an inactive lane of a tile
    op. ``_mask`` is a pointer read per lane; only a one-element mask is bound by value.

    :param out_conn: The by-value output connector.
    :param rhs_expr: The C++ value.
    :param out_dtype: The output element C++ type.
    :param widths: Per-tile-dim widths of ``_mask``.
    :param mask_elements: Element count of the ``_mask`` memlet; ``None`` when unmasked.
    :returns: The C++ body.
    """
    if mask_elements is None:
        return f"{out_conn} = {rhs_expr};"
    if mask_elements == 1:
        return f"{out_conn} = _mask ? ({rhs_expr}) : {out_dtype}(0);"
    any_lane = nested_loops(widths, f"__any_lane = __any_lane || _mask[{tile_offset(widths)}];")
    return f"bool __any_lane = false;\n{any_lane}\n{out_conn} = __any_lane ? ({rhs_expr}) : {out_dtype}(0);"


def offset_via_strides(
    coeffs: Sequence[int],
    strides: Sequence[str],
    replicate_factors: Sequence[int] = (),
    lane_index_exprs: Sequence[str] = ()) -> str:
    """Return the flat offset expression
    ``sum_d coeffs[d] * strides[d] * (__l<d> / replicate_factors[d])``.

    Used by ``TileLoad`` / ``TileStore`` to address the source / dest
    array's flat memory through its own per-dim strides scaled by the
    optional per-tile-dim ``dim_strides`` coefficient. When
    ``replicate_factors[d] > 1``, the per-dim lane index is divided by
    the replicate factor so ``k`` consecutive lanes index the same
    source element -- the within-dim group-broadcast lowering for the
    ``int_floor`` / ``int_ceil`` regime.

    Per-lane override: when ``lane_index_exprs[d]`` is a non-empty string,
    dim ``d`` uses it verbatim as the per-lane element offset *relative to
    the connector base* (the dim contributes ``(lane_index_exprs[d]) *
    strides[d]`` and the dim's ``coeffs`` / ``replicate_factors`` are
    bypassed). The ``TileLoad`` pure expansion supplies it for a
    non-dividing ``int_floor(c*iter + c0, D)`` (``W % D != 0`` or symbolic
    ``D``): the contracted-box broadcast ``_src[__l/D]`` is correct only
    when every tile starts on a phase boundary (``W % D == 0``), so the
    expansion instead passes the phase-aware ``(c*iter + c0 + c*__l)/D -
    (c*iter + c0)/D`` (relative to the box base ``&src[(c*iter+c0)/D]``).

    :param coeffs: Per-tile-dim integer coefficient (``1`` for
        contiguous; >1 for strided access).
    :param strides: Per-tile-dim source-array stride as a C++
        expression (typically the symbolic stride rendered with
        :func:`dace.symbolic.symstr`).
    :param replicate_factors: Per-tile-dim replicate factor (``1`` =
        each lane reads a distinct element; ``k > 1`` = ``k`` lanes
        share each element). Defaults to all-1 (no replication) when
        empty or omitted.
    :param lane_index_exprs: Per-tile-dim per-lane element offset C++
        expression (relative to the connector base); an empty / missing
        entry uses the standard ``coeff * stride * (__l / replicate)``
        addressing. Defaults to all-empty.
    :returns: The C++ offset expression, or ``"0"`` if K==0.
    """
    if not coeffs:
        return "0"
    parts = []
    for d, (c, s) in enumerate(zip(coeffs, strides)):
        if d < len(lane_index_exprs) and lane_index_exprs[d]:
            parts.append(f"(({lane_index_exprs[d]}) * ({s}))")
            continue
        lane = f"__l{d}"
        if d < len(replicate_factors):
            r = replicate_factors[d]
            # Symbolic replicate factors (e.g. ``DV`` in ``c[i // DV]``)
            # can't be compared via ``> 1`` (sympy raises TypeError).
            # Coerce to int when possible; symbolic falls through to the
            # runtime divisor emission -- ``__l / DV`` evaluates safely
            # at any DV >= 1.
            try:
                emit_div = int(r) > 1
            except (TypeError, ValueError):
                emit_div = True
            if emit_div:
                lane = f"({lane} / {r})"
        parts.append(f"({c} * ({s}) * {lane})")
    return " + ".join(parts)


def resolve_gather_deps(idx_shape: Sequence[int | sympy.Basic], widths: Sequence[int]) -> tuple[int, ...] | None:
    """Find the sorted subset of tile dims an ``_idx_<d>`` index tile depends on.

    Implements the design section 9.2 lane-dependency lookup: given an
    ``_idx_<d>`` connector's descriptor shape and the lib node's tile widths,
    return the sorted tuple of tile dim indices ``deps_d`` the index tile
    varies over, or ``None`` if the shape cannot be reconciled with ``widths``.
    The special ``(1,)`` shape (scalar gather index, no lane dep) returns the
    empty tuple ``()``.

    **Index tiles are full-K-dim and resolve POSITIONALLY — ``ONE`` is NEVER
    collapsed** (user direction 2026-06-14: the markers must be preserved). A
    ``K``-dim index tile encodes its per-tile-dim dependencies *by position*:
    tile dim ``d`` is a dependency iff ``idx_shape[d]`` is neither literal ``1``
    nor the :data:`~dace.symbolic.ONE` broadcast marker (and a non-marker extent
    must equal ``widths[d]``). ``(W, ONE)`` (col gather, dep dim 0) and
    ``(ONE, W)`` (row gather, dep dim 1) are DISTINCT — collapsing both to
    ``(W,)`` would make equal-width tiles (``widths=(8, 8)``) ambiguous, the
    exact bug the ``ONE`` marker exists to prevent. The index-tile emitters
    (:meth:`InsertTileLoadStore._stage_array_read_tile`) therefore always emit
    the full-rank ``ONE``-padded form; a shape whose rank is not ``K`` (other
    than the scalar ``(1,)``) is rejected.

    :param idx_shape: The descriptor shape of an ``_idx_<d>`` connector
        (e.g. ``(4, 8)``, ``(4, ONE)``, ``(ONE, 8)``).
    :param widths: The lib node's full tile widths ``(W_0, ..., W_{K-1})``.
    :returns: Sorted tuple of tile dim indices, ``()`` for the scalar case,
        or ``None`` when the shape cannot be reconciled with ``widths``.
    """

    def extents_equal(a: int | sympy.Basic, b: int | sympy.Basic) -> bool:
        """Symbolic-safe extent equality."""
        try:
            return bool(dace.symbolic.simplify(a - b) == 0)
        except Exception:  # noqa: BLE001
            return a == b

    idx_shape = tuple(idx_shape)
    K = len(widths)
    # Scalar gather index (no lane dep): the legacy literal ``(1,)`` shape. A
    # K-dim all-``ONE`` shape is also scalar and falls out of the positional
    # loop below (every dim skipped -> empty deps).
    if idx_shape == (1, ):
        return ()
    # Full-K-dim positional resolution. The ONE markers are PRESERVED, never
    # collapsed: dim d is a dep iff its extent is not a 1/ONE broadcast marker.
    if len(idx_shape) != K:
        return None
    deps = []
    for d in range(K):
        if has_one_marker(idx_shape[d]):
            continue  # broadcast dim -- not a dependency
        if not extents_equal(idx_shape[d], widths[d]):
            return None  # non-marker extent disagrees with the tile width
        deps.append(d)
    return tuple(deps)


def gather_lane_offset(deps: Sequence[int], widths: Sequence[int], conn: str) -> str:
    """Build the row-major flat lane offset C expression into an ``_idx_<d>`` tile.

    Given ``deps = (p_0, ..., p_{n-1})`` (the tile dims the gather expression
    depends on) and the lib node's widths, returns the CPP expression
    ``conn[<flat offset>]`` where the flat offset is
    ``__l<p_0> * (W_<p_1> * W_<p_2> * ...) + __l<p_1> * (W_<p_2> * ...) + ... + __l<p_{n-1}>``.

    For the scalar case (``deps == ()``) returns ``conn[0]``.

    The read is cast to ``long long``: callers sum it with affine terms that may
    be negative, and an unsigned index dtype would make the whole sum wrap.

    :param deps: Sorted tuple of tile dim indices from :func:`resolve_gather_deps`.
    :param widths: Lib node's full tile widths.
    :param conn: The connector name (e.g. ``"_idx_0"``).
    :returns: A CPP expression string of the form ``(long long)conn[<offset>]``.
    """
    if not deps:
        return f"(long long)({conn}[0])"
    parts = []
    for i, p in enumerate(deps):
        inner = 1
        for q in deps[i + 1:]:
            inner *= widths[q]
        parts.append(f"__l{p}" if inner == 1 else f"(__l{p} * {inner})")
    return f"(long long)({conn}[{' + '.join(parts)}])"
