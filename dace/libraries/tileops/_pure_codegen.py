# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared CPP-codegen helpers for the tile-op lib nodes' ``pure`` expansions.

These helpers keep the bodies short and idiomatic: a K-fold nested
``for``-loop with per-dim lane indices (``__l0, __l1, ...``) rather
than the flattened linear-index-plus-decode shape. Each lib node's
pure expansion plugs in its own per-lane body via :func:`nested_loops`
and uses :func:`tile_offset` to flatten the tile transient's index
(register tiles are always row-major-contiguous).
"""
import numbers
from typing import Sequence

import sympy


def constant_trip_count(width) -> bool:
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
    import sympy
    return isinstance(width, sympy.Basic) and bool(width.is_Integer)


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


def resolve_gather_deps(idx_shape, widths):
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
    import sympy
    import dace
    from dace.symbolic import ONE

    def _is_one(s):
        """True only for the :data:`ONE` broadcast marker -- NOT a literal ``1``.

        ``ONE`` is the deliberate broadcast / collapsed-dim marker; a literal
        ``1`` extent means a genuine width-1 tile dim (a real dependency). The
        two must stay distinct here -- that disambiguation (broadcast vs a
        coincidental width-1 dep) is the whole reason ``ONE`` is a symbol and
        not just ``1`` (user 2026-06-14), and it is what keeps the index-tile
        rank aligned with the data tile (cuTile-faithful broadcast dims)."""
        return isinstance(s, sympy.Basic) and ONE in s.free_symbols

    def _extent_eq(a, b):
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
        if _is_one(idx_shape[d]):
            continue  # broadcast dim -- not a dependency
        if not _extent_eq(idx_shape[d], widths[d]):
            return None  # non-marker extent disagrees with the tile width
        deps.append(d)
    return tuple(deps)


def _strides_match_packed(shape, strides, order):
    """True when ``strides`` is the packed contiguous form for ``shape`` in
    ``order`` ("C" -- innermost-last, stride 1 on the last dim; or "F" --
    innermost-first, stride 1 on the first dim) with NO padding between dims.

    Symbolic shapes / strides are compared via sympy ``simplify == 0``.

    :param shape: Tuple of dim sizes (may be symbolic).
    :param strides: Tuple of per-dim strides (may be symbolic).
    :param order: "C" or "F".
    :returns: ``True`` iff the layout is exactly packed in the requested order.
    """
    import dace
    if len(shape) != len(strides):
        return False
    if order == "C":
        order_range = range(len(shape) - 1, -1, -1)
    elif order == "F":
        order_range = range(len(shape))
    else:
        raise ValueError(f"order must be 'C' or 'F'; got {order!r}")
    expected = 1
    for d in order_range:
        try:
            # ``.rewrite(Pow)`` unwraps ``ipow(N, k)`` (the integer-power form
            # RelaxIntegerPowers freezes a symbolic size / stride into) to ``N**k``
            # so the packed-C stride ``ipow(N, 2)`` compares equal to ``N*N``;
            # without it the opaque ``ipow`` never simplifies against ``expected``
            # and a genuinely packed layout is misread as padded (heat3d).
            diff = dace.symbolic.simplify(sympy.sympify(strides[d] - expected).rewrite(sympy.Pow))
            if diff != 0:
                return False
        except Exception:  # noqa: BLE001 -- conservative refusal on un-comparable expressions.
            return False
        expected = expected * shape[d]
    return True


def validate_packed_layout(node_label, conn_name, desc):
    """Refuse any source / dest array whose stride pattern is neither packed C
    nor packed Fortran (design section 2.3).

    Padded layouts -- where strides exceed the product of inner dims -- raise
    :class:`NotImplementedError` until per-arch codegen support lands. 1-D
    arrays trivially satisfy both packings and are accepted iff their single
    stride is 1.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the array (typically ``_src``
        or ``_dst``).
    :param desc: The array descriptor (``dace.data.Data`` subclass) wired to
        the connector.
    :raises NotImplementedError: On non-packed-C non-packed-Fortran layout.
    """
    import dace
    if not isinstance(desc, dace.data.Array):
        return  # Scalars / Streams have no per-dim stride pattern to check.
    shape = tuple(desc.shape)
    strides = tuple(desc.strides)
    if len(shape) == 0:
        return
    if len(shape) == 1:
        try:
            if dace.symbolic.simplify(strides[0] - 1) != 0:
                raise NotImplementedError(f"{node_label}: {conn_name!r} has non-unit stride "
                                          f"{strides[0]} on its single dim; only packed layouts are "
                                          f"supported (section 2.3).")
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001
            raise NotImplementedError(f"{node_label}: {conn_name!r} stride {strides[0]} could not be "
                                      f"verified against the packed-layout invariant (section 2.3).")
        return
    if not (_strides_match_packed(shape, strides, "C") or _strides_match_packed(shape, strides, "F")):
        raise NotImplementedError(f"{node_label}: {conn_name!r} has non-packed stride pattern "
                                  f"(shape={shape}, strides={strides}). Only packed-C and packed-"
                                  f"Fortran layouts are supported (section 2.3); padded layouts raise "
                                  f"NotImplementedError until codegen lands.")


def validate_mask_descriptor_lock(node_label, conn_name, desc, widths):
    """Refuse any mask descriptor that breaks the design section 10.2 lock.

    The locked shape: ``Array(shape=widths, dtype=bool_, storage=Register,
    transient=True)``. Anything else -- scalar masks, per-dim masks, non-bool
    predicates, non-Register storage, non-transient -- is rejected with a
    named error so the codegen never silently mis-emits.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the mask (typically ``_mask``
        or ``_o``).
    :param desc: The descriptor (``dace.data.Data`` subclass) of the array
        wired to the connector.
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :raises ValueError: On any descriptor lock violation.
    """
    import dace
    if not isinstance(desc, dace.data.Array):
        raise ValueError(f"{node_label}: {conn_name!r} mask must be a dace.data.Array, "
                         f"got {type(desc).__name__}")
    if tuple(desc.shape) != tuple(widths):
        raise ValueError(f"{node_label}: {conn_name!r} mask shape {tuple(desc.shape)} must "
                         f"match widths {tuple(widths)} (section 10.2)")
    if desc.dtype != dace.bool_:
        raise ValueError(f"{node_label}: {conn_name!r} mask dtype {desc.dtype} must be bool_ "
                         f"(section 10.2)")
    if desc.storage != dace.dtypes.StorageType.Register:
        raise ValueError(f"{node_label}: {conn_name!r} mask storage {desc.storage} must be "
                         f"Register (section 10.2)")
    if not desc.transient:
        raise ValueError(f"{node_label}: {conn_name!r} mask must be transient (section 10.2)")


def gather_lane_offset(deps, widths, conn):
    """Build the row-major flat lane offset C expression into an ``_idx_<d>`` tile.

    Given ``deps = (p_0, ..., p_{n-1})`` (the tile dims the gather expression
    depends on) and the lib node's widths, returns the CPP expression
    ``conn[<flat offset>]`` where the flat offset is
    ``__l<p_0> * (W_<p_1> * W_<p_2> * ...) + __l<p_1> * (W_<p_2> * ...) + ... + __l<p_{n-1}>``.

    For the scalar case (``deps == ()``) returns ``conn[0]``.

    :param deps: Sorted tuple of tile dim indices from :func:`resolve_gather_deps`.
    :param widths: Lib node's full tile widths.
    :param conn: The connector name (e.g. ``"_idx_0"``).
    :returns: A CPP expression string of the form ``conn[<offset>]``.
    """
    if not deps:
        return f"{conn}[0]"
    parts = []
    for i, p in enumerate(deps):
        inner = 1
        for q in deps[i + 1:]:
            inner *= widths[q]
        parts.append(f"__l{p}" if inner == 1 else f"(__l{p} * {inner})")
    return f"{conn}[{' + '.join(parts)}]"
