# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared CPP-codegen helpers for the tile-op lib nodes' ``pure`` expansions.

These helpers keep the bodies short and idiomatic: a K-fold nested
``for``-loop with per-dim lane indices (``__l0, __l1, ...``) rather
than the flattened linear-index-plus-decode shape. Each lib node's
pure expansion plugs in its own per-lane body via :func:`nested_loops`
and uses :func:`tile_offset` to flatten the tile transient's index
(register tiles are always row-major-contiguous).
"""
from typing import Sequence


def nested_loops(widths: Sequence[int], body: str, indent: str = "    ") -> str:
    """Wrap ``body`` in a K-fold nested for-loop iterating per-dim
    lane indices ``__l0, __l1, ...``.

    :param widths: Per-tile-dim widths, innermost-last.
    :param body: The per-lane C++ body (already trailing-``;`` if
        needed); may contain newlines (each line is indented).
    :param indent: One indent level (default 4 spaces).
    :returns: A C++ snippet with the nested loops + indented body.
    """
    K = len(widths)
    lines = []
    for d, w in enumerate(widths):
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


def offset_via_strides(coeffs: Sequence[int],
                       strides: Sequence[str],
                       replicate_factors: Sequence[int] = ()) -> str:
    """Return the flat offset expression
    ``sum_d coeffs[d] * strides[d] * (__l<d> / replicate_factors[d])``.

    Used by ``TileLoad`` / ``TileStore`` to address the source / dest
    array's flat memory through its own per-dim strides scaled by the
    optional per-tile-dim ``dim_strides`` coefficient. When
    ``replicate_factors[d] > 1``, the per-dim lane index is divided by
    the replicate factor so ``k`` consecutive lanes index the same
    source element -- the within-dim group-broadcast lowering for the
    ``int_floor`` / ``int_ceil`` regime.

    :param coeffs: Per-tile-dim integer coefficient (``1`` for
        contiguous; >1 for strided access).
    :param strides: Per-tile-dim source-array stride as a C++
        expression (typically the symbolic stride rendered with
        :func:`dace.symbolic.symstr`).
    :param replicate_factors: Per-tile-dim replicate factor (``1`` =
        each lane reads a distinct element; ``k > 1`` = ``k`` lanes
        share each element). Defaults to all-1 (no replication) when
        empty or omitted.
    :returns: The C++ offset expression, or ``"0"`` if K==0.
    """
    if not coeffs:
        return "0"
    parts = []
    for d, (c, s) in enumerate(zip(coeffs, strides)):
        lane = f"__l{d}"
        if d < len(replicate_factors) and replicate_factors[d] > 1:
            lane = f"({lane} / {replicate_factors[d]})"
        parts.append(f"({c} * ({s}) * {lane})")
    return " + ".join(parts)
