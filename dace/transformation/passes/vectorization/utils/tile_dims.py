# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K-dim tile-dimension analysis for the v2 multi-dim vectorization track.

Exports a frozen :class:`TileDimSpec` describing which K innermost map
parameters are being tiled (with their widths and original exclusive
upper bounds), plus :func:`classify_tile_access` which inspects a
memlet subset and returns one :class:`TileAccessKind` summarizing how
the operand depends on the tile iter-vars.

Both pieces are consumed by the prep / emitter passes; this module
contains no SDFG mutation — only analysis.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import dace
from dace import subsets
from dace.symbolic import pystr_to_symbolic


@dataclass(frozen=True)
class TileDimSpec:
    """Per-map tile specification.

    All three tuples have the same length ``K`` and are ordered
    innermost-last (matches ``map.params[-K:]``). ``global_ubs`` are
    the original exclusive upper bounds captured *before* any
    remainder split, so the mask init has a stable reference.

    :param iter_vars: Per-dim outer-map iter-var name.
    :param widths: Per-dim tile width.
    :param global_ubs: Per-dim exclusive upper-bound expression
        (string form so symbolic expressions survive serialization).
    """

    iter_vars: Tuple[str, ...]
    widths: Tuple[int, ...]
    global_ubs: Tuple[str, ...]

    def __post_init__(self):
        if not (1 <= len(self.widths) <= 3):
            raise ValueError(f"TileDimSpec: K must be in {{1, 2, 3}}, got {len(self.widths)}")
        if not (len(self.iter_vars) == len(self.widths) == len(self.global_ubs)):
            raise ValueError(
                f"TileDimSpec: iter_vars / widths / global_ubs lengths must agree; "
                f"got {len(self.iter_vars)}, {len(self.widths)}, {len(self.global_ubs)}"
            )

    @property
    def K(self) -> int:
        """Return the number of tiled dimensions (1, 2, or 3)."""
        return len(self.widths)


class TileAccessKind(Enum):
    """Per-operand classification produced by :func:`classify_tile_access`.

    :cvar CONTIGUOUS: Every tile dim's index is ``base_d + tile_var_d``
        and the source array has unit stride on that dim — emit
        :class:`TileLoad` / :class:`TileStore` with ``dim_strides``
        all 1.
    :cvar STRIDED: At least one tile dim's index is
        ``base_d + s_d * tile_var_d`` with ``s_d > 1`` (or a non-unit
        array stride on that dim) — emit :class:`TileLoad` /
        :class:`TileStore` with the actual ``dim_strides``.
    :cvar BROADCAST_SYMBOL: The subset has no dependency on any tile
        iter-var — the operand is broadcast inline as
        :class:`TileBinop` ``kind=Symbol``.
    :cvar GATHER: At least one source-array dim's index is read from a
        SEPARATE index array (e.g. ``b[idx[i]]``). Emit ``TileGather``.
        After ``simplify=True`` the indirection lives INSIDE a body
        NestedSDFG as a sliced access node ``arr[__sym_X] -> [0]``;
        the classifier here only emits the enum value, the actual
        emission requires NestedSDFG-body pattern-matching (post-MVP).
    :cvar UNRECOGNIZED: Anything the classifier cannot put into the
        above buckets (halve-index multiplex, irregular indices, etc.).
    """

    CONTIGUOUS = "Contiguous"
    STRIDED = "Strided"
    BROADCAST_SYMBOL = "BroadcastSymbol"
    GATHER = "Gather"
    UNRECOGNIZED = "Unrecognized"


@dataclass(frozen=True)
class TileAccessClassification:
    """Output of :func:`classify_tile_access`.

    :param kind: Bucket the operand falls into.
    :param dim_strides: Per tile dim, the integer coefficient of the
        corresponding iter-var in the subset's begin expression. ``1``
        means contiguous along that tile dim. Defined for
        ``CONTIGUOUS`` and ``STRIDED``; ``(0,) * K`` for
        ``BROADCAST_SYMBOL``; ``()`` for ``GATHER`` / ``UNRECOGNIZED``.
    """

    kind: TileAccessKind
    dim_strides: Tuple[int, ...] = field(default_factory=tuple)


def _coeff_of(expr_sym, var_sym) -> Optional[int]:
    """Return the integer coefficient of ``var_sym`` in ``expr_sym``.

    :param expr_sym: Sympy expression.
    :param var_sym: Variable to extract the coefficient of.
    :returns: Integer coefficient if ``expr_sym`` is linear in
        ``var_sym`` with an integer coefficient and no other tile-var
        dependency on this expression; else ``None``.
    """
    try:
        poly = expr_sym.as_poly(var_sym)
    except Exception:
        return None
    if poly is None:
        return None
    if poly.degree() != 1:
        return None
    coeff = poly.coeff_monomial(var_sym)
    try:
        ci = int(coeff)
    except (TypeError, ValueError):
        return None
    return ci


def classify_tile_access(
    subset: subsets.Range,
    array_strides: Tuple,
    tile_iter_vars: Tuple[str, ...],
) -> TileAccessClassification:
    """Classify how a memlet subset depends on the tile iter-vars.

    Operates per tile dim of the source array: locates the subset's
    begin expression along each tile dim, extracts the linear
    coefficient of the matching tile iter-var, and combines per-dim
    findings into the overall :class:`TileAccessKind`. Returns
    :attr:`TileAccessKind.UNRECOGNIZED` for shapes T3 does not handle
    (gather, halve-index, mixed kinds).

    :param subset: The memlet subset to classify.
    :param array_strides: Strides of the source array (matches the
        subset's dimensionality).
    :param tile_iter_vars: Tile iter-var names, ordered to match
        ``TileDimSpec.iter_vars``.
    :returns: A :class:`TileAccessClassification` describing the kind
        plus per-tile-dim stride coefficients.
    """
    if not isinstance(subset, subsets.Range):
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
    if len(subset) != len(array_strides):
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)

    tile_syms = [pystr_to_symbolic(v) for v in tile_iter_vars]
    free_syms_per_dim: List[set] = []
    for (b, e, _) in subset.ranges:
        try:
            b_sym = pystr_to_symbolic(str(b))
        except Exception:
            return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
        free_syms_per_dim.append({str(s) for s in b_sym.free_symbols})

    tile_var_names = set(tile_iter_vars)
    any_tile_dep = any(free & tile_var_names for free in free_syms_per_dim)
    if not any_tile_dep:
        return TileAccessClassification(
            kind=TileAccessKind.BROADCAST_SYMBOL,
            dim_strides=(0,) * len(tile_iter_vars),
        )

    # For each tile iter-var, locate the array dim that carries it as
    # the (only) tile-dependent variable, extract its linear coefficient
    # and multiply by the array stride for that dim.
    per_tile_dim_strides: List[int] = []
    used_array_dims: set = set()
    for tvar, tsym in zip(tile_iter_vars, tile_syms):
        match_dim = None
        for d, free in enumerate(free_syms_per_dim):
            if d in used_array_dims:
                continue
            tile_deps = free & tile_var_names
            if tile_deps == {tvar}:
                match_dim = d
                break
        if match_dim is None:
            return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
        b_sym = pystr_to_symbolic(str(subset.ranges[match_dim][0]))
        coeff = _coeff_of(b_sym, tsym)
        if coeff is None:
            return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
        try:
            arr_stride = int(array_strides[match_dim])
        except (TypeError, ValueError):
            arr_stride = 1
        per_tile_dim_strides.append(coeff * arr_stride)
        used_array_dims.add(match_dim)

    dim_strides = tuple(per_tile_dim_strides)
    if all(s == 1 for s in dim_strides):
        kind = TileAccessKind.CONTIGUOUS
    else:
        kind = TileAccessKind.STRIDED
    return TileAccessClassification(kind=kind, dim_strides=dim_strides)
