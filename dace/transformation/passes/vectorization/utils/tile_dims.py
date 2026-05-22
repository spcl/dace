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
    :cvar STRUCTURED: A tile dim's index is a DETERMINISTIC non-affine
        function of the tile var with an affine argument — specifically
        ``int_floor`` / ``int_ceil`` of an affine expression (e.g.
        ``a[i // 2]``, the lane-replication / multiplex pattern). It is a
        regular box (each lane's index is computable at lowering time with
        no data dependence), so it lowers like a gather over an index map
        built by direct substitution — distinct from a true (data-dependent)
        GATHER. The affine-argument requirement is what excludes a
        data-loaded index (``a[int_floor(idx[i], 2)]`` is GATHER, not
        STRUCTURED).
    :cvar GATHER: A source-array dim's index is data-dependent / not a
        perfect box: read from a SEPARATE index array (``b[idx[i]]``), a
        diagonal (``a[i, i]`` — one tile var in two dims), or a dim mixing
        tile vars (``a[i + j]``). Emit ``TileGather`` / ``TileScatter``.
    :cvar UNRECOGNIZED: Anything the classifier cannot put into the
        above buckets (a tile var bound to no subset dim, etc.).
    """

    CONTIGUOUS = "Contiguous"
    STRIDED = "Strided"
    STRUCTURED = "Structured"
    BROADCAST_SYMBOL = "BroadcastSymbol"
    GATHER = "Gather"
    UNRECOGNIZED = "Unrecognized"


@dataclass(frozen=True)
class TileAccessClassification:
    """Output of :func:`classify_tile_access`.

    :param kind: Bucket the operand falls into.
    :param dim_strides: Per tile dim, the integer coefficient of the
        corresponding iter-var in the subset's begin expression. ``1``
        means unit step along that tile dim. Defined for ``CONTIGUOUS``
        and ``STRIDED``; ``(0,) * K`` for ``BROADCAST_SYMBOL``; ``()``
        for ``GATHER`` / ``UNRECOGNIZED``. This is the index coefficient
        only — the array's per-dim memory stride is applied separately at
        lowering time via :attr:`match_dims`.
    :param match_dims: Per tile dim (innermost-last), the source-array
        dimension that tile dim maps to. The lowering steps through the
        array using ``array.strides[match_dims[d]]`` so a transposed /
        non-last mapping (``cc[j, i]`` with tile dim ``j`` -> array dim 0)
        is addressed along the correct axis. Same length as
        ``dim_strides`` for ``CONTIGUOUS`` / ``STRIDED``; ``()`` otherwise.
    """

    kind: TileAccessKind
    dim_strides: Tuple[int, ...] = field(default_factory=tuple)
    match_dims: Tuple[int, ...] = field(default_factory=tuple)


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


_STRUCTURED_FN_NAMES = {"int_floor", "int_ceil"}


def _structured_subtrees(expr):
    """Yield every ``int_floor`` / ``int_ceil`` sub-expression of ``expr``.

    :param expr: A sympy expression (from :func:`pystr_to_symbolic`).
    :returns: List of the structured-function sub-expressions.
    """
    out = []
    if type(expr).__name__ in _STRUCTURED_FN_NAMES:
        out.append(expr)
    for arg in expr.args:
        out.extend(_structured_subtrees(arg))
    return out


def _structured_coeff_in(b_sym, var_sym) -> Optional[int]:
    """Coefficient of ``var_sym`` when it appears ONLY inside ``int_floor`` /
    ``int_ceil`` with an affine argument (a deterministic replication index,
    e.g. ``int_floor(2*i + 1, 2)``), or ``None`` if not such a structured form.

    Returns ``None`` (not structured) when ``var_sym`` appears bare or inside a
    structured function whose argument is non-affine in ``var_sym`` (which is
    how a data-loaded index ``int_floor(idx[i], 2)`` is excluded — the loaded
    value is not affine in the tile var).

    :param b_sym: The dim's begin expression.
    :param var_sym: The tile iter-var symbol.
    :returns: The affine coefficient of ``var_sym`` inside the structured
        function (e.g. ``2`` for ``int_floor(2*i, 2)``), else ``None``.
    """
    if var_sym not in b_sym.free_symbols:
        return None
    subtrees = [s for s in _structured_subtrees(b_sym) if var_sym in s.free_symbols]
    if not subtrees:
        return None
    coeff = None
    reduced = b_sym
    for sub in subtrees:
        # The structured function's first argument (the numerator) must be
        # affine in the tile var — a data-loaded index is not.
        arg_coeff = _coeff_of(sub.args[0], var_sym)
        if arg_coeff is None:
            return None
        coeff = arg_coeff if coeff is None else coeff
        reduced = reduced.subs(sub, 0)
    # Every occurrence of the tile var must be inside the structured funcs.
    if var_sym in reduced.free_symbols:
        return None
    return coeff


@dataclass(frozen=True)
class DimIndex:
    """Per-array-dim summary of how one access dim's index depends on the
    tile iter-vars — the single analysis the box taxonomy projects from.

    :param dep: Tile dims (positions in ``tile_iter_vars``) whose iter-var
        appears in this array dim's begin expression, sorted.
    :param affine_coeffs: ``{tile_dim_p: int_coeff}`` for the affinely-bound
        deps (``base + coeff*v_p``).
    :param structured: ``True`` iff a dep appears only inside an
        ``int_floor`` / ``int_ceil`` with an affine argument (deterministic
        replication, no data dependence).
    """

    dep: Tuple[int, ...]
    affine_coeffs: dict = field(default_factory=dict)
    structured: bool = False


def build_dim_index_map(
    subset: subsets.Range,
    tile_iter_vars: Tuple[str, ...],
) -> Optional[List[DimIndex]]:
    """Build the per-array-dim index map for ``subset`` under ``tile_iter_vars``.

    For each array dim, records which tile vars its begin expression depends on
    and whether each dependence is affine (coefficient extracted) or structured
    (``int_floor`` / ``int_ceil`` of an affine argument). This is the single
    source of truth for :func:`classify_tile_access` and for the gather
    index-map emission.

    :param subset: The memlet subset.
    :param tile_iter_vars: Tile iter-var names, innermost-last.
    :returns: One :class:`DimIndex` per array dim, or ``None`` if a begin
        expression cannot be parsed.
    """
    tile_syms = [pystr_to_symbolic(v) for v in tile_iter_vars]
    name_to_pos = {v: p for p, v in enumerate(tile_iter_vars)}
    out: List[DimIndex] = []
    for (b, _e, _s) in subset.ranges:
        try:
            b_sym = pystr_to_symbolic(str(b))
        except Exception:
            return None
        free = {str(s) for s in b_sym.free_symbols}
        dep = sorted(name_to_pos[v] for v in tile_iter_vars if v in free)
        affine_coeffs: dict = {}
        structured = False
        for p in dep:
            tsym = tile_syms[p]
            c = _coeff_of(b_sym, tsym)
            if c is not None:
                affine_coeffs[p] = c
            elif _structured_coeff_in(b_sym, tsym) is not None:
                structured = True
        out.append(DimIndex(dep=tuple(dep), affine_coeffs=affine_coeffs, structured=structured))
    return out


def classify_tile_access(
    subset: subsets.Range,
    array_strides: Tuple,
    tile_iter_vars: Tuple[str, ...],
) -> TileAccessClassification:
    """Classify how a memlet subset depends on the tile iter-vars.

    Derives the box kind from the per-array-dim :func:`build_dim_index_map`:

    - **BROADCAST_SYMBOL**: no array dim depends on a tile var.
    - **UNRECOGNIZED**: a tile var is bound to no subset dim.
    - **GATHER**: not a perfect box — a tile var spans ≥2 array dims
      (diagonal ``a[i, i]``), a dim mixes ≥2 tile vars (``a[i + j]``), or a
      tile-dependent dim is neither affine nor structured (data-dependent).
    - **STRUCTURED**: a perfect box where a dim's index is ``int_floor`` /
      ``int_ceil`` of an affine argument (``a[i // 2]`` replication).
    - **CONTIGUOUS / STRIDED**: an affine perfect box (CONTIGUOUS = aligned
      to the last K dims, unit coefficients, unit innermost memory stride).

    :param subset: The memlet subset to classify.
    :param array_strides: Strides of the source array (matches the
        subset's dimensionality).
    :param tile_iter_vars: Tile iter-var names, ordered to match
        ``TileDimSpec.iter_vars``.
    :returns: A :class:`TileAccessClassification` (kind + per-tile-dim
        ``dim_strides`` coefficients + ``match_dims``).
    """
    if not isinstance(subset, subsets.Range):
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
    if len(subset) != len(array_strides):
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)

    dims = build_dim_index_map(subset, tile_iter_vars)
    if dims is None:
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)

    K = len(tile_iter_vars)
    if all(not di.dep for di in dims):
        return TileAccessClassification(kind=TileAccessKind.BROADCAST_SYMBOL, dim_strides=(0,) * K)

    # Bijection check: each tile var must bind to exactly one array dim, and
    # each tile-dependent dim to exactly one tile var.
    tilevar_dims = {p: [d for d, di in enumerate(dims) if p in di.dep] for p in range(K)}
    if any(len(ds) == 0 for ds in tilevar_dims.values()):
        return TileAccessClassification(TileAccessKind.UNRECOGNIZED)  # tile var bound to no dim
    if any(len(ds) > 1 for ds in tilevar_dims.values()):
        # A tile var spanning ≥2 array dims is a diagonal (``a[i, i]``).
        # K=1 lowers it via a TileGather over an affine per-dim index map
        # (one shared lane offset folded into every spanned dim). For K>1 a
        # diagonal inside a multi-dim register tile is unsupported — the
        # gather map would have to fold one shared tile var across multiple
        # tile dimensions, which the K-dim gather emitter does not model —
        # so refuse (UNRECOGNIZED) and let the orchestrator skip it cleanly.
        if K > 1:
            return TileAccessClassification(TileAccessKind.UNRECOGNIZED)
        return TileAccessClassification(TileAccessKind.GATHER)        # K=1 diagonal
    if any(len(di.dep) > 1 for di in dims):
        return TileAccessClassification(TileAccessKind.GATHER)        # dim mixes ≥2 tile vars (a[i+j])

    # Perfect box: per tile dim, classify affine vs structured vs neither.
    match_dims: List[int] = []
    per_tile_dim_strides: List[int] = []
    structured = False
    for p in range(K):
        d = tilevar_dims[p][0]
        di = dims[d]
        match_dims.append(d)
        if p in di.affine_coeffs:
            per_tile_dim_strides.append(di.affine_coeffs[p])
        elif di.structured:
            structured = True
            per_tile_dim_strides.append(1)  # structured: index built by substitution at lowering
        else:
            return TileAccessClassification(TileAccessKind.GATHER)    # non-affine, non-structured

    dim_strides = tuple(per_tile_dim_strides)
    match_dims_t = tuple(match_dims)
    if structured:
        return TileAccessClassification(kind=TileAccessKind.STRUCTURED,
                                        dim_strides=dim_strides, match_dims=match_dims_t)

    # CONTIGUOUS = aligned to the last K dims, unit coefficients, unit
    # innermost memory stride; else STRIDED. Both lower via
    # ``strides[match_dims[d]]`` scaled by ``dim_strides[d]``.
    ndim = len(array_strides)
    aligned = all(match_dims[p] == ndim - K + p for p in range(K))
    try:
        innermost_unit_stride = int(array_strides[match_dims[-1]]) == 1
    except (TypeError, ValueError):
        innermost_unit_stride = pystr_to_symbolic(str(array_strides[match_dims[-1]] - 1)) == 0

    if aligned and innermost_unit_stride and all(s == 1 for s in dim_strides):
        kind = TileAccessKind.CONTIGUOUS
    else:
        kind = TileAccessKind.STRIDED
    return TileAccessClassification(kind=kind, dim_strides=dim_strides, match_dims=match_dims_t)
