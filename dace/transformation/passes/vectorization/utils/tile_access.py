# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K-dim tile access classifier for the vectorization tile pipeline.

One primitive -- :func:`classify_tile_access` -- answers: *given a memlet
subset, the K tile iter-vars, and (optionally) the inner SDFG, how does
the access vary as the K-dim tile iterates over its (W_0, ..., W_{K-1})
lanes?*

The classifier is **target-isa-agnostic** and does **no SDFG mutation**:
its sole output is a :class:`TileAccess` record. The emitter consumes
that record to pick a tile lib node (``TileLoad`` / ``TileLoadStrided``
/ ``TileGather`` / ``TileStore`` / ``TileScatter`` / ``TileBroadcast``)
and arch-specific intrinsics live in the lib-node expansions.

Per-dim classification
----------------------

Every dim of the subset is tagged independently as one of
:class:`PerDimKind`:

* **BROADCAST** -- no tile iter-var anywhere in the dim's expression
  (loop-invariant; one element splats to all lanes of this dim).
* **STRUCTURED_1** -- one tile iter-var as a *direct top-level symbol*
  with identity coefficient (``iter_var + c``). The cleanest case: lane
  ``l`` reads ``addr + l * elem_size``.
* **AFFINE** -- one or more tile iter-vars present as direct symbols
  but with non-unit coefficient (``2*i + 3``) or multiple iter-vars
  combined (``i + j``). Emitter chooses strided-load intrinsic vs.
  loop-of-loads vs. fallback to GATHER per arch.
* **GATHER** -- a tile iter-var appears inside a :class:`Subscript`
  (``arr[idx[i]]``). Data-dependent access; always lowers to TileGather.

Whole-subset composition (DIAGONAL / TRANSPOSE flags)
-----------------------------------------------------

The whole-subset kind is the strongest per-dim kind, in order
``GATHER > AFFINE > STRUCTURED > BROADCAST``. Two composition flags are
also captured:

* **diagonal** -- a STRUCTURED_1 iter-var appears as the direct symbol
  in *multiple* dims (``arr[i, i]``). Per the locked design, lowers as
  GATHER with a 1-D index tile.
* **transpose** -- STRUCTURED_1 iter-vars appear in a non-canonical
  permutation of ``spec.iter_vars``. Lowers as GATHER with a permuted
  index tile (or a transpose intrinsic when the emitter can use one).

The classifier captures these as informational fields; the emitter
decides the actual lib-node target. Per the user's locked direction,
DIAGONAL and TRANSPOSE both fold into ``TileGather`` today.

dim_strides convention
----------------------

For non-GATHER classifications the per-dim coefficient is exposed as
``dim_strides`` so the load/store lib nodes can directly consume it:

* ``0`` -- BROADCAST dim (this dim doesn't vary per lane; splat).
* ``1`` -- STRUCTURED_1 dim (lane ``l`` reads index ``l`` more).
* ``c > 1`` -- AFFINE dim with coefficient ``c``.

Mixed-rank broadcasts (K=0 -> K=2 full splat; K=1 -> K=2 broadcast on
one dim; etc.) drop out automatically: the BROADCAST dims get
``dim_stride = 0`` and the load lib node's expansion replicates that
value across the dim's lanes.

Gather-side composition rule
----------------------------

When ANY dim is GATHER, the whole-subset kind is GATHER. Non-GATHER
dims fold into the gather's index expression (the structured dims
contribute affine sub-expressions inline; no separate index tile
materialisation is needed for them).
"""
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import sympy

from dace import symbolic
from dace.sdfg import SDFG, nodes
from dace.subsets import Range


class PerDimKind(enum.Enum):
    """Per-dim classification of a memlet subset for tile lowering.

    All five kinds sit on a unified ``replicate_factor`` axis -- the
    number of consecutive lanes that read the same source element on
    this dim. ``BROADCAST`` (factor = "all W lanes"), ``REPLICATE``
    (factor = k, 1 < k < W) and ``STRUCTURED_1`` (factor = 1) are points
    on this spectrum; the codegen uses the per-dim factor uniformly to
    pick the right intrinsic. ``AFFINE`` and ``GATHER`` step outside
    the spectrum because non-unit-stride and data-dependent accesses
    aren't characterised by a replicate factor.
    """

    #: No tile iter-var anywhere in this dim's expression -- loop-invariant.
    #: All W lanes of this dim share one source element. The endpoint of
    #: the replicate-factor spectrum (factor = W); the codegen loads one
    #: element and splats it across the whole dim.
    BROADCAST = "broadcast"

    #: A tile iter-var as a direct top-level symbol with identity
    #: coefficient (``iter_var + c``). Each lane reads a distinct
    #: consecutive element (replicate_factor = 1). Lane ``l`` reads
    #: ``addr + l``.
    STRUCTURED_1 = "structured-1"

    #: A tile iter-var inside ``int_floor(c * iter_var + c0, k)`` or
    #: ``int_ceil(...)`` -- the dim varies WITHIN itself but groups
    #: ``k`` consecutive lanes onto the same source element. The
    #: replicate factor ``k`` lives in
    #: :attr:`TileAccess.replicate_factor_per_dim`. The codegen loads
    #: ``W / k`` elements and group-broadcasts each ``k`` times.
    REPLICATE = "replicate"

    #: Tile iter-var(s) as direct top-level symbol(s) with non-unit
    #: coefficient (``2*i``) or multiple iter-vars combined (``i + j``).
    #: ``dim_stride`` is the coefficient when isolatable; otherwise the
    #: emitter falls back to GATHER.
    AFFINE = "affine"

    #: A tile iter-var appears nested inside a :class:`Subscript`
    #: (``arr[idx[i]]``). Data-dependent index; emits :class:`TileGather`
    #: with the index expression.
    GATHER = "gather"


class TileAccessKind(enum.Enum):
    """Whole-subset kind. The strongest per-dim kind wins, in the order
    ``GATHER > AFFINE > STRUCTURED > BROADCAST``. Two specialised
    compositions (DIAGONAL / TRANSPOSE) of STRUCTURED_1 dims are tagged
    on the :class:`TileAccess` record for emitter information only --
    both fold into ``TileGather`` per the locked design."""

    BROADCAST = "broadcast"
    STRUCTURED = "structured"
    AFFINE = "affine"
    GATHER = "gather"


@dataclass(frozen=True)
class TileAccess:
    """Result of :func:`classify_tile_access` (data only, no behaviour).

    The emitter reads this record + the surrounding scope (mask
    availability, target_isa) to pick a lib node and wire its
    properties.
    """

    #: Whole-subset kind. Determined by the strongest per-dim kind.
    kind: TileAccessKind

    #: Per-dim classification, one entry per subset dim (in subset order).
    per_dim_kind: Tuple[PerDimKind, ...]

    #: Per-dim stride. ``0`` = broadcast dim; ``1`` = identity stride;
    #: ``c > 1`` = affine coefficient. ``None`` for GATHER dims (the
    #: stride lives in the gather-index expression). Same length as
    #: ``per_dim_kind``.
    dim_strides: Tuple[Optional[int], ...]

    #: For dims where the per-dim kind is STRUCTURED_1 or AFFINE: which
    #: tile iter-var is the dim's per-lane stride driver. ``None`` for
    #: BROADCAST and GATHER dims. Same length as ``per_dim_kind``.
    dim_iter_var: Tuple[Optional[str], ...]

    #: For GATHER dims: the in-scope :class:`AccessNode` of the index
    #: array, when resolvable. ``None`` for non-GATHER dims OR when the
    #: index source isn't a direct array read. Same length as
    #: ``per_dim_kind``.
    gather_index_per_dim: Tuple[Optional[nodes.AccessNode], ...]

    #: For dims where the per-dim kind is STRUCTURED_1 or AFFINE: the
    #: constant offset (the ``c`` in ``iter_var + c``). ``None`` for
    #: BROADCAST and GATHER dims.
    dim_offset: Tuple[Optional[sympy.Expr], ...]

    #: Per-dim replicate factor (lanes-per-distinct-value within the dim).
    #: The unifying field across BROADCAST / STRUCTURED_1 / REPLICATE
    #: regimes:
    #:
    #: * ``None`` for BROADCAST dims -- the codegen reads one element and
    #:   splats it to the dim's full W lanes (equivalently, factor = W
    #:   but the dim's W isn't visible to the classifier, so ``None`` is
    #:   the sentinel for "all lanes").
    #: * ``1`` for STRUCTURED_1 dims -- each lane reads a distinct
    #:   consecutive element, no replication.
    #: * ``k > 1`` for REPLICATE dims (``int_floor`` / ``int_ceil`` of an
    #:   affine arg) -- ``k`` consecutive lanes share each source
    #:   element; the codegen loads ``W / k`` elements and group-broadcasts
    #:   each ``k`` times.
    #: * ``None`` for AFFINE and GATHER dims (the concept doesn't apply).
    #:
    #: Same length as :attr:`per_dim_kind`.
    replicate_factor_per_dim: Tuple[Optional[int], ...]

    #: Diagonal composition: maps a tile iter-var name to the tuple of
    #: subset dims it appears in directly. Only populated when one
    #: iter-var spans multiple dims. Empty dict when no diagonal pattern.
    diagonal: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    #: Transpose composition: when present, the permutation of tile
    #: iter-vars across the subset dims, as a tuple ``(perm_0, ..., perm_{K-1})``
    #: with ``perm_d`` the iter-var INDEX in ``spec.iter_vars`` that
    #: drives dim ``d``. ``None`` when the dim order is canonical or
    #: the subset isn't a pure permutation.
    transpose: Optional[Tuple[int, ...]] = None


# ----- internal helpers --------------------------------------------------


def _safe_sympify(expr) -> Optional[sympy.Expr]:
    """Best-effort sympify. Returns ``None`` on failure (the caller treats
    unparseable expressions as opaque -- typically classified as GATHER
    or refused)."""
    try:
        return symbolic.pystr_to_symbolic(str(expr))
    except Exception:
        return None


def _direct_symbols(expr: sympy.Expr) -> Set[str]:
    """Set of symbol names that appear **outside any gather Subscript**
    in ``expr``. Math functions on iter-vars (``floor(i / 4)``,
    ``exp(i)``) still expose ``i`` as direct -- the iter-var participates
    in the address arithmetic without going through a memory load; the
    affine-coefficient analysis then catches the non-affine cases.
    :class:`~dace.symbolic.Subscript` is the only stop boundary because
    it explicitly represents a data-dependent access."""
    if expr is None:
        return set()
    if isinstance(expr, sympy.Symbol):
        return {str(expr)}
    # Subscript is the gather stop boundary: its inner symbols are
    # gather-index inputs, not address-arithmetic contributors.
    if isinstance(expr, symbolic.Subscript):
        return set()
    args = getattr(expr, 'args', None)
    if not args:
        return set()
    result: Set[str] = set()
    for arg in args:
        result |= _direct_symbols(arg)
    return result


def _gather_subscripts(expr: sympy.Expr) -> List[symbolic.Subscript]:
    """Every :class:`Subscript` node anywhere in ``expr``. Used to
    detect data-dependent indices (``arr[idx[i]]``)."""
    if expr is None:
        return []
    result: List[symbolic.Subscript] = []
    if isinstance(expr, symbolic.Subscript):
        result.append(expr)
    args = getattr(expr, 'args', None)
    if args:
        for arg in args:
            result.extend(_gather_subscripts(arg))
    return result


def _find_named_symbol(expr: sympy.Expr, var_name: str) -> Optional[sympy.Symbol]:
    """Return the actual :class:`sympy.Symbol` instance in ``expr`` whose
    name is ``var_name``. DaCe's ``symbol`` subclasses :class:`sympy.Symbol`
    but does not compare equal to a freshly-constructed ``sympy.Symbol``;
    :func:`sympy.Poly` treats the mismatched generator as opaque (degree
    0). Picking the real symbol out of ``expr.free_symbols`` lets the
    Poly call see the variable correctly."""
    if expr is None:
        return None
    for s in expr.free_symbols:
        if str(s) == var_name:
            return s
    return None


def _affine_coeff_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Return the coefficient of ``var_name`` in ``expr`` if ``expr`` is
    affine in ``var_name`` (i.e. ``expr = coeff * var + rest`` with
    ``rest`` having no ``var_name``). ``None`` if non-affine or
    unresolvable."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or sympy.Symbol(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    if poly.degree() == 0:
        return sympy.Integer(0)
    # Degree 1: coefficient of the variable.
    return poly.coeff_monomial(sym)


def _affine_offset_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Constant term of ``expr`` w.r.t. ``var_name``, i.e. the ``c`` in
    ``coeff * var + c``. ``None`` if non-affine."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or sympy.Symbol(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    # ``nth(0)`` returns the constant term.
    return poly.nth(0)


def _detect_replicate_factor(expr: sympy.Expr, var_name: str) -> Optional[int]:
    """Detect the ``int_floor(affine_in_var, k)`` / ``int_ceil(...)``
    pattern at the top of ``expr`` and return the integer divisor ``k``
    when the inner argument is affine in ``var_name``.

    Returns ``None`` when:

    * ``expr`` is not an ``int_floor`` / ``int_ceil`` call,
    * the divisor isn't a concrete positive integer,
    * the dividend isn't affine in ``var_name``.

    This is the within-dim replicate detection: the dim still walks the
    source array (so it's not a full-dim broadcast), but every ``k``
    consecutive lanes share the same source element. The codegen reads
    a ``W / k``-element contracted box and group-broadcasts.
    """
    if expr is None:
        return None
    fname = type(expr).__name__
    # DaCe has two equivalent function names for these patterns:
    # ``int_floor`` / ``int_ceil`` are the user-facing forms, and
    # ``__int_floor`` / ``__int_ceil`` are the Python-operator-derived
    # forms (produced when ``i // 4`` is parsed). Both lower to the
    # same operation.
    if fname not in ("int_floor", "int_ceil", "__int_floor", "__int_ceil"):
        return None
    if len(expr.args) != 2:
        return None
    dividend, divisor = expr.args
    # Divisor must be a concrete positive integer for the replicate
    # factor to be meaningful.
    try:
        k = int(divisor)
    except (TypeError, ValueError):
        return None
    if k <= 1:
        return None
    # Dividend must be affine in ``var_name`` (so the replication is
    # regular -- ``int_floor(idx[i], 2)`` is data-dependent and lowers
    # as GATHER, not REPLICATE).
    coeff = _affine_coeff_for(dividend, var_name)
    if coeff is None:
        return None
    return k


def _resolve_gather_index_an(inner_sdfg: Optional[SDFG], expr: sympy.Expr) -> Optional[nodes.AccessNode]:
    """If ``expr`` contains exactly one Subscript whose base is an array
    name in ``inner_sdfg.arrays``, find an :class:`AccessNode` for that
    array in any state and return it. Returns ``None`` when there are
    zero or multiple subscripts, the base isn't an array, or no
    AccessNode exists."""
    if inner_sdfg is None or expr is None:
        return None
    subs = _gather_subscripts(expr)
    if len(subs) != 1:
        return None
    base = subs[0].args[0]
    base_name = str(base)
    if base_name not in inner_sdfg.arrays:
        return None
    for st in inner_sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == base_name:
                return n
    return None


# ----- public API --------------------------------------------------------


def classify_tile_access(subset: Range,
                         iter_vars: Sequence[str],
                         inner_sdfg: Optional[SDFG] = None) -> TileAccess:
    """Classify a memlet subset for tile lib-node dispatch.

    :param subset: The :class:`Range` to classify (typically a memlet's
        ``subset`` field).
    :param iter_vars: The K tile iter-var names, in tile-lane order
        (innermost-last by convention).
    :param inner_sdfg: Optional inner SDFG used to resolve GATHER index
        AccessNodes. Pass ``None`` when the analysis runs outside the
        body context (the gather-index field is left empty).
    :returns: A :class:`TileAccess` record. Always returns; never
        raises. Unrecognisable patterns degrade to GATHER (correctness
        fallback).
    """
    iter_var_set: Set[str] = set(iter_vars)
    n_dims = len(subset.ranges)

    per_dim_kind: List[PerDimKind] = []
    dim_strides: List[Optional[int]] = []
    dim_iter_var: List[Optional[str]] = []
    gather_index_per_dim: List[Optional[nodes.AccessNode]] = []
    dim_offset: List[Optional[sympy.Expr]] = []
    replicate_factor_per_dim: List[Optional[int]] = []

    # Per-iter-var tracking for diagonal / transpose detection.
    iter_var_in_dim: Dict[str, List[int]] = {v: [] for v in iter_vars}
    dim_to_canonical_iter_var: List[Optional[int]] = []

    for d, (lo, _hi, _stp) in enumerate(subset.ranges):
        lo_sym = _safe_sympify(lo)
        direct = _direct_symbols(lo_sym) if lo_sym is not None else set()
        direct_tile_vars = direct & iter_var_set

        # Stop 1: any tile iter-var inside a Subscript -> GATHER dim.
        # Recompute the symbols nested inside Subscripts to detect this.
        gather_dim = False
        if lo_sym is not None:
            for sub in _gather_subscripts(lo_sym):
                # A Subscript stores (container, idx0, idx1, ...) in args.
                # Iter-var appears nested if it shows up in any subscript
                # argument's free symbols.
                for sub_arg in sub.args[1:]:
                    fs = {str(s) for s in sub_arg.free_symbols}
                    if fs & iter_var_set:
                        gather_dim = True
                        break
                if gather_dim:
                    break
        if gather_dim:
            per_dim_kind.append(PerDimKind.GATHER)
            dim_strides.append(None)
            dim_iter_var.append(None)
            gather_index_per_dim.append(_resolve_gather_index_an(inner_sdfg, lo_sym))
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue

        # Stop 2: no tile iter-var as direct symbol -> BROADCAST dim
        # (replicate_factor = None, meaning all W lanes of the dim
        # share one element).
        if not direct_tile_vars:
            per_dim_kind.append(PerDimKind.BROADCAST)
            dim_strides.append(0)
            dim_iter_var.append(None)
            gather_index_per_dim.append(None)
            dim_offset.append(_safe_sympify(lo))
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue

        # Stop 3: exactly one tile iter-var directly -> STRUCTURED_1,
        # REPLICATE, or AFFINE.
        if len(direct_tile_vars) == 1:
            tvar = next(iter(direct_tile_vars))
            # Stop 3a: ``int_floor(c*tvar + c0, k)`` / ``int_ceil(...)``
            # -> REPLICATE with factor k. Detected BEFORE the affine
            # check so the function call doesn't fall through to AFFINE.
            replicate_k = _detect_replicate_factor(lo_sym, tvar)
            if replicate_k is not None:
                per_dim_kind.append(PerDimKind.REPLICATE)
                dim_strides.append(1)  # contracted-box stride = 1
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(None)
                replicate_factor_per_dim.append(replicate_k)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            coeff = _affine_coeff_for(lo_sym, tvar)
            offset = _affine_offset_for(lo_sym, tvar)
            if coeff is not None and coeff == 1:
                per_dim_kind.append(PerDimKind.STRUCTURED_1)
                dim_strides.append(1)
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(offset)
                replicate_factor_per_dim.append(1)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            if coeff is not None:
                # Affine in one iter-var. Coerce to int when possible
                # (the emitter wants a concrete stride for the lib node).
                try:
                    int_coeff = int(coeff)
                except (TypeError, ValueError):
                    int_coeff = None
                per_dim_kind.append(PerDimKind.AFFINE)
                dim_strides.append(int_coeff)
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(offset)
                replicate_factor_per_dim.append(None)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            # Non-affine and not int_floor/int_ceil (e.g. ``i**2``):
            # fall through to AFFINE without an int stride; the emitter
            # degrades to GATHER with the per-lane expression.
            per_dim_kind.append(PerDimKind.AFFINE)
            dim_strides.append(None)
            dim_iter_var.append(tvar)
            gather_index_per_dim.append(None)
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            iter_var_in_dim[tvar].append(d)
            dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
            continue

        # Stop 4: multiple tile iter-vars directly -> AFFINE (multi-var).
        per_dim_kind.append(PerDimKind.AFFINE)
        dim_strides.append(None)
        # No single representative iter-var for the dim; report the
        # first one alphabetically for consistency.
        rep = sorted(direct_tile_vars)[0]
        dim_iter_var.append(rep)
        gather_index_per_dim.append(None)
        dim_offset.append(None)
        replicate_factor_per_dim.append(None)
        for tv in direct_tile_vars:
            iter_var_in_dim[tv].append(d)
        dim_to_canonical_iter_var.append(list(iter_vars).index(rep))

    # Whole-subset kind: strongest per-dim kind. REPLICATE shares the
    # same whole-subset bucket as STRUCTURED (both are perfectly
    # regular -- the codegen picks contiguous vs group-broadcast from
    # the per-dim ``replicate_factor`` field).
    has_gather = any(k == PerDimKind.GATHER for k in per_dim_kind)
    has_affine = any(k == PerDimKind.AFFINE for k in per_dim_kind)
    has_struct = any(k in (PerDimKind.STRUCTURED_1, PerDimKind.REPLICATE) for k in per_dim_kind)
    if has_gather:
        kind = TileAccessKind.GATHER
    elif has_affine:
        kind = TileAccessKind.AFFINE
    elif has_struct:
        kind = TileAccessKind.STRUCTURED
    else:
        kind = TileAccessKind.BROADCAST

    # Diagonal: any iter-var spans >= 2 dims.
    diagonal = {v: tuple(dims) for v, dims in iter_var_in_dim.items() if len(dims) >= 2}

    # Transpose: STRUCTURED with iter-vars in non-canonical order.
    # Canonical order: dim ``d`` carries ``iter_vars[d]``.
    transpose: Optional[Tuple[int, ...]] = None
    if kind == TileAccessKind.STRUCTURED and len(per_dim_kind) == len(iter_vars):
        # Build the permutation from each dim's iter-var index.
        perm = tuple(dim_to_canonical_iter_var)  # type: ignore[arg-type]
        if all(p is not None for p in perm) and tuple(perm) != tuple(range(len(iter_vars))):
            transpose = perm  # type: ignore[assignment]

    return TileAccess(
        kind=kind,
        per_dim_kind=tuple(per_dim_kind),
        dim_strides=tuple(dim_strides),
        dim_iter_var=tuple(dim_iter_var),
        gather_index_per_dim=tuple(gather_index_per_dim),
        dim_offset=tuple(dim_offset),
        replicate_factor_per_dim=tuple(replicate_factor_per_dim),
        diagonal=diagonal,
        transpose=transpose,
    )
