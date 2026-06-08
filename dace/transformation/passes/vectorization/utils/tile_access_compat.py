# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compat shim: drive the legacy :class:`TileAccessClassification` API
from the per-dim classifier in :mod:`tile_access`.

The vectorization descent (``promote_nsdfg_body_to_tiles`` /
``emit_tile_ops``) currently consumes the legacy enum and
``(dim_strides, match_dims)`` tuple. The locked-in design replaces the
legacy whole-subset enum with the per-dim classifier in
:func:`tile_access.classify_tile_access`, but the consumer rewrite is a
separate slice. This adapter calls the new classifier and packages the
result in the legacy format so the consumers don't change yet -- the
descent gets the new analysis under the existing API.

Mapping (locked-in design):

* every per-dim ``BROADCAST`` -> legacy ``BROADCAST_SYMBOL``.
* every per-dim ``STRUCTURED_1`` (and/or ``BROADCAST``) -> legacy
  ``CONTIGUOUS``.
* any per-dim ``AFFINE`` (with integer coefficient) -> legacy
  ``STRIDED``.
* any per-dim ``GATHER`` -> legacy ``GATHER``.
* any per-dim ``AFFINE`` with non-isolable coefficient (e.g. ``i // 2``)
  -> legacy ``STRUCTURED`` (lowerable as GATHER over an index map).

The ``match_dims`` permutation is the inverse of the new classifier's
per-dim ``dim_iter_var`` field: for each tile lane index ``k``, find the
source dim ``d`` whose iter-var is ``iter_vars[k]``.
"""
from typing import Optional, Sequence, Tuple

from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.tile_access import (
    PerDimKind,
    classify_tile_access as _classify_new,
)
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessClassification,
    TileAccessKind,
)


def _compute_match_dims(dim_iter_var: Sequence[Optional[str]], iter_vars: Sequence[str]) -> Tuple[int, ...]:
    """Permutation: ``match_dims[k]`` is the source dim that tile lane
    ``k`` (iter-var ``iter_vars[k]``) addresses.

    Returns an identity permutation when the source has more dims than
    tile lanes and not every lane has a distinct driver, or when any
    lane has no driver (BROADCAST or GATHER dims).
    """
    K = len(iter_vars)
    out = [None] * K  # type: ignore[var-annotated]
    used: set = set()
    for k, iv in enumerate(iter_vars):
        for d, driver in enumerate(dim_iter_var):
            if d in used:
                continue
            if driver == iv:
                out[k] = d
                used.add(d)
                break
    # Default unassigned lanes to a stable identity-like mapping; the
    # consumer only relies on this when ``kind in (CONTIGUOUS, STRIDED)``
    # and in that case every lane has a driver, so the fallback only
    # matters for the corner cases.
    for k in range(K):
        if out[k] is None:
            out[k] = k
    return tuple(out)  # type: ignore[return-value]


def classify_tile_access_compat(subset: Range, array_strides: Sequence,
                                tile_iter_vars: Sequence[str]) -> TileAccessClassification:
    """Adapter: run the per-dim classifier and return a legacy
    :class:`TileAccessClassification`. Drop-in replacement for the
    legacy ``classify_tile_access`` signature.

    ``array_strides`` is accepted for legacy-signature parity but is
    not consulted -- the new classifier infers stride information from
    the subset alone; transpose / row-major mapping handling that used
    to rely on ``array_strides`` falls out of the per-dim ``dim_iter_var``
    permutation.
    """
    iter_vars = tile_iter_vars
    ta = _classify_new(subset, iter_vars)

    has_gather = any(k == PerDimKind.GATHER for k in ta.per_dim_kind)
    has_affine = any(k == PerDimKind.AFFINE for k in ta.per_dim_kind)
    has_structured_1 = any(k == PerDimKind.STRUCTURED_1 for k in ta.per_dim_kind)
    has_replicate = any(k == PerDimKind.REPLICATE for k in ta.per_dim_kind)

    if has_gather:
        return TileAccessClassification(kind=TileAccessKind.GATHER)
    # Diagonal: same iter-var directly drives 2+ subset dims (``a[i, i]``).
    # Legacy classifies this as GATHER -- the new per-dim classifier flags
    # it via the ``diagonal`` field but the whole-subset kind stays
    # STRUCTURED. Promote to GATHER here to preserve the contract.
    if ta.diagonal:
        return TileAccessClassification(kind=TileAccessKind.GATHER)
    # REPLICATE (``int_floor`` / ``int_ceil``): the new classifier
    # recognises these as a within-dim group-broadcast optimization
    # carrying a per-dim ``replicate_factor``. The legacy lib nodes
    # don't have a replicate-factor property yet (that's slice 2 of the
    # rollout), so route through GATHER -- the existing TileLoad (gather_dims) path
    # with a computed index map is the safe fallback that preserves
    # correctness. Once the lib node grows the property the shim can
    # switch this to STRUCTURED + populate the factor on the result.
    if has_replicate:
        return TileAccessClassification(kind=TileAccessKind.GATHER)

    if has_affine:
        # Pull the integer coefficients out for the lib node;
        # BROADCAST dims get stride 0, AFFINE dims get their coefficient,
        # STRUCTURED_1 dims get 1. If any AFFINE dim has a non-isolable
        # coefficient (``None`` in the new classifier -- e.g. ``i // 2``),
        # the legacy bucket is GATHER (lowered as a TileLoad (gather_dims) over a
        # computed index map at emit time).
        if any(k == PerDimKind.AFFINE and ta.dim_strides[d] is None for d, k in enumerate(ta.per_dim_kind)):
            return TileAccessClassification(kind=TileAccessKind.GATHER)
        per_dim_int_strides: list = []
        for d, kind in enumerate(ta.per_dim_kind):
            per_dim_int_strides.append(int(ta.dim_strides[d]) if ta.dim_strides[d] is not None else 0)
        # Per tile-iter-var, the corresponding source dim coefficient.
        K = len(iter_vars)
        match = _compute_match_dims(ta.dim_iter_var, iter_vars)
        dim_strides_legacy = tuple(per_dim_int_strides[match[k]] for k in range(K))
        return TileAccessClassification(
            kind=TileAccessKind.STRIDED,
            dim_strides=dim_strides_legacy,
            match_dims=match,
        )

    if has_structured_1:
        K = len(iter_vars)
        match = _compute_match_dims(ta.dim_iter_var, iter_vars)
        dim_strides_legacy = tuple(1 for _ in range(K))
        return TileAccessClassification(
            kind=TileAccessKind.CONTIGUOUS,
            dim_strides=dim_strides_legacy,
            match_dims=match,
        )

    # All dims are BROADCAST.
    return TileAccessClassification(kind=TileAccessKind.BROADCAST_SYMBOL, dim_strides=tuple(0 for _ in iter_vars))
