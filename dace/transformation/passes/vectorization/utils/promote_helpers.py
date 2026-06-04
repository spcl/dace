# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Body-agnostic helpers shared by the body-NSDFG descent and the
outer-state (inlined) tile-promotion passes.

The descent (:class:`PromoteNSDFGBodyToTiles`) and the K=2 outer-state
port (:class:`PromoteInlinedMapToTiles`, landing in subsequent slices)
share several pure analyses: the perfect-box classifier, the set of
classification kinds that count as "tileable as a single load / store",
and the lane-index expression for tile iter-vars. Centralising them
here keeps the two passes from drifting (different refusal predicates
in body vs outer scope would silently change which kernels promote).

Each helper is body-context-free: callers that need an NSDFG-side
shortcut (e.g. "connector array already at tile shape -> skip
classify") layer it on top of the body-agnostic API rather than
patching the API itself.
"""
from typing import Optional, Tuple

import dace
from dace import subsets

from dace.transformation.passes.vectorization.utils.tile_access_compat import (
    classify_tile_access_compat as classify_tile_access, )
from dace.transformation.passes.vectorization.utils.tile_dims import (TileAccessClassification, TileAccessKind)

#: Per-dim access kinds the perfect-box classifier accepts.
#:
#: These are the kinds the :class:`TileLoad` / :class:`TileStore` lib
#: nodes lower in a single tile op. Anything outside this set
#: (``GATHER``, ``STRUCTURED``, etc.) needs separate handling
#: (gather-index fan-out / refusal).
BOX_KINDS: Tuple[TileAccessKind, ...] = (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED)


def classify_box_for_widths(subset: subsets.Range,
                            arr: dace.data.Data,
                            iter_vars: Tuple[str, ...],
                            widths: Tuple[int, ...]) -> TileAccessClassification:
    """Classify ``subset`` on ``arr`` as a perfect tile box, or raise.

    The classification kinds in :data:`BOX_KINDS` are tileable as a
    single :class:`TileLoad` / :class:`TileStore`; anything else
    (``GATHER``, ``STRUCTURED``, ``BROADCAST_SYMBOL`` except the K=1
    single-lane endpoint) raises :class:`NotImplementedError` so the
    caller can either fall back to a separate promotion or refuse the
    kernel out loud.

    Two short-circuits before invoking the full classifier:

    1. ``tuple(arr.shape) == tuple(widths)`` -- the array is already at
       tile shape (the body-NSDFG descent gets this when a connector
       has been pre-widened to the tile shape; the inlined outer-state
       pass gets this when a transient has been widened by
       ``_widen_body_scalars``). Treat as a perfect contiguous box with
       identity ``match_dims`` and unit ``dim_strides``.
    2. ``BROADCAST_SYMBOL`` at K=1 (``len(iter_vars) == 1``) -- the
       degenerate single-lane postamble case where the subset is
       iter-var-free; a "broadcast scalar load" is a contiguous
       1-element load.

    :param subset: Per-iteration access subset on ``arr``.
    :param arr: Array descriptor being accessed.
    :param iter_vars: Tile iter-var names (innermost-last).
    :param widths: Tile widths matching ``iter_vars``.
    :returns: A :class:`TileAccessClassification` describing the box.
    :raises NotImplementedError: For non-box access kinds.
    """
    if tuple(arr.shape) == tuple(widths):
        K = len(widths)
        return TileAccessClassification(kind=TileAccessKind.CONTIGUOUS,
                                        dim_strides=(1, ) * K,
                                        match_dims=tuple(range(K)))
    cls = classify_tile_access(subset, tuple(arr.strides), iter_vars)
    if cls.kind in BOX_KINDS:
        return cls
    if cls.kind == TileAccessKind.BROADCAST_SYMBOL and len(iter_vars) == 1:
        return TileAccessClassification(kind=TileAccessKind.CONTIGUOUS, dim_strides=(1, ), match_dims=(0, ))
    raise NotImplementedError(f"classify_box_for_widths: access {subset} is {cls.kind.value}; "
                              f"only perfect-box (contiguous / strided) loads/stores are supported -- "
                              f"promote separately (gather / structured) or refuse the kernel")
