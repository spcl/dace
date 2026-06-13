# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitMapForTileRemainder`` — split a K-dim tile map into a divisible
interior region plus masked boundary remainders (the ``masked_tail`` strategy).

The K innermost (tiled) dims of every eligible inner map are peeled into one
fully-tiled *interior* plus K masked *slabs* (the standard ``K``-slab
loop-peeling decomposition, not the ``2**K`` Cartesian corner split). For each
peeled dim ``d`` the interior's dim is tightened to the largest multiple-of-``W``
boundary; the slab for dim ``d`` covers dims ``< d`` at their interior extent,
dim ``d`` at the trailing tail, and dims ``> d`` at their full extent. The slabs
are pairwise disjoint and together with the interior tile the whole space, so a
tile lands in a slab iff *any* tiled dim is partial (the "interior + everything
else" shape) — ``K + 1`` regions, not ``2**K``. The interior — every tiled dim
fully within bounds — is marked with the ``__tile_main`` label suffix so
:class:`GenerateTileIterationMask` skips its mask and the descent / emit lower
it with ``has_mask=False`` (the perf fast path). Every slab keeps its mask (its
tile touches the trailing tail on at least one dim).

K=2 layout (aligned bounds ``A_d = floor(N_d / W) * W``)::

            dim1 ->
           [0 : A1)        [A1 : N1)
          +---------------+--------+
    [0:A0)|   INTERIOR    | slab 1 |   slab_1 = [0:A0, A1:N1]
     dim0 | (mask-free)   |        |   (dim1 tail, dim0 INTERIOR height)
          +---------------+--------+
   [A0:N0)|     slab 0  (full dim1 width)  |   slab_0 = [A0:N0, 0:N1]
          +-------------------------------+   (dim0 tail, dim1 FULL width)

Each slab claims the tail of exactly one dim: ``slab_0`` takes dim0's tail
across the *full* dim1 extent (so it absorbs the both-dims-partial corner),
``slab_1`` takes dim1's tail only over dim0's *interior* height (the corner is
already slab_0's). That corner absorption is why peeling needs only ``K`` slabs
rather than the ``2**K - 1`` cells a Cartesian corner split would produce.

This runs **before** :class:`MarkTileDims` so the newly replicated boundary maps
are tagged and tiled like the original; it operates on step-1 maps (before
:class:`StrideMapByTileWidths`). A dim provably divisible by ``W`` (or provably
shorter than ``W``) is not split — so a fully-divisible map produces just the
marked interior (mask-free) with no remainder.
"""
from typing import Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import replicate_scope
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)

# Label suffix marking the all-main (fully-in-bounds, divisible) interior region
# a tile-remainder split produces. GenerateTileIterationMask recognises it and
# skips the mask so the interior lowers with has_mask=False.
TILE_MAIN_MARKER = "__tile_main"

# Label suffix marking a boundary region that runs as a plain step-1 scalar loop
# (the ``scalar_postamble`` strategy's tail). Every tile prep pass
# (MarkTileDims / GenerateTileIterationMask / StrideMapByTileWidths /
# PromoteNSDFGBodyToTiles / EmitTileOps) skips a map carrying this suffix, so the
# tail keeps its original (branch-normalized) scalar body and is neither tiled,
# strided, nor masked.
SCALAR_TAIL_MARKER = "__scalar_tail"

# Label suffix marking a boundary region that flows through the tile-op pipeline
# at K=1 ``widths=(1,)`` — a single-lane "scalar tile" remainder. Every tile prep
# pass treats this suffix as a tile-main region pinned at K=1 width=1: the
# stride is 1 (no W-stride), no iteration mask, but the body is rewritten to
# tile ops (TileBinop, TileLoad, TileStore at one lane). Enables uniform
# emission for the remainder when the user opts in via ``scalar_remainder_emit
# ="tile"`` on the orchestrator.
TILE_K1_TAIL_MARKER = "__tile_k1_tail"


@properties.make_properties
class SplitMapForTileRemainder(ppl.Pass):
    """Split each K-dim tile map into a divisible interior + masked boundary
    regions, marking the interior ``__tile_main``. See module docstring."""

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    tail_mode = properties.Property(
        dtype=str,
        allow_none=False,
        default="masked",
        desc="Boundary-region handling: 'masked' (W-strided masked slabs, the "
        "masked_tail strategy), 'scalar' (step-1 scalar-loop slabs marked "
        "__scalar_tail, the scalar_postamble strategy), or 'tile_k1' (step-1 "
        "tile-op slabs at widths=(1,) marked __tile_k1_tail, the K=0/single-lane "
        "tile-op variant of the scalar_postamble strategy).",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, ), tail_mode: str = "masked"):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param tail_mode: ``"masked"`` (W-strided masked slabs), ``"scalar"``
            (step-1 scalar-loop slabs marked :data:`SCALAR_TAIL_MARKER`), or
            ``"tile_k1"`` (step-1 tile-op slabs at ``widths=(1,)`` marked
            :data:`TILE_K1_TAIL_MARKER` — the K=0 single-lane tile-op variant
            of the ``scalar_postamble`` strategy).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}`` or
            ``tail_mode`` is not one of the supported values.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"SplitMapForTileRemainder: widths length {len(widths)} not in {{1, 2, 3}}")
        if tail_mode not in ("masked", "scalar", "tile_k1"):
            raise ValueError(f"SplitMapForTileRemainder: tail_mode {tail_mode!r} not in "
                             f"{{'masked', 'scalar', 'tile_k1'}}")
        self.widths = list(widths)
        self.tail_mode = tail_mode

    def modifies(self) -> ppl.Modifies:
        """Pass replicates scopes and retightens ranges.

        :returns: ``ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Scopes``.
        """
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Runs once.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _provably_divisible_or_short(self, lb, ub, W: int) -> bool:
        """Whether the dim ``[lb:ub]`` needs no split for tile width ``W``.

        :param lb: Inclusive lower bound.
        :param ub: Inclusive upper bound.
        :param W: Tile width.
        :returns: ``True`` if the trip is provably a multiple of ``W`` (whole
            tiles only) or provably shorter than ``W`` (no whole tile — the
            single masked tile suffices, splitting would only add noise).
        """
        trip = symbolic.simplify(ub - lb + 1)
        try:
            if bool((trip % W).simplify() == 0):
                return True
            if bool((trip < W).simplify()):
                return True
        except Exception:  # noqa: BLE001 - non-decidable symbolic trip -> split
            pass
        return False

    def _split(self, state: dace.SDFGState, map_entry: MapEntry, K: int) -> bool:
        """Peel ``map_entry``'s K innermost dims into an interior + K slabs.

        Uses the standard ``K``-slab loop-peeling decomposition (not the
        ``2**K`` Cartesian corner split): the box is split into one fully-tiled
        interior plus, for each tiled dim ``d`` that needs peeling, one masked
        *slab* in which dims ``< d`` are at their interior extent, dim ``d`` is
        the trailing tail, and dims ``> d`` span their full extent. The slabs
        are pairwise disjoint and together with the interior tile the whole
        space — so a tile is in a slab iff *any* tiled dim is partial (the
        "interior + everything else" shape), with only ``K + 1`` regions
        instead of ``2**K``.

        Concretely the interior map (``map_entry`` itself, tightened on every
        dim) is marked ``__tile_main`` (mask-free); each slab is a fresh
        ``replicate_scope`` copy of the interior-so-far with dim ``d`` set to
        its tail (and dims ``> d`` still full, because they have not been
        tightened yet at that point).

        :param state: The state holding the map.
        :param map_entry: The innermost map entry to peel (becomes the interior).
        :param K: Number of tiled (innermost) dims.
        :returns: ``True`` if the interior was marked (always, when the map has
            >= K dims); ``False`` if the map is too small.
        """
        ranges = list(map_entry.map.range.ranges)
        if len(ranges) < K:
            return False
        tiled_dims = list(range(len(ranges) - K, len(ranges)))
        for d, W in zip(tiled_dims, self.widths):
            lb, ub, step = map_entry.map.range[d]
            if self._provably_divisible_or_short(lb, ub, W):
                continue
            trip = symbolic.simplify(ub - lb + 1)
            # ``int_floor`` (not ``//``) so the C++ codegen emits integer division
            # (mirrors the 1D SplitMapForVectorRemainder rationale).
            main_end = lb + (symbolic.int_floor(trip, W) * W) - 1
            # Slab for dim ``d``: a copy of the interior-so-far (dims < d already
            # tightened to interior, dim d + later dims still full) with dim d
            # set to the tail. Replicate before tightening dim d on the interior.
            scope_view = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
            slab = replicate_scope(state.sdfg, state, scope_view)
            slab.entry.map.range[d] = (main_end + 1, ub, step)
            map_entry.map.range[d] = (lb, main_end, step)
            # ``scalar`` tail mode: mark the slab so every tile prep pass skips
            # it -> it stays a plain step-1 scalar loop running the original body.
            if self.tail_mode == "scalar" and not slab.entry.map.label.endswith(SCALAR_TAIL_MARKER):
                slab.entry.map.label = slab.entry.map.label + SCALAR_TAIL_MARKER
            # ``tile_k1`` tail mode: mark the slab so every tile prep pass
            # treats it as a tile-main region pinned at K=1 widths=(1,) — step
            # 1, no mask, body lowered to single-lane tile ops.
            elif self.tail_mode == "tile_k1" and not slab.entry.map.label.endswith(TILE_K1_TAIL_MARKER):
                slab.entry.map.label = slab.entry.map.label + TILE_K1_TAIL_MARKER
        # The interior is fully tiled on every dim (skipped dims were divisible;
        # peeled dims are tightened to a multiple of W) — mark it mask-free.
        if not map_entry.map.label.endswith(TILE_MAIN_MARKER):
            map_entry.map.label = map_entry.map.label + TILE_MAIN_MARKER
        return True

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Split every eligible innermost K-dim map into interior + remainders.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of maps split, or ``None`` if none.
        """
        K = len(self.widths)
        applied = 0
        # Snapshot up front: splitting mutates the state graph and we must not
        # re-split a freshly replicated remainder map.
        eligible = [(n, g) for n, g in sdfg.all_nodes_recursive()
                    if isinstance(n, MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)
                    and len(n.map.params) >= K and not n.map.label.endswith(TILE_MAIN_MARKER)
                    and not n.map.label.endswith(SCALAR_TAIL_MARKER) and not n.map.label.endswith(TILE_K1_TAIL_MARKER)]
        for n, g in eligible:
            if self._split(g, n, K):
                applied += 1
        if applied:
            # ``replicate_scope`` deep-copies any body NestedSDFG without
            # registering the clone in the SDFG's ``cfg_list``; rebuild it so
            # later passes (and ``expand_library_nodes``) can resolve the new
            # nested CFGs.
            sdfg.reset_cfg_list()
        assert_invariant(no_memlet_dim_mismatch(sdfg), "SplitMapForTileRemainder",
                         "memlet subset and other_subset have matching dimensionality")
        return applied or None
