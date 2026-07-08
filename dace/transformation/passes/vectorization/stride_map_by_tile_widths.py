# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StrideMapByTileWidths`` — set inner map step to the per-dim tile width on
every K-dim eligible inner map.

Rewrites ``map.range`` in place so the K innermost dims step by ``widths[k]``
(one tile per iteration). Masked iteration handles partial tiles at the trip
boundary; no main + remainder split.
"""
from typing import Dict, Optional, Tuple

import dace
from dace import properties, subsets, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch,
                                                                            tile_main_map_step_is_widths)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


@properties.make_properties
class StrideMapByTileWidths(ppl.Pass):
    """Stride every K-dim eligible inner map's innermost dims by ``widths``.

    Walks each inner map flagged by :class:`MarkTileDims`, replaces the step on
    the K innermost dims with the matching ``widths`` entry. Outer dims keep
    their original step.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"StrideMapByTileWidths: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Pass rewrites map ranges.

        :returns: ``ppl.Modifies.Scopes``.
        """
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent — runs once.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _stride_one(self, map_entry: MapEntry) -> bool:
        """Stride the K innermost dims of ``map_entry`` by ``widths``.

        :param map_entry: Inner map to rewrite.
        :returns: ``True`` if ``map.range`` was rewritten; ``False`` if
            the map already steps by ``widths`` (idempotent no-op).
        """
        # ``__tile_k1_tail`` maps = K=1 widths=(1,): stride stays 1 (postamble
        # is a per-element single-lane tile-op loop).
        widths = (1, ) if map_entry.map.label.endswith(TILE_K1_TAIL_MARKER) else tuple(self.widths)
        K = len(widths)
        ranges = list(map_entry.map.range.ranges)
        if len(ranges) < K:
            return False
        prefix = ranges[:-K]
        new_inner: list = []
        rewritten = False
        for (lb, ub, step), w in zip(ranges[-K:], widths):
            if step == w or str(step) == str(w):
                new_inner.append((lb, ub, step))
                continue
            new_inner.append((lb, ub, symbolic.SymExpr(w)))
            rewritten = True
        if not rewritten:
            return False
        map_entry.map.range = subsets.Range(prefix + new_inner)
        return True

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Optional[Dict]) -> Optional[int]:
        """Walk every innermost map; stride those that MarkTileDims tagged.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: When the orchestrator ran
            :class:`MarkTileDims` earlier, the spec dict is fetched
            from here under the key ``"MarkTileDims"``; otherwise every
            innermost map with enough params is treated as eligible.
        :returns: Number of maps rewritten, or ``None`` if none.
        """
        specs: Optional[Dict[MapEntry, TileDimSpec]] = None
        if pipeline_results and "MarkTileDims" in pipeline_results:
            specs = pipeline_results["MarkTileDims"]
        rewritten = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, dace.SDFGState):
                continue
            if not is_vectorizable_map(g, n):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: keep step 1
                continue
            if specs is not None and n not in specs:
                continue
            map_widths = (1, ) if n.map.label.endswith(TILE_K1_TAIL_MARKER) else tuple(self.widths)
            if len(n.map.params) < len(map_widths):
                continue
            if self._stride_one(n):
                rewritten += 1
        K = len(self.widths)
        assert_invariant(no_memlet_dim_mismatch(sdfg), "StrideMapByTileWidths",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(tile_main_map_step_is_widths(sdfg, K, tuple(self.widths)), "StrideMapByTileWidths",
                         "TILE_MAIN map's last-K dim steps equal widths")
        return rewritten or None
