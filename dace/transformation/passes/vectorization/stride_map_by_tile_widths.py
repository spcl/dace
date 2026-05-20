# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StrideMapByTileWidths`` — set the inner map step to the per-dim
tile width on every K-dim eligible inner map.

The pass rewrites ``map.range`` in place so the K innermost dims step
by ``widths[k]`` (each map iteration covers one tile worth of work).
The masked iteration handles partial tiles at the trip boundary; no
main + remainder split.
"""
from typing import Dict, Optional, Tuple

import dace
from dace import properties, subsets, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


@properties.make_properties
class StrideMapByTileWidths(ppl.Pass):
    """Stride every K-dim eligible inner map's innermost dims by ``widths``.

    The pass walks each inner map flagged by :class:`MarkTileDims`,
    replaces the step on the K innermost dims with the corresponding
    entry of ``widths``, and otherwise leaves the map untouched. Outer
    dims keep their original step.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8,)):
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
        K = len(self.widths)
        ranges = list(map_entry.map.range.ranges)
        if len(ranges) < K:
            return False
        prefix = ranges[:-K]
        new_inner: list = []
        rewritten = False
        for (lb, ub, step), w in zip(ranges[-K:], self.widths):
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
            if not is_innermost_map(g, n):
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < len(self.widths):
                continue
            if self._stride_one(n):
                rewritten += 1
        return rewritten or None
