# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MarkTileDims`` — validation-only pass that picks K innermost
parameters per inner map and constructs a :class:`TileDimSpec` per
candidate.

Runs as the first per-map analysis step in the v2 orchestrator. Loud
failure on any inner map that cannot be K-dim tiled (degenerate dim,
step != 1, fewer than K params, etc.) so the orchestrator's error
points at the map that needs attention rather than downstream
masked-tail emission failing in a confusing way.
"""
from typing import Dict, Optional, Tuple

import dace
from dace import properties
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


@properties.make_properties
class MarkTileDims(ppl.Pass):
    """Validate and record the K innermost tiled dims per inner map.

    For every innermost map in the SDFG: take its last ``K`` parameters
    (where ``K = len(widths)``), check step == 1 and trip > 1 on each,
    then build a :class:`TileDimSpec` recording the iter-vars, widths
    and original exclusive upper bounds. The pass returns the
    ``{MapEntry: TileDimSpec}`` map for downstream passes that need it.

    No SDFG mutation. Failures raise loudly (``NotImplementedError``)
    with the offending map identified by its label.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    skip_ineligible = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, ineligible maps are silently dropped from the result instead of "
        "raising. Default is loud failure so the orchestrator surfaces the problem.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, ), skip_ineligible: bool = False):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param skip_ineligible: When True, soft-skip ineligible inner
            maps; default raises ``NotImplementedError``.
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"MarkTileDims: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)
        self.skip_ineligible = skip_ineligible

    def modifies(self) -> ppl.Modifies:
        """Pass is read-only.

        :returns: ``ppl.Modifies.Nothing``.
        """
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Validation pass; no re-apply is needed.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _classify_one(self, map_entry: MapEntry) -> Optional[TileDimSpec]:
        """Build a :class:`TileDimSpec` for ``map_entry`` if eligible.

        :param map_entry: The candidate inner map entry.
        :returns: The spec when the K innermost params each have
            step == 1; ``None`` otherwise.
        :raises NotImplementedError: When ``skip_ineligible`` is
            ``False`` and the map is ineligible.
        """
        # ``__tile_k1_tail`` tail maps are pinned to K=1 widths=(1,)
        # regardless of the orchestrator-level widths; the postamble is
        # always a single-lane scalar-tile remainder over the innermost
        # iter var only.
        widths = (1, ) if map_entry.map.label.endswith(TILE_K1_TAIL_MARKER) else tuple(self.widths)
        K = len(widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        if len(params) < K:
            return self._fail_or_skip(f"map {map_entry.label!r} has only {len(params)} params (< K={K})")
        iter_vars = tuple(params[-K:])
        slice_ranges = ranges[-K:]
        global_ubs = []
        # Unified "no-mask interior + w-mask remainder" model: every tiled dim
        # gets a spec regardless of trip, and the trip is NOT required to be
        # >= W. A ``trip < W`` dim lowers to a single w-mask remainder tile
        # (mask ``l < trip``); ``trip == 0`` is a correct all-false-mask no-op.
        # SplitMapForTileRemainder peels each non-divisible dim into a (possibly
        # empty) ``__tile_main`` interior (mask-free) + a masked remainder, and
        # GenerateTileIterationMask masks every non-interior region -- so a
        # short, symbolic, or per-iteration-varying (e.g. wavefront) trip is
        # handled by masking, or under ``scalar_postamble`` by the scalar
        # remainder loop. There is no ``trip >= W`` precondition and no runtime
        # trap: a too-small trip just executes fewer active lanes.
        for (lb, ub, step), iv in zip(slice_ranges, iter_vars):
            if step != 1 and str(step) != "1":
                return self._fail_or_skip(
                    f"map {map_entry.label!r} dim {iv!r} has step {step!r}; v2 requires step == 1")
            global_ubs.append(str(ub + 1))
        return TileDimSpec(
            iter_vars=iter_vars,
            widths=widths,
            global_ubs=tuple(global_ubs),
        )

    def _fail_or_skip(self, msg: str) -> Optional[TileDimSpec]:
        """Either raise or return ``None`` based on ``skip_ineligible``.

        :param msg: Diagnostic message included in the raised error.
        :returns: ``None`` when ``skip_ineligible`` is True.
        :raises NotImplementedError: When ``skip_ineligible`` is False.
        """
        if self.skip_ineligible:
            return None
        raise NotImplementedError(f"MarkTileDims: {msg}")

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[Dict[MapEntry, TileDimSpec]]:
        """Walk every innermost map and record the K-dim spec.

        :param sdfg: SDFG to analyze.
        :param _: Pipeline-results placeholder (unused).
        :returns: ``{MapEntry: TileDimSpec}`` for every eligible inner
            map; ``None`` when no candidate matched.
        :raises NotImplementedError: When an inner map is ineligible
            and ``skip_ineligible`` is False.
        """
        specs: Dict[MapEntry, TileDimSpec] = {}
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: stays scalar
                continue
            spec = self._classify_one(n)
            if spec is not None:
                specs[n] = spec
        assert_invariant(no_memlet_dim_mismatch(sdfg), "MarkTileDims",
                         "memlet subset and other_subset have matching dimensionality")
        return specs or None
