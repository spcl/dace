# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MarkTileDims`` — validation-only: pick K innermost params per inner map, build a
:class:`TileDimSpec` per candidate.

First per-map analysis step in the v2 orchestrator. Loud failure on any inner map that can't
be K-dim tiled (step != 1, < K params, ...) so error points at the offending map, not a
confusing downstream masked-tail failure.
"""
from typing import Dict, Optional, Tuple

import dace
from dace import properties, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_gpu_resident_map, is_vectorizable_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


@properties.make_properties
class MarkTileDims(ppl.Pass):
    """Validate + record K innermost tiled dims per inner map.

    Per innermost map: take last ``K`` params (``K = len(widths)``), check step == 1 on each,
    build a :class:`TileDimSpec` recording iter-vars, widths, original exclusive upper bounds.
    Returns ``{MapEntry: TileDimSpec}`` for downstream passes. No SDFG mutation; failures raise
    ``NotImplementedError`` naming the offending map by label.
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
    require_gpu_resident = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True (GPU device path), tile only innermost maps that execute inside a GPU "
        "kernel -- either GPU_Device-scheduled, or a Sequential map nested (through scopes and "
        "NestedSDFG boundaries) inside a GPU_Device map. A host-side innermost map is skipped, not "
        "tiled: its half2 __device__ intrinsics would not compile in host code.",
    )

    def __init__(self,
                 widths: Tuple[int, ...] = (8, ),
                 skip_ineligible: bool = False,
                 require_gpu_resident: bool = False):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param skip_ineligible: True -> soft-skip ineligible inner maps; default raises
            ``NotImplementedError``.
        :param require_gpu_resident: True -> skip any innermost map not executing inside a GPU
            kernel (see property doc). Set by GPU orchestrator so only device-resident maps are
            half2-tiled.
        :raises ValueError: If ``widths`` length not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"MarkTileDims: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)
        self.skip_ineligible = skip_ineligible
        self.require_gpu_resident = require_gpu_resident

    def modifies(self) -> ppl.Modifies:
        """Read-only pass.

        :returns: ``ppl.Modifies.Nothing``.
        """
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Validation pass; never re-applies.

        :param modified: Earlier passes' modifications (unused).
        :returns: ``False``.
        """
        return False

    def _classify_one(self, map_entry: MapEntry) -> Optional[TileDimSpec]:
        """Build a :class:`TileDimSpec` for ``map_entry`` if eligible.

        :param map_entry: The candidate inner map entry.
        :returns: Spec when the K innermost params each have step == 1; ``None`` otherwise.
        :raises NotImplementedError: When ``skip_ineligible`` is ``False`` and map ineligible.
        """
        # ``__tile_k1_tail`` maps pin K=1 widths=(1,) regardless of orchestrator
        # widths: single-lane scalar-tile remainder over the innermost iter-var only.
        widths = (1, ) if map_entry.map.label.endswith(TILE_K1_TAIL_MARKER) else tuple(self.widths)
        K = len(widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        if len(params) < K:
            return self._fail_or_skip(f"map {map_entry.label!r} has only {len(params)} params (< K={K})")
        iter_vars = tuple(params[-K:])
        slice_ranges = ranges[-K:]
        global_ubs = []
        # Unified "mask-free interior + w-mask remainder" model: every tiled dim gets a spec
        # regardless of trip; trip NOT required >= W. ``trip < W`` -> single w-mask remainder
        # tile (mask ``l < trip``); ``trip == 0`` = correct all-false no-op.
        # SplitMapForTileRemainder peels each non-divisible dim into a (possibly empty) mask-free
        # ``__tile_main`` + masked remainder; GenerateTileIterationMask masks every non-interior
        # region -> short, symbolic, or wavefront trips handled by masking (or the scalar
        # remainder loop under ``scalar_postamble``). No ``trip >= W`` precondition, no runtime
        # trap: too-small trip just runs fewer active lanes.
        for (lb, ub, step), iv, W in zip(slice_ranges, iter_vars, widths):
            if step != 1 and str(step) != "1":
                return self._fail_or_skip(
                    f"map {map_entry.label!r} dim {iv!r} has step {step!r}; v2 requires step == 1")
            # A provably-too-small dim (extent < tile width) cannot be tiled -- keep the whole map
            # scalar. NOT an error (unlike step != 1): a scalar / constant ``gmap`` or a short loop
            # simply is not a width-W vector target (widening it reads past the extent). Return
            # None to skip; SplitMapForTileRemainder refuses the same dim, so the two passes agree.
            # A symbolic extent stays tiled -- the masked remainder / assume_even runtime guard
            # handles a runtime trip < W.
            try:
                if int(symbolic.simplify(ub - lb + 1)) < W:
                    return None
            except (TypeError, ValueError):
                pass
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
        :returns: ``{MapEntry: TileDimSpec}`` per eligible inner map; ``None`` when none matched.
        :raises NotImplementedError: When an inner map is ineligible and ``skip_ineligible`` False.
        """
        specs: Dict[MapEntry, TileDimSpec] = {}
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_vectorizable_map(g, n):
                continue
            # GPU path: only tile maps running inside a GPU kernel (GPU_Device, or
            # nested under one). Host-map half2 __device__ tile ops won't compile.
            if self.require_gpu_resident and not is_gpu_resident_map(g, n):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: stays scalar
                continue
            spec = self._classify_one(n)
            if spec is not None:
                specs[n] = spec
        assert_invariant(no_memlet_dim_mismatch(sdfg), "MarkTileDims",
                         "memlet subset and other_subset have matching dimensionality")
        return specs or None
