# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeCPUMultiDim`` — v2 orchestrator for the K-dim masked
tile-op vectorization track.

The orchestrator threads the locked single-knob configuration through
the four prep / emit passes (T3..T5) and runs ``expand_library_nodes()``
at the tail to lower the tile-op lib nodes to their ``pure`` expansion.
The post-orchestrator audit asserts that no per-lane scalar leaked into
the SDFG (lib-node emission must carry lane offsets implicitly).

Refer to the v2 plan for the locked knobs:

* ``backend = VectorizeCPUMultiDim``.
* ``target_isa = "AVX512" | "SCALAR"`` (pure-only in MVP; cuTile in T9).
* ``widths`` — innermost-last, length in ``{1, 2, 3}``, powers of 2.
* ``remainder_strategy = MASKED_TAIL`` (one map, mask covers the tail).
* ``branch_normalization = True``, ``use_fp_factor = False`` (v2 only
  consumes ``merge``-form output; the path is reserved for the
  post-MVP :class:`TileMerge` slice).

Refuses every other combination with ``NotImplementedError`` so the
caller is pointed at the supported config.
"""
from typing import Literal, Optional, Tuple

import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern,
)
from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask,
)
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import (
    PromoteNSDFGBodyToTiles,
)
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths,
)
from dace.transformation.passes.vectorization.utils.name_schemes import (
    assert_no_laneid_in_tile_path,
)

_VALID_ISAS = ("AVX512", "SCALAR")


def _is_power_of_two(n: int) -> bool:
    """Return True iff ``n`` is a strictly-positive power of 2.

    :param n: Integer to test.
    :returns: ``True`` iff ``n & (n - 1) == 0`` and ``n > 0``.
    """
    return n > 0 and (n & (n - 1)) == 0


@properties.make_properties
class VectorizeCPUMultiDim(ppl.Pipeline):
    """Drive the v2 K-dim masked tile-op pipeline.

    The constructor validates the locked knob row (raises
    ``NotImplementedError`` on every unsupported combo) and assembles
    the prep + emit passes into a :class:`ppl.Pipeline`. Library-node
    expansion runs after the pipeline finishes; the audit fires last.
    """

    CATEGORY: str = "Vectorization"

    def __init__(self,
                 widths: Tuple[int, ...],
                 target_isa: Literal["AVX512", "SCALAR"] = "AVX512",
                 num_cores: int = 1):
        """Build the orchestrator.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param target_isa: Locked to ``"AVX512"`` or ``"SCALAR"`` for
            MVP. Both share the same ``pure`` expansion until T9 lands
            cuTile / non-pure backends.
        :param num_cores: Reserved for future per-core tiling; currently
            unused.
        :raises NotImplementedError: On any disallowed combination.
        """
        if not (1 <= len(widths) <= 3):
            raise NotImplementedError(
                f"VectorizeCPUMultiDim: K={len(widths)} not in {{1, 2, 3}}; "
                f"got widths={widths!r}"
            )
        if target_isa not in _VALID_ISAS:
            raise NotImplementedError(
                f"VectorizeCPUMultiDim: target_isa {target_isa!r} not in {_VALID_ISAS}; "
                f"cuTile lowering lands in T9."
            )
        if not all(_is_power_of_two(w) for w in widths):
            raise NotImplementedError(
                f"VectorizeCPUMultiDim: every width must be a power of 2; got {widths!r}"
            )
        if target_isa == "AVX512" and widths[-1] % 8 != 0:
            raise NotImplementedError(
                f"VectorizeCPUMultiDim: AVX-512 requires widths[-1] % 8 == 0; got "
                f"widths[-1]={widths[-1]}"
            )

        widths_t = tuple(widths)
        passes = [
            # Fold ``A -> A_slice (length-1) -> tasklet`` so binop tasklets read
            # the original tile-dependent array access directly, not a length-1
            # scalar slice (which ``EmitTileOps`` would mis-classify as a Scalar
            # broadcast). Mirrors the run-at-front placement on the 1D path.
            CleanAccessNodeToScalarSliceToTaskletPattern(),
            MarkTileDims(widths=widths_t),
            GenerateTileIterationMask(widths=widths_t),
            StrideMapByTileWidths(widths=widths_t),
            # Tile a flat body-NSDFG (vbor-style scalar chain) in place so
            # EmitTileOps can skip it; EmitTileOps still raises for un-handled
            # NSDFG bodies (the carried-dep LoopRegion cases stay clean skips).
            PromoteNSDFGBodyToTiles(widths=widths_t),
            EmitTileOps(widths=widths_t),
        ]
        super().__init__(passes)
        self._widths = widths_t
        self._target_isa = target_isa
        self._num_cores = num_cores

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """Run the prep + emit pipeline, then expand lib nodes + audit.

        For K >= 2 the K-dim tile is taken over the last ``K`` params of
        one innermost map. A realistic K-dim kernel
        (``@dace.program`` -> ``LoopToMap`` -> ``simplify``) instead leaves
        perfectly-nested *single*-param maps (``for i: for j: ...`` becomes
        an ``i`` map wrapping a ``j`` map), on which ``MarkTileDims`` would
        raise ("only 1 param < K") and the kernel would silently degrade to
        a 1D inner-dim tile. Collapse those nested maps into one K-param map
        first so the tile genuinely spans K dims and lowers to K-fold scalar
        loops. ``MapCollapse`` only fires on a perfectly-nested same-schedule
        pair (its own ``can_be_applied`` guard), so a sequential/carried-dep
        inner loop (which stays a ``LoopRegion``, not a map) is left alone.
        K == 1 never collapses (a 2D kernel under ``widths=(W,)`` must keep
        its outer map separate and tile only the inner dim).

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Carry-in from any enclosing pipeline.
        :returns: Whatever the inner pipeline returned (count of rewrites).
        """
        if len(self._widths) >= 2:
            from dace.transformation.dataflow import MapCollapse
            sdfg.apply_transformations_repeated(MapCollapse(), permissive=False, validate=False)
        result = super().apply_pass(sdfg, pipeline_results)
        sdfg.expand_library_nodes()
        assert_no_laneid_in_tile_path(sdfg)
        return result
