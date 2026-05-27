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
* ``target_isa = "AUTO" | "AVX512" | "AVX2" | "ARM_SVE" | "ARM_NEON" |
  "SCALAR"`` (K=1 ISA backend; ``"AUTO"`` detects the host ISA at expansion).
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
    CleanAccessNodeToScalarSliceToTaskletPattern, )
from dace.transformation.passes.vectorization.emit_tile_ops import EmitTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask, )
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.promote_nsdfg_body_to_tiles import (
    PromoteNSDFGBodyToTiles, )
from dace.transformation.passes.vectorization.same_write_set_if_else_to_merge_cfg import (
    SameWriteSetIfElseToMergeCFG, )
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    PowerOperatorExpansion,
    RemoveFPTypeCasts,
    RemoveIntTypeCasts,
    RemoveMathCall,
)
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.remove_redundant_assignment_tasklets import RemoveRedundantAssignmentTasklets
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths, )
from dace.transformation.passes.vectorization.utils.name_schemes import (
    assert_no_laneid_in_tile_path, )

# "AUTO" resolves to the host's best ISA at expansion time (see
# dace.libraries.tileops._dispatch.detect_host_isa); the others pin one backend.
_VALID_ISAS = ("AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR")


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
                 target_isa: Literal["AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR"] = "AUTO",
                 num_cores: int = 1,
                 remainder_strategy: Literal["full_mask", "masked_tail", "scalar_postamble"] = "full_mask",
                 branch_mode: Literal["merge", "fp_factor"] = "merge"):
        """Build the orchestrator.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :param target_isa: K=1 backend to lower tile ops to. ``"AUTO"``
            (default) resolves to the host's best ISA at expansion
            (``dace.libraries.tileops._dispatch.detect_host_isa``);
            ``"AVX512"`` / ``"AVX2"`` / ``"ARM_SVE"`` / ``"ARM_NEON"`` pin one;
            ``"SCALAR"`` is the portable reference. K>=2 always uses ``pure``.
        :param num_cores: Reserved for future per-core tiling; currently
            unused.
        :param remainder_strategy: Tile remainder handling (all implemented).
            ``"full_mask"`` (default) is a single W-strided map with a
            ``_tile_iter_mask`` on every tile (the masked expansions
            short-circuit OOB lanes, so it is already OOB-safe).
            ``"masked_tail"`` splits the map into a provably-divisible interior
            (``has_mask=False`` fast path) plus a masked boundary remainder.
            ``"scalar_postamble"`` (K=1 only) splits into the divisible interior
            plus a step-1 sequential scalar tail. ``scalar_postamble`` and
            ``branch_mode="fp_factor"`` are K=1-only; both raise at K>=2.
        :param branch_mode: Branch lowering. ``"merge"`` (default) lowers a
            same-write-set if/else to a per-lane :class:`TileMerge` select;
            ``"fp_factor"`` (K=1 only, requires ``scalar_postamble``) lowers it
            to ``c*x + (1-c)*y`` tile-binop arithmetic.
        :raises NotImplementedError: On any disallowed combination (e.g. a
            K=1-only knob at K>=2, or fp_factor with a masked remainder).
        """
        if not (1 <= len(widths) <= 3):
            raise NotImplementedError(f"VectorizeCPUMultiDim: K={len(widths)} not in {{1, 2, 3}}; "
                                      f"got widths={widths!r}")
        if target_isa not in _VALID_ISAS:
            raise NotImplementedError(f"VectorizeCPUMultiDim: target_isa {target_isa!r} not in {_VALID_ISAS}; "
                                      f"cuTile lowering lands in T9.")
        if not all(_is_power_of_two(w) for w in widths):
            raise NotImplementedError(f"VectorizeCPUMultiDim: every width must be a power of 2; got {widths!r}")
        if target_isa == "AVX512" and widths[-1] % 8 != 0:
            raise NotImplementedError(f"VectorizeCPUMultiDim: AVX-512 requires widths[-1] % 8 == 0; got "
                                      f"widths[-1]={widths[-1]}")
        if remainder_strategy not in ("full_mask", "masked_tail", "scalar_postamble"):
            raise NotImplementedError(f"VectorizeCPUMultiDim: remainder_strategy {remainder_strategy!r} not in "
                                      f"{{'full_mask', 'masked_tail', 'scalar_postamble'}}")
        if branch_mode not in ("merge", "fp_factor"):
            raise NotImplementedError(f"VectorizeCPUMultiDim: branch_mode {branch_mode!r} not in "
                                      f"{{'merge', 'fp_factor'}}")
        # K-dependent knob support. K=1 covers every legacy knob variant (so the
        # 1D path can be dropped); K>=2 is the pure-lowered tile path and supports
        # ONLY the merge branch with the full_mask / masked_tail remainder. A
        # K=1-only knob (fp_factor / scalar_postamble) on a K>=2 tile raises
        # loudly rather than silently degrading.
        if len(widths) >= 2:
            if branch_mode != "merge":
                raise NotImplementedError(f"VectorizeCPUMultiDim: K={len(widths)} supports only "
                                          f"branch_mode='merge' (fp_factor is K=1-only); got {branch_mode!r}")
            if remainder_strategy not in ("full_mask", "masked_tail"):
                raise NotImplementedError(f"VectorizeCPUMultiDim: K={len(widths)} supports only "
                                          f"remainder full_mask/masked_tail (scalar_postamble is K=1-only); "
                                          f"got {remainder_strategy!r}")
        elif branch_mode == "fp_factor" and remainder_strategy != "scalar_postamble":
            # K=1: fp_factor (c*x + (1-c)*y) can't combine with an iteration mask
            # cleanly (the legacy plan rule); it pairs with the scalar postamble.
            raise NotImplementedError("VectorizeCPUMultiDim: branch_mode='fp_factor' requires "
                                      "remainder_strategy='scalar_postamble' (fp-factor is incompatible "
                                      "with a masked remainder)")

        widths_t = tuple(widths)
        # Fold ``A -> A_slice (length-1) -> tasklet`` so binop tasklets read the
        # original tile-dependent array access directly, not a length-1 scalar
        # slice (which ``EmitTileOps`` would mis-classify as a Scalar broadcast).
        # Mirrors the run-at-front placement on the 1D path.
        passes = [CleanAccessNodeToScalarSliceToTaskletPattern()]
        if branch_mode == "fp_factor":
            # FP-factor branch lowering (the legacy front): collapse a
            # same-write-set if/else to ``a = c*x + (1-c)*y`` arithmetic,
            # fold the condition into a tasklet, then split the multi-op RHS
            # into single-op binop tasklets so ``EmitTileOps`` lowers each to a
            # ``TileBinop`` (no merge/TileMerge). Pairs with scalar_postamble.
            passes += [
                EliminateBranches(),
                LowerInterstateConditionalAssignmentsToTasklets(),
            ]
        else:
            # Merge branch lowering: rewrite a same-write-set ``if/else`` into
            # compute-then / compute-else / apply-merge dataflow states carrying
            # ``merge(c, t, e)`` tasklets. Flattens the ConditionalBlock so
            # PromoteNSDFGBodyToTiles can descend; the merge tasklets lower to a
            # per-lane TileMerge select (the K-dim analogue of the 1D
            # ``vector_select`` blend), gated by the tile map's iteration mask.
            passes += [SameWriteSetIfElseToMergeCFG()]
        # Full prep, run BEFORE tiling exactly as the legacy 1D pipeline
        # (vectorize_cpu.py) does, so the tile path handles the same kernels:
        #   * RemoveEmptyStates / RemoveRedundantAssignmentTasklets — clean up
        #     the branch-lowering output. (The live ``RemoveRedundantAssignment
        #     Tasklets`` is used, NOT the dead vectorization-local
        #     ``RemoveRedundantAssignments``, whose ``depends_on``
        #     EliminateBranches trips the Pipeline's reapply machinery.)
        #   * RemoveFP/IntTypeCasts — strip ``dace.floatNN``/``intNN`` casts (the
        #     tile binop re-promotes operands to the output dtype).
        #   * PowerOperatorExpansion — ``x**2`` -> ``x*x``; ``x**c`` ->
        #     ``exp(c*log(x))`` (reusing TileUnop exp/log + TileBinop mul).
        #   * SplitTasklets — one op per tasklet (also splits the expanded power
        #     / fp_factor merge arithmetic) so the tile emitter can classify each.
        #   * RemoveMathCall — drop the ``math.`` prefix the power expansion emits
        #     so ``math.exp``/``math.log`` match TileUnop's ``exp``/``log``.
        # (``MapCollapse`` runs in ``apply_pass``. ``LoopToMap``, ``InlineSDFGs``
        # and ``InsertAssignTaskletsAtMapBoundary`` are intentionally NOT run:
        # InlineSDFGs would flatten the body NSDFGs that ``PromoteNSDFGBodyToTiles``
        # descends into; InsertAssignTaskletsAtMapBoundary and LoopToMap perturb
        # the gather / strided staging edges the tile descent classifies — e.g.
        # LoopToMap rewrites the ``a[i]=b[idx[i]]+e[i]`` gather so the tile body
        # loses its per-tile offset, duplicating tile 0 into every tile.)
        passes += [
            RemoveRedundantAssignmentTasklets(),
            RemoveFPTypeCasts(),
            RemoveIntTypeCasts(),
            PowerOperatorExpansion(),
            SplitTasklets(),
            RemoveMathCall(),
            # Clean up empty states left by the branch lowering + body rewrites
            # above, so the tiling passes see a tidy CFG. (A ``ppl.Pipeline``
            # forbids duplicate pass types, so this single end-of-prep cleanup
            # covers both the branch-front and the AST-rewrite output.)
            RemoveEmptyStates(),
        ]
        if remainder_strategy in ("masked_tail", "scalar_postamble"):
            # Split each K-dim tile map into a provably-divisible interior
            # (marked ``__tile_main`` -> GenerateTileIterationMask skips its
            # mask -> the descent / emit lower it with has_mask=False, the perf
            # fast path) plus boundary remainder regions. Runs before
            # MarkTileDims so the replicated boundary maps are tagged + tiled.
            # ``masked_tail`` keeps the boundary regions as W-strided masked
            # slabs; ``scalar_postamble`` marks them ``__scalar_tail`` so they
            # stay plain step-1 scalar loops (the legacy postamble shape) that
            # every tile prep pass skips.
            from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (
                SplitMapForTileRemainder, )
            tail_mode = "scalar" if remainder_strategy == "scalar_postamble" else "masked"
            passes.append(SplitMapForTileRemainder(widths=widths_t, tail_mode=tail_mode))
        passes += [
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
        self._remainder_strategy = remainder_strategy
        self._branch_mode = branch_mode

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """Run the prep + emit pipeline, then expand lib nodes + audit.

        The K-dim tile is taken over the last ``K`` params of one innermost
        map. A realistic kernel (``@dace.program`` -> ``LoopToMap`` ->
        ``simplify``) leaves perfectly-nested *single*-param maps
        (``for i: for j: ...`` becomes an ``i`` map wrapping a ``j`` map). We
        collapse those into one multi-param map first (regardless of ``K``):
        for K >= 2 the tile then genuinely spans K dims; for K == 1 the merged
        ``(i, j)`` map iterates the outer dims normally and ``MarkTileDims``
        tiles only the innermost param ``j`` (last-``K`` slice) — the same
        effective result as the un-collapsed nest, but uniform. ``MapCollapse``
        only fires on a perfectly-nested same-schedule pair (its own
        ``can_be_applied`` guard), so a sequential/carried-dep inner loop
        (which stays a ``LoopRegion``, not a map) is left alone.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Carry-in from any enclosing pipeline.
        :returns: Whatever the inner pipeline returned (count of rewrites).
        """
        from dace.transformation.dataflow import MapCollapse, WCRToAugAssign
        # The vectorization / tile path does NOT handle WCR: convert every
        # write-conflict-resolution memlet (a reduction ``s += a[i]``) into an
        # explicit augmented-assignment RMW tasklet (``s = s + a[i]``) so no WCR
        # is left for the tile passes (matches the SVE path's pass-0 convention).
        # Then collapse perfectly-nested single-param maps into one multi-param
        # map so a K-dim tile genuinely spans K dims. (``LoopToMap`` is NOT run:
        # it rewrites the gather staging so tiles lose their per-tile offset.)
        sdfg.apply_transformations_repeated([WCRToAugAssign, MapCollapse], permissive=False, validate=False)
        result = super().apply_pass(sdfg, pipeline_results)
        self._select_tile_implementations(sdfg)
        sdfg.expand_library_nodes()
        assert_no_laneid_in_tile_path(sdfg)
        return result

    def _select_tile_implementations(self, sdfg: dace.SDFG) -> None:
        """Stamp ``target_isa`` on every emitted tile lib node and resolve its
        concrete implementation before expansion.

        The choice depends only on ``target_isa`` (this orchestrator's) + the
        node's ``K`` — both known now — so we set ``node.implementation``
        directly (the standard DaCe model) rather than through an ``'Auto'``
        re-dispatch. ``select_tile_implementation`` returns ``'pure'`` for
        ``K >= 2`` and falls back to ``'pure'`` whenever the per-ISA expansion is
        not yet defined on the node, so this is a no-op until the real intrinsic
        expansions land.

        :param sdfg: SDFG whose tile lib nodes are resolved in place.
        """
        from dace.libraries.tileops._dispatch import select_tile_implementation
        from dace.libraries.tileops.nodes import (TileBinop, TileGather, TileLoad, TileMaskGen, TileMerge, TileReduce,
                                                  TileScatter, TileStore, TileUnop)
        tile_node_types = (TileBinop, TileGather, TileLoad, TileMaskGen, TileMerge, TileReduce, TileScatter, TileStore,
                           TileUnop)
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, tile_node_types):
                node.target_isa = self._target_isa
                node.implementation = select_tile_implementation(node)
