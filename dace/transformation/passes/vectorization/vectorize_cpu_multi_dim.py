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
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.resolve_other_subset_an_edges import (
    ResolveOtherSubsetANEdges, )
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
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths, )
from dace.transformation.passes.vectorization.utils.name_schemes import (
    assert_no_laneid_in_tile_path, )

# "AUTO" resolves to the host's best ISA at expansion time (see
# dace.libraries.tileops._dispatch.detect_host_isa); the others pin one backend.
_VALID_ISAS = ("AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR")

#: Convergence cap for the per-NSDFG ``RefineNestedAccess`` re-check loop (one
#: application already refines every candidate; the cap guards against a
#: pathological non-converging fixpoint).
_MAX_REFINE_ITERS = 8


def _is_power_of_two(n: int) -> bool:
    """Return True iff ``n`` is a strictly-positive power of 2.

    :param n: Integer to test.
    :returns: ``True`` iff ``n & (n - 1) == 0`` and ``n > 0``.
    """
    return n > 0 and (n & (n - 1)) == 0


def normalize_loop_nests(sdfg: dace.SDFG) -> None:
    """Normalise loop-nest map bodies so a K-dim tile spans K genuine map dims.

    ``LoopToMap`` wraps each parallelised loop body in an NSDFG, so a nested
    ``for j: for i:`` becomes ``j-map -> NSDFG -> i-map`` — non-adjacent maps that
    ``MapCollapse`` cannot fuse. This:

    1. Inlines the loop-nesting wrapper NSDFGs (and any *single-state* leaf
       compute body) via ``InlineSDFG`` / ``InlineMultistateSDFG`` so the maps
       become adjacent. A leaf body ``InlineSDFG`` refuses — a *multi-state* body
       or one with an **inout connector** (the cloudsc ``zqlhs`` RMW chain, the
       ``vbor`` reused-scalar chain) — is left intact for the tile descent
       (:class:`PromoteNSDFGBodyToTiles`).
    2. Collapses the now-adjacent perfectly-nested single-param maps into one
       multi-param map (``MapCollapse``).
    3. Re-propagates memlets. Inlining merges the wrapper scope into its parent
       and collapse fuses two map scopes into one, both of which leave the outer
       memlets over-wide (the whole-array form the wrapper carried, not the fused
       per-iteration slice). ``propagate_memlets_sdfg`` re-tightens every scope
       edge to what the fused body actually accesses, so the tile boundary-widening
       sees the correct per-iteration granularity.

    Net effect: fewer downstream body shapes — a flattened single-state body tiles
    via :class:`EmitTileOps`; a preserved inout / multi-state body tiles via the
    descent.

    :param sdfg: SDFG to normalise in place.
    """
    from dace.sdfg.propagation import propagate_memlets_sdfg
    from dace.transformation.dataflow import MapCollapse
    from dace.transformation.interstate import InlineMultistateSDFG, InlineSDFG
    sdfg.apply_transformations_repeated([InlineSDFG, InlineMultistateSDFG], permissive=False, validate=False)
    sdfg.apply_transformations_repeated(MapCollapse, permissive=False, validate=False)
    propagate_memlets_sdfg(sdfg)


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
                 branch_mode: Literal["merge", "fp_factor"] = "merge",
                 loop_to_map_permissive: bool = False,
                 nest_map_bodies: bool = False,
                 insert_copies: bool = False,
                 fuse_overlapping_loads: bool = False):
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
        :param insert_copies: Accepted for harness parity with the legacy
            ``VectorizeCPU``. The tile path does not need explicit boundary
            copy nodes — its ``TileLoad``/``TileStore`` lib nodes already make
            the NSDFG-boundary memlets explicit at expansion time — so this
            knob is a no-op here. Keeping it on the constructor lets the
            shared harness forward the same knob to either pipeline without a
            branching wrapper.
        :param fuse_overlapping_loads: Accepted for harness parity with the
            legacy ``VectorizeCPU``. The tile path does not yet fuse
            overlapping loads (every ``TileLoad`` emits its own window); this
            knob is a no-op here, kept on the constructor for the same
            harness-parity reason as ``insert_copies``.
        :param nest_map_bodies: Emit-path selector. ``False`` (default) keeps
            the hybrid path — a flat (bare-tasklet) map body tiles via
            :class:`EmitTileOps`, and only a body that is *already* a NestedSDFG
            (a non-inlinable reused-scalar / multi-state chain) descends via
            :class:`PromoteNSDFGBodyToTiles`. ``True`` nests *every* innermost
            map body into a NestedSDFG first
            (:class:`NestInnermostMapBodyIntoNSDFG`), so the descent is the
            single emit path. Both values must produce identical numerics; the
            descent path is what a strided / gather / structured access inside a
            non-inlinable body needs regardless of this knob.
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
        # K-dependent knob support. K=1 and K>=2 both support every (branch,
        # remainder) combo: the iter_mask only gates stores (the fp_factor
        # arithmetic itself runs on every lane unchanged), and the fp_factor
        # branch lowering reduces to a chain of single-op tile binops after
        # ``SplitTasklets``, which the K-dim tile emitter classifies the same
        # way regardless of K.

        widths_t = tuple(widths)
        # ``NormalizeWCRSource`` runs first so every WCR sink is sourced by an
        # AccessNode (private scalar) — the post-vectorise reduction lowering
        # in ``EmitTileOps`` keys on this shape (``tile -> _wcr_src (Scalar)
        # -[wcr]-> sink``) and emits a ``TileReduce`` per matching edge so the
        # tile collapses to a scalar before the outer OpenMP reduction fires.
        # Idempotent — re-runs are no-ops.
        from dace.transformation.passes.normalize_wcr_source import (NormalizeWCRSource)
        # Fold ``A -> A_slice (length-1) -> tasklet`` so binop tasklets read the
        # original tile-dependent array access directly, not a length-1 scalar
        # slice (which ``EmitTileOps`` would mis-classify as a Scalar broadcast).
        # Mirrors the run-at-front placement on the 1D path.
        passes = [NormalizeWCRSource(), CleanAccessNodeToScalarSliceToTaskletPattern()]
        if branch_mode == "fp_factor":
            # FP-factor branch lowering (the legacy front): collapse a
            # same-write-set if/else to ``a = c*x + (1-c)*y`` arithmetic,
            # fold the condition into a tasklet, then split the multi-op RHS
            # into single-op binop tasklets so ``EmitTileOps`` lowers each to a
            # ``TileBinop`` (no merge/TileMerge). Pairs with scalar_postamble.
            #
            # ``permissive=True`` is REQUIRED for the vectorisation pipeline:
            # the default ``can_be_applied`` refuses any conditional whose
            # condition references a map parameter (out of caution against
            # synthetic OOB reads from unconditional execution). At
            # vectorisation time we accept that risk by design — the
            # downstream descent EXPECTS every conditional to be lifted into
            # straight-line arithmetic, and the kernels we tile (TSVC, cloudsc,
            # icon) are written with map-param-indexed access patterns that
            # are bounded by the surrounding map range. Without this flag
            # every map-param-conditional kernel (``s273``, the boolean-op
            # branches, anything with ``if a[i] < 0`` etc.) is left with a
            # ``ConditionalBlock`` that ``PromoteNSDFGBodyToTiles`` now
            # refuses loudly.
            _eb = EliminateBranches()
            _eb.permissive = True
            passes += [
                _eb,
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
        # (``WCRToAugAssign``, ``LoopToMap``, ``RefineNestedAccess`` and
        # ``MapCollapse`` run in ``apply_pass``. ``InlineSDFGs`` and
        # ``InsertAssignTaskletsAtMapBoundary`` are intentionally NOT run:
        # InlineSDFGs would flatten the body NSDFGs ``PromoteNSDFGBodyToTiles``
        # descends into; InsertAssignTaskletsAtMapBoundary perturbs the gather /
        # strided staging edges.)
        passes += [
            RemoveRedundantAssignmentTasklets(),
            RemoveFPTypeCasts(),
            RemoveIntTypeCasts(),
            PowerOperatorExpansion(),
            SplitTasklets(),
            RemoveMathCall(),
            # Stage every ``Tasklet -> global-array -> Tasklet`` hop through
            # transient scalars (the cloudsc zsolqa / zqlhs reuse). A global
            # array routed between two tasklets forces a memory round-trip the
            # tile descent cannot register-promote; staging it (disjoint -> two
            # scalars + preserved store; RMW -> one scalar carrying the value +
            # an assign store) decouples the producer/consumer dataflow from the
            # global node while keeping the store. Runs after SplitTasklets (so
            # the hops are single-op) and before MarkTileDims (so the staged
            # transients are what gets tiled).
            StageGlobalArrayThroughScalars(),
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
        # ``fuse_overlapping_loads`` requires a NestedSDFG body to fuse inside
        # (the fusion pass walks ``TileLoad`` groups in a single state — the
        # body-NSDFG shape that ``PromoteNSDFGBodyToTiles`` produces). Promote
        # only fires on already-NSDFG bodies, so when the knob is on we also
        # nest the body so a flat axpy / jacobi body becomes one too. Applies
        # at every K (the fuse pass keeps per-load dense tile transients;
        # binop consumers stay on dense (W,)/(W_0, W_1) views regardless of
        # whether the source ``<base>_vec`` boundary is 1D or N-D).
        if nest_map_bodies or fuse_overlapping_loads:
            # Single-emit-path mode: nest EVERY innermost map body into one
            # NestedSDFG so the descent (PromoteNSDFGBodyToTiles) tiles all
            # bodies — a flat axpy-style body and a reused-scalar (vbor) body
            # then look identical to the tiler, and EmitTileOps is a no-op. This
            # nests unconditionally and is K-dim-agnostic: it wraps the body of
            # whatever innermost map exists, including the collapsed multi-param
            # ``(i, j[, k])`` map. ``nest_provably_divisible=True`` disables the
            # (single-dim, ``range[-1]``-only) divisibility skip — we always
            # nest, so ``vector_width`` is unused and deliberately left default.
            passes.append(NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True))
        passes += [
            MarkTileDims(widths=widths_t),
            GenerateTileIterationMask(widths=widths_t),
            StrideMapByTileWidths(widths=widths_t),
        ]
        # Pre-emit: split a body-NSDFG boundary connector whose propagated
        # outer subset has a non-tiled dim of extent > 1 (heat3d-style
        # stencil pattern: ``j, k`` tiled, ``i`` carries 3-point stencil at
        # ``i-1, i, i+1``) into per-slice connectors. Promote's downstream
        # widening then sees clean extent-1 non-tiled dims per slice and
        # handles them via the existing degenerate path. Runs after
        # ``StrideMapByTileWidths`` (so the tile-var classification matches
        # the post-stride spec) and before Promote.
        from dace.transformation.passes.vectorization.split_multi_slice_boundary_connectors import (
            SplitMultiSliceBoundaryConnectors, )
        passes.append(SplitMultiSliceBoundaryConnectors(widths=widths_t))
        # Reify body-NSDFG ``AccessNode -[other_subset]-> AccessNode`` edges
        # left behind by ``RemoveRedundantAssignmentTasklets`` (and any other
        # pass that collapses an assign-tasklet into a single AN -> AN
        # memlet) into an ``_out = _in`` tasklet on 1-element residuals so the
        # descent's classify / promote / fan-out walkers find them. Refuses
        # multi-element residuals — auto-vectorization does not support
        # multi-element ``other_subset``. Runs right before Promote so it's
        # the last shape-pass before the descent and is scoped to inner-body
        # NSDFGs only (the outer-SDFG AN -> AN scatter/gather staging used by
        # legacy 1D detection is left alone).
        passes.append(ResolveOtherSubsetANEdges())
        passes += [
            # Tile a flat body-NSDFG (vbor-style scalar chain) in place so
            # EmitTileOps can skip it; EmitTileOps still raises for un-handled
            # NSDFG bodies (the carried-dep LoopRegion cases stay clean skips).
            PromoteNSDFGBodyToTiles(widths=widths_t),
        ]
        if fuse_overlapping_loads:
            # Collapse overlapping per-lane TileLoad fans (the stencil pattern
            # ``A[i:i+W], A[i+1:i+W+1], ...``) into one wider union transient
            # ``<base>_vec`` shared across the consumers. Runs after Promote
            # (when the per-subset TileLoad nodes exist in the body NSDFG) and
            # before ``EmitTileOps`` (which would otherwise rewrite the body
            # again). See ``fuse_overlapping_tile_loads.py`` for the structural
            # contract.
            from dace.transformation.passes.vectorization.fuse_overlapping_tile_loads import (
                FuseOverlappingTileLoads, )
            passes.append(FuseOverlappingTileLoads())
        passes += [
            EmitTileOps(widths=widths_t),
        ]
        super().__init__(passes)
        self._widths = widths_t
        self._target_isa = target_isa
        self._num_cores = num_cores
        self._remainder_strategy = remainder_strategy
        self._branch_mode = branch_mode
        self._nest_map_bodies = nest_map_bodies
        self._loop_to_map_permissive = loop_to_map_permissive

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
        from dace.transformation.dataflow import WCRToAugAssign
        from dace.transformation.interstate import LoopToMap, RefineNestedAccess
        # WCR sinks that the tile path CAN handle (the post-:class:`NormalizeWCRSource`
        # ``tile -> _wcr_src (Scalar) -[wcr]-> sink`` shape, lowered to a
        # ``TileReduce``) are left alone — running ``WCRToAugAssign`` on them
        # would convert each ``s += a[i]`` reduction into an RMW tasklet
        # ``s = s + a[i]`` whose operands are both Scalar (acc + per-iteration
        # scalar), losing the reduction semantics the tile reduce expansion
        # needs. Every OTHER WCR (the historical "we can't handle this"
        # case) stays converted via the augassign fallback on a follow-up
        # pass below — first lower the recognised reductions, then convert
        # anything left over.
        from dace.transformation.passes.normalize_wcr_source import (NormalizeWCRSource)
        NormalizeWCRSource().apply_pass(sdfg, {})
        sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
        # ``LoopToMap`` parallelises every data-parallel ``for`` loop into a map
        # so the tile path can tile it. On its own it propagates WHOLE-array body
        # edges (``e[0:N]`` instead of the per-iteration slice ``e[i]`` a
        # hand-written ``dace.map`` produces), which the tile boundary-widening
        # mis-tiles (duplicate tiles). ``RefineNestedAccess`` is the canonicaliser
        # that recovers the per-iteration form: it tightens the outer memlet to
        # what the body NSDFG actually accesses (``e[i]``; a gather ``b[idx[i]]``
        # stays the whole-array ``b(1)[0:N]``, matching the hand-map granularity).
        self._refine_loop_to_map_bodies(sdfg, LoopToMap, RefineNestedAccess)
        # Normalise loop-nest map bodies just before MapCollapse needs them:
        # inline the loop-nesting wrapper NSDFGs so nested ``for`` loops' maps
        # become adjacent + collapse them into one multi-param map. An inout /
        # multi-state leaf compute body (cloudsc RMW chain, vbor) resists inlining
        # and stays an NSDFG for the tile descent. See ``normalize_loop_nests``.
        normalize_loop_nests(sdfg)
        result = super().apply_pass(sdfg, pipeline_results)
        self._select_tile_implementations(sdfg)
        sdfg.expand_library_nodes()
        assert_no_laneid_in_tile_path(sdfg)
        return result

    def _refine_loop_to_map_bodies(self, sdfg: dace.SDFG, loop_to_map, refine_nested_access) -> None:
        """Parallelise data-parallel loops, then canonicalise body-NSDFG memlets
        to the per-iteration form the tile descent tiles correctly.

        ``RefineNestedAccess`` tightens a whole-array outer edge (``e[0:N]``,
        what ``LoopToMap`` emits) to what the body actually reads (``e[i]``) and
        offsets the inner access to ``[0]``. This is exactly the per-iteration
        boundary form the tile-load widening wants, and it also makes a
        pre-existing multi-dim-destination write (``zqx[i, j, 4]`` into a 3-D
        array) tile-valid.

        It is applied to every body NSDFG EXCEPT a pre-existing one with an
        inout connector (an array in both ``in_connectors`` and
        ``out_connectors``). Such a body is a read-modify-write chain — e.g. the
        cloudsc snippet accumulating into ``zqlhs[i, 0, 0]`` across 6 tasklets —
        which the tile descent handles in its vbor-style form (inner access
        carries the tile var); refining it to the outer-offset boundary form
        instead mis-tiles the chain. A ``LoopToMap``-created body is always
        refined (its whole-array edges must be tightened to tile at all).

        :param sdfg: SDFG to transform in place.
        :param loop_to_map: The ``LoopToMap`` transformation class.
        :param refine_nested_access: The ``RefineNestedAccess`` transformation class.
        """
        pre = {id(n) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
        # ``loop_to_map_permissive`` (set on scatter benchmarks) lets LoopToMap
        # parallelise a scatter loop (``a[idx[i]] = ...`` — a write not uniquely
        # indexed by the iteration var, which non-permissive LoopToMap refuses) so
        # the tile path can vectorise it. Default off keeps the conservative form.
        sdfg.apply_transformations_repeated(loop_to_map, permissive=self._loop_to_map_permissive, validate=False)
        for node, graph in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, dace.nodes.NestedSDFG):
                continue
            preexisting_rmw_chain = id(node) in pre and bool(set(node.in_connectors) & set(node.out_connectors))
            if preexisting_rmw_chain:
                continue
            parent = graph.sdfg
            # One application refines every candidate connector; the bounded
            # re-check converges (each pass tightens strictly fewer dims).
            for _ in range(_MAX_REFINE_ITERS):
                if not refine_nested_access.can_be_applied_to(parent, nsdfg=node):
                    break
                refine_nested_access.apply_to(parent, nsdfg=node, save=False)

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
