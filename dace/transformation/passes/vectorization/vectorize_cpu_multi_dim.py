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
                 nest_map_bodies: bool = False):
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
        # K-dependent knob support. K=1 covers every legacy knob variant (so the
        # 1D path can be dropped); K>=2 is the pure-lowered tile path and supports
        # the merge branch with any remainder strategy (full_mask / masked_tail /
        # scalar_postamble — the scalar tail is split off per dim and stays a plain
        # step-1 scalar loop). The fp_factor branch is still K=1-only (its float
        # blend has no K-dim tile-op form yet).
        if len(widths) >= 2:
            if branch_mode != "merge":
                raise NotImplementedError(f"VectorizeCPUMultiDim: K={len(widths)} supports only "
                                          f"branch_mode='merge' (fp_factor is K=1-only); got {branch_mode!r}")
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
        if nest_map_bodies:
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
        # The vectorization / tile path does NOT handle WCR: convert every
        # write-conflict-resolution memlet (a reduction ``s += a[i]``) into an
        # explicit augmented-assignment RMW tasklet (``s = s + a[i]``) so no WCR
        # is left for the tile passes (matches the SVE path's pass-0 convention).
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
