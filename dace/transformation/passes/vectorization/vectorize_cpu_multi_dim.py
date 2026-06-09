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
from dace.transformation.passes.length_one_array_scalar_conversion import (
    ConvertLengthOneArraysToScalars, )
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.clear_per_lane_index_symbols import ClearPerLaneIndexSymbols
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import ConvertTaskletsToTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask, )
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.prepare_per_lane_indices import PreparePerLaneIndices
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, )
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.lower_ite_to_fp_factor import LowerITEToFpFactor
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )
from dace.transformation.passes.vectorization.stage_inside_body import StageInsideBody
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    PowerOperatorExpansion,
    RemoveFPTypeCasts,
    RemoveIntTypeCasts,
    RemoveMathCall,
)
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths, )
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import SplitMapForTileRemainder
# Walker-primary pipeline -- no legacy descent / emit_tile_ops imports. The walker
# (StageInsideBody + PreparePerLaneIndices) replaces both PromoteNSDFGBodyToTiles and the legacy
# EmitTileOps boundary emission. Tasklet-to-TileBinop / TileMerge / TileReduce conversion is
# pending; for now the SDFG returns with raw tasklets between staged tile transients and lib-node
# expansion handles only TileLoad / TileStore / TileMaskGen.
from dace.transformation.dataflow import MapCollapse, WCRToAugAssign
from dace.transformation.interstate import (InlineMultistateSDFG, InlineSDFG, LoopToMap, RefineNestedAccess)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.libraries.tileops.nodes import (TileBinop, TileLoad, TileMaskGen, TileMerge, TileReduce, TileStore, TileUnop)
from dace.libraries.tileops._dispatch import select_tile_implementation

#: Tile lib-node types -- all of them, used by the implementation selector.
_TILE_NODE_TYPES = (TileBinop, TileLoad, TileMaskGen, TileMerge, TileReduce, TileStore, TileUnop)

#: "AUTO" resolves to the host's best ISA at expansion time
#: (``dace.libraries.tileops._dispatch.detect_host_isa``); the others pin one.
_VALID_ISAS = ("AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR")
_VALID_REMAINDER = ("full_mask", "masked_tail", "scalar_postamble")
_VALID_BRANCH = ("merge", "fp_factor")
_VALID_SCALAR_REMAINDER = ("scalar", "tile_k1")

#: Convergence cap for the per-NSDFG ``RefineNestedAccess`` re-check loop.
_MAX_REFINE_ITERS = 8


def _is_power_of_two(n: int) -> bool:
    """True iff ``n > 0`` and ``n`` is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _validate_knobs(widths: Tuple[int, ...], target_isa: str, remainder_strategy: str, branch_mode: str,
                    scalar_remainder_emit: str) -> None:
    """Reject unsupported knob combinations with one ``NotImplementedError``.

    Replaces an 8-deep ``if ...: raise`` cascade with a single table
    pass. See :class:`VectorizeCPUMultiDim` constructor for semantics.
    """
    checks = [
        (scalar_remainder_emit in _VALID_SCALAR_REMAINDER,
         f"scalar_remainder_emit {scalar_remainder_emit!r} not in {_VALID_SCALAR_REMAINDER}"),
        (scalar_remainder_emit != "tile_k1" or remainder_strategy == "scalar_postamble",
         f"scalar_remainder_emit='tile_k1' requires remainder_strategy='scalar_postamble'; "
         f"got remainder_strategy={remainder_strategy!r}"),
        (1 <= len(widths) <= 3, f"K={len(widths)} not in {{1, 2, 3}}; got widths={widths!r}"),
        (target_isa in _VALID_ISAS, f"target_isa {target_isa!r} not in {_VALID_ISAS}"),
        (all(_is_power_of_two(w) for w in widths), f"every width must be a power of 2; got {widths!r}"),
        (target_isa != "AVX512"
         or widths[-1] % 8 == 0, f"AVX-512 requires widths[-1] % 8 == 0; got widths[-1]={widths[-1]}"),
        (remainder_strategy
         in _VALID_REMAINDER, f"remainder_strategy {remainder_strategy!r} not in {_VALID_REMAINDER}"),
        (branch_mode in _VALID_BRANCH, f"branch_mode {branch_mode!r} not in {_VALID_BRANCH}"),
    ]
    for ok, msg in checks:
        if not ok:
            raise NotImplementedError(f"VectorizeCPUMultiDim: {msg}")


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

    Memlet propagation is intentionally NOT run here: the new pipeline calls
    ``ExpandNestedSDFGInputs`` later, which widens every body-NSDFG boundary
    memlet to the full source-array subset (design section 2.4). A tighten-via-
    propagate step here would be undone immediately. The inner per-tile classifier
    (``classify_tile_access``) reads inner memlets directly, so no propagation is
    needed between inlining / collapse and ``ExpandNestedSDFGInputs``.

    Net effect: fewer downstream body shapes — a flattened single-state body tiles
    via :class:`EmitTileOps`; a preserved inout / multi-state body tiles via the
    descent.

    :param sdfg: SDFG to normalise in place.
    """
    sdfg.apply_transformations_repeated([InlineSDFG, InlineMultistateSDFG], permissive=False, validate=False)
    sdfg.apply_transformations_repeated(MapCollapse, permissive=False, validate=False)


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
                 fuse_overlapping_loads: bool = False,
                 scalar_remainder_emit: Literal["scalar", "tile_k1"] = "scalar",
                 expand_tile_nodes: bool = True):
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
        :param insert_copies: NO-OP under the multi-dim design (kept for harness parity with the 1D
            path). The inside-body staging (design section 3) inserts every copy intrinsically via
            tile transients on each non-transient access.
        :param fuse_overlapping_loads: Accepted for harness parity with the
            legacy ``VectorizeCPU``; **always a no-op** under the multi-dim
            design. With ``ExpandNestedSDFGInputs`` widening every body-NSDFG
            boundary memlet to the full source-array subset (section 2.4),
            every inner ``TileLoad`` reads the same full-array connector --
            there are no per-tile windows to fuse. Multiple non-transient
            AccessNodes referring to the same outer array each get their own
            tile-bridge transient via the walker; this is benign SDFG bloat,
            not a correctness issue. Fusion is overengineering until a kernel
            proves it matters.
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
        :param scalar_remainder_emit: How the ``scalar_postamble`` tail is
            emitted. ``"scalar"`` (default) keeps the legacy step-1 scalar
            loop running the original body, untouched by every tile prep
            pass. ``"tile_k1"`` routes the tail through the tile-op
            pipeline at ``widths=(1,)`` (single-lane "scalar tile") so the
            remainder body uses the same ``TileBinop`` / ``TileLoad`` /
            ``TileStore`` shape as the main interior, just at one element
            per iteration. Only meaningful when ``remainder_strategy=
            "scalar_postamble"``; raises at any other strategy.
        :param expand_tile_nodes: When ``True`` (default), call
            ``sdfg.expand_library_nodes()`` after the pipeline finishes so
            every emitted ``TileBinop`` / ``TileLoad`` / ``TileStore`` /
            ``TileMaskGen`` / ``TileReduce`` / ... is lowered to its
            per-ISA pure body (the SDFG is ready to compile). When
            ``False``, the orchestrator returns the SDFG with the tile
            lib nodes still present so the caller can inspect, save, or
            run further transformations against the lib-node shape;
            ``sdfg.expand_library_nodes()`` is the caller's responsibility
            and the post-expansion ``assert_no_laneid_in_tile_path``
            audit is skipped (it can only run after expansion).
        :raises NotImplementedError: On any disallowed combination (e.g. a
            K=1-only knob at K>=2, or fp_factor with a masked remainder).
        """
        _validate_knobs(widths, target_isa, remainder_strategy, branch_mode, scalar_remainder_emit)
        # K-dependent knob support. K=1 and K>=2 both support every (branch,
        # remainder) combo: the iter_mask only gates stores (the fp_factor
        # arithmetic itself runs on every lane unchanged), and the fp_factor
        # branch lowering reduces to a chain of single-op tile binops after
        # ``SplitTasklets``, which the K-dim tile emitter classifies the same
        # way regardless of K.

        widths_t = tuple(widths)
        # Front passes (target-agnostic normalization):
        #   * ConvertLengthOneArraysToScalars -- length-1 arrays become true Scalars so the per-tile
        #     classifier sees CONSTANT-only sources directly as the Scalar operand kind (section 6.2).
        #   * NormalizeWCRSource -- ensures every WCR sink is sourced by an AccessNode; design 3.5 locks
        #     WCR to the outside-NSDFG AN -> MapExit boundary so the inner state has none after staging.
        #   * BypassTrivialAssignTasklets -- design 3.6: every staged copy is a direct AN -> AN edge with
        #     no `_out = _in` tasklet between. Runs early so the classifier and the staging pass see clean
        #     edges everywhere.
        passes = [
            ConvertLengthOneArraysToScalars(recursive=True, transient_only=False),
            NormalizeWCRSource(),
            BypassTrivialAssignTasklets(),
        ]
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
            # fp_factor lowering canonicalises every same-write-set
            # ``ConditionalBlock`` to ``ITE(c, t, e)`` tasklets FIRST (same
            # path the merge mode takes), then folds those ITE calls into
            # the FP-factor arithmetic form ``c * t + (1 - c) * e`` via
            # :class:`LowerITEToFpFactor`. Without this lowering, the
            # descent later refuses every kernel that still carries a
            # ``ConditionalBlock`` (cloudsc_one fp_factor regressions).
            # ``EliminateBranches`` runs LAST as a safety net for the
            # residual disjoint-write / single-arm shapes; ``permissive``
            # is preserved because some kernels (e.g. cloudsc_tidy_branch)
            # rely on it accepting map-param-conditional shapes.
            _eb = EliminateBranches()
            _eb.permissive = True
            passes += [
                SameWriteSetIfElseToITECFG(),
                BranchNormalization(),
                LowerITEToFpFactor(),
                _eb,
                LowerInterstateConditionalAssignmentsToTasklets(),
            ]
        else:
            # Merge branch lowering: rewrite a same-write-set ``if/else`` into
            # compute-then / compute-else / apply-merge dataflow states carrying
            # ``ITE(c, t, e)`` tasklets. Flattens the ConditionalBlock so
            # PromoteNSDFGBodyToTiles can descend; the ITE tasklets lower to a
            # per-lane TileMerge select (the K-dim analogue of the 1D
            # ``vector_select`` blend), gated by the tile map's iteration mask.
            # SameWriteSetIfElseToITECFG handles two-arm same-write-set
            # if/else by emitting per-target ITE tasklets; the residual
            # BranchNormalization flattens any remaining single-arm /
            # disjoint-write two-arm ConditionalBlocks (and recurses through
            # nested ones via the fix-point loop) so the descent sees pure
            # dataflow.
            passes += [SameWriteSetIfElseToITECFG(), BranchNormalization()]
        # Full prep, run BEFORE tiling exactly as the legacy 1D pipeline
        # (vectorize_cpu.py) does, so the tile path handles the same kernels:
        #   * RemoveEmptyStates — tidy the CFG after branch lowering.
        #   * RemoveFP/IntTypeCasts — strip ``dace.floatNN``/``intNN`` casts (the
        #     tile binop re-promotes operands to the output dtype).
        #   * PowerOperatorExpansion — ``x**2`` -> ``x*x``; ``x**c`` ->
        #     ``exp(c*log(x))`` (reusing TileUnop exp/log + TileBinop mul).
        #   * SplitTasklets — one op per tasklet (also splits the expanded power
        #     / fp_factor merge arithmetic) so the tile emitter can classify each.
        #   * RemoveMathCall — drop the ``math.`` prefix the power expansion emits
        #     so ``math.exp``/``math.log`` match TileUnop's ``exp``/``log``.
        # (``WCRToAugAssign``, ``LoopToMap``, ``RefineNestedAccess`` and
        # ``MapCollapse`` run in ``apply_pass``. ``InlineSDFGs`` is
        # intentionally NOT run: it would flatten the body NSDFGs
        # ``PromoteNSDFGBodyToTiles`` descends into.)
        # The redundant-copy / staging cleanup that previously lived here was
        # dropped because the required safety analysis is intractable; the
        # replacement design stages every global-array I/O through transient
        # scalars and emits broadcast / scatter / replicate / reduce only on
        # copy nodes inserted after the tile compute (see the multi-dim K=1
        # and K=2 path docs in the orchestrator slice that follows).
        passes += [
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
            if remainder_strategy == "scalar_postamble":
                tail_mode = "tile_k1" if scalar_remainder_emit == "tile_k1" else "scalar"
            else:
                tail_mode = "masked"
            passes.append(SplitMapForTileRemainder(widths=widths_t, tail_mode=tail_mode))
        # ``fuse_overlapping_loads`` is a no-op under multi-dim (full-subset boundary makes fusion
        # structurally unnecessary; see the constructor's :param: doc). It no longer gates
        # NestInnermostMapBodyIntoNSDFG.
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
        ]
        # Pre-emit: split a body-NSDFG boundary connector whose propagated
        # outer subset has a non-tiled dim of extent > 1 (heat3d-style
        # stencil pattern: ``j, k`` tiled, ``i`` carries 3-point stencil at
        # Walker-primary tiling (replaces SplitMultiSliceBoundaryConnectors + PromoteNSDFGBodyToTiles
        # + EmitTileOps boundary emission). The walker stages every non-transient AccessNode inside
        # tile-tagged body NSDFGs through TileLoad / TileStore based on the per-dim lattice
        # (CONSTANT -> Scalar bridge; LINEAR/AFFINE/REPLICATE/MODULAR -> tile bridge; GATHER ->
        # materialised _idx_<k> + gather_dims). PreparePerLaneIndices runs in parallel as the
        # standalone gather-index materialiser; for the canonical case the walker inlines its
        # logic, so this pass is a no-op for bodies the walker already handled (it's kept here for
        # pipelines that prefer the standalone materialisation step).
        # NOTE: ``fuse_overlapping_loads`` is intentionally NOT honoured here. Under the multi-dim
        # design ``ExpandNestedSDFGInputs`` widens every body-NSDFG boundary memlet to the full
        # source-array subset (section 2.4), so every inner TileLoad already reads the same
        # full-array connector -- there are no per-tile windows to fuse. The constructor
        # docstring documents this; the knob is kept on the constructor for harness parity with
        # the legacy 1D path but has no effect on the multi-dim pipeline.
        passes += [
            PreparePerLaneIndices(widths=widths_t),
            StageInsideBody(widths=widths_t),
            # Convert in-body binary tasklets (``_o = _a <op> _b``) to TileBinop lib nodes so the
            # post-expansion pure-loop body operates on tile-shape register transients (design
            # section 5.1 + section 6.7). First-slice scope: BINARY Tile+Tile only.
            ConvertTaskletsToTileOps(widths=widths_t),
        ]
        super().__init__(passes)
        self._widths = widths_t
        self._target_isa = target_isa
        self._num_cores = num_cores
        self._remainder_strategy = remainder_strategy
        self._branch_mode = branch_mode
        self._nest_map_bodies = nest_map_bodies
        self._loop_to_map_permissive = loop_to_map_permissive
        self._expand_tile_nodes = expand_tile_nodes
        self._insert_copies = insert_copies

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
        # WCRToAugAssign converts every WCR memlet that isn't a recognised reduction shape into an
        # in-place RMW tasklet. The recognised tile-path reductions (post-NormalizeWCRSource in the
        # constructor pipeline) land as `tile -> scalar -[wcr]-> sink` and are left alone -- they're
        # lowered to TileReduce; everything else converts so no stray WCR survives into the body.
        sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
        # LoopToMap parallelises data-parallel `for` loops; RefineNestedAccess tightens the body's
        # outer memlet to the per-iteration slice (LoopToMap on its own emits whole-array body edges).
        self._refine_loop_to_map_bodies(sdfg, LoopToMap, RefineNestedAccess)
        # Inline wrapper NSDFGs + collapse adjacent perfectly-nested single-param maps so the K-dim
        # tile spans K genuine map dims.
        normalize_loop_nests(sdfg)
        # ExpandNestedSDFGInputs establishes the section 2.4 boundary contract: every body NSDFG's
        # in/out connector reads/writes the full outer array; inner descriptors mirror outer shape.
        # All downstream classification / staging / lowering runs inside the body against the inner
        # full-shape mirror.
        sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
        result = super().apply_pass(sdfg, pipeline_results)
        # ``expand_tile_nodes=False`` defers ``sdfg.expand_library_nodes()``
        # to the caller — the SDFG returns with tile lib nodes still
        # present (for inspection / saving / further transformations).
        # The :class:`ClearPerLaneIndexSymbols` audit (design section 10.6)
        # runs only after expansion -- it walks lowered tasklets so any
        # ``_laneid_<i>``-style leak would be visible. Skip it on the
        # deferred path.
        if self._expand_tile_nodes:
            self._select_tile_implementations(sdfg)
            sdfg.expand_library_nodes()
            ClearPerLaneIndexSymbols().apply_pass(sdfg, {})
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
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, _TILE_NODE_TYPES):
                node.target_isa = self._target_isa
                node.implementation = select_tile_implementation(node)
