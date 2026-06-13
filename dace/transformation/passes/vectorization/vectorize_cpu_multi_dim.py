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
  post-MVP :class:`TileITE` slice).

Refuses every other combination with ``NotImplementedError`` so the
caller is pointed at the supported config.
"""
from typing import Literal, Optional, Set, Tuple

import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.length_one_array_scalar_conversion import (
    ConvertLengthOneArraysToScalars, )
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.remove_unused_per_lane_symbols import RemoveUnusedPerLaneSymbols
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import ConvertTaskletsToTileOps
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import (
    GenerateTileIterationMask, )
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, )
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.lower_ite_to_fp_factor import LowerITEToFpFactor
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )
from dace.transformation.passes.vectorization.insert_tile_load_store import InsertTileLoadStore
# Unified WidenAccesses pass (replaces InferBodyTransientShapes + WidenScalarsToTiles
# per user direction 2026-06-10). See the pass docstring for the 5-step algorithm.
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
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
# (InsertTileLoadStore + PreparePerLaneIndices) replaces both PromoteNSDFGBodyToTiles and the legacy
# EmitTileOps boundary emission. Tasklet-to-TileBinop / TileITE / TileReduce conversion is
# pending; for now the SDFG returns with raw tasklets between staged tile transients and lib-node
# expansion handles only TileLoad / TileStore / TileMaskGen.
from dace.transformation.dataflow import MapCollapse, WCRToAugAssign
from dace.transformation.interstate import (InlineMultistateSDFG, InlineSDFG, LoopToMap, RefineNestedAccess)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.libraries.tileops.nodes import (TileBinop, TileLoad, TileMaskGen, TileITE, TileReduce, TileStore, TileUnop)
from dace.libraries.tileops._dispatch import select_tile_implementation

#: Tile lib-node types -- all of them, used by the implementation selector.
_TILE_NODE_TYPES = (TileBinop, TileLoad, TileMaskGen, TileITE, TileReduce, TileStore, TileUnop)

#: "AUTO" resolves to the host's best ISA at expansion time
#: (``dace.libraries.tileops._dispatch.detect_host_isa``); the others pin one.
_VALID_ISAS = ("AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR")
_VALID_REMAINDER = ("full_mask", "masked_tail", "scalar_postamble")
_VALID_BRANCH = ("merge", "fp_factor")
_VALID_SCALAR_REMAINDER = ("scalar", "tile_k1")

#: Convergence cap for the per-NSDFG ``RefineNestedAccess`` re-check loop.
_MAX_REFINE_ITERS = 8


class _RunExpandNestedSDFGInputs(ppl.Pass):
    """Pipeline-embedded wrapper that runs :class:`ExpandNestedSDFGInputs` to fixed point.

    The walker (:class:`InsertTileLoadStore`) traverses body NSDFGs which only exist AFTER
    :class:`NestInnermostMapBodyIntoNSDFG` runs. ``ExpandNestedSDFGInputs`` widens the
    body NSDFG's per-iteration boundary memlets to the full source-array subset
    (design section 2.4). It MUST run between Nest and the walker; otherwise the
    walker classifies inner ``A[0]`` memlets as CONSTANT and stages them via Scalar
    bridges, breaking numerics.

    Embedded as a Pass (vs a side call from ``apply_pass``) so the standard
    ``ppl.Pipeline.apply_pass`` machinery executes the whole list in order without
    fighting the cached dependency graph.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        applied = sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
        return applied or None


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
            same-write-set if/else to a per-lane :class:`TileITE` select;
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
            # ``TileBinop`` (no merge/TileITE). Pairs with scalar_postamble.
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
            # per-lane TileITE select (the K-dim analogue of the 1D
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
        # Always-on under walker-primary: every innermost map body must be nested in a body
        # NSDFG so the walker (InsertTileLoadStore) has something to traverse. The walker is the
        # ONLY emit path now -- the legacy ``EmitTileOps`` / ``PromoteNSDFGBodyToTiles``
        # descents were deleted. A flat (bare-tasklet) body without an NSDFG wrapper would
        # leave the walker with nothing to do; the orchestrator would strip the map step
        # to W but never produce a real per-tile body, resulting in silently wrong numerics.
        # ``nest_provably_divisible=True`` disables the legacy single-dim divisibility skip;
        # we always nest. The ``nest_map_bodies`` knob is kept on the constructor for harness
        # parity with the legacy 1D path but is otherwise unused.
        passes.append(NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True))
        # Embedded wrapper -- expands body NSDFG boundary memlets to the full source-array
        # subset (design section 2.4). MUST run between Nest and the walker; see the class
        # docstring for the rationale.
        passes.append(_RunExpandNestedSDFGInputs())
        # Stage every ``Tasklet -> non-transient -> Tasklet`` bridge through
        # per-subset transient scalars (user direction 2026-06-10). Width-
        # independent normalization: one scalar per distinct subset, RMW
        # subsets fold onto one scalar, sibling-write-and-sibling-read pairs
        # join via W x R dep edges. After this pass, no global access node
        # mediates intermediate computation -- every non-transient is a
        # boundary source or sink. Downstream tile widening + lib-node
        # insertion see a clean, uniform graph.
        passes.append(StageGlobalArrayThroughScalars())
        # Index-subset propagation (per user direction): the frontend promotes a
        # computed index ``i + offset`` to a scalar then a symbol ``__sym`` used in
        # the memlet subset (``A[__sym]``), hiding the iter-var from the tile-access
        # classifier. Undo that here so the subset reads ``A[i + offset]`` directly
        # and widens to a dense load. Order: SymbolPropagation (folds ``__sym`` ->
        # the scalar) -> PropagateIndexSubsets (resolver last hop: scalar -> i+offset,
        # crossing the defining tasklet; leaves data-dependent gather indices) ->
        # RemoveUnusedSymbols (sweeps the now-dead promotion symbols). Runs after
        # branch-lowering (flat states) + int-cast removal and before the tiling passes.
        passes += [
            SymbolPropagation(),
            PropagateIndexSubsets(),
            RemoveUnusedSymbols(),
        ]
        # Conceptual order (per user direction 2026-06-09):
        #   MarkTileDims                (tag the outer map with TileDimSpec)
        #   StrideMapByTileWidths       (map step 1 -> W; iter_var now means "tile start")
        #   InferBodyTransientShapes    (widen memlets + intermediate transient descriptors
        #                                AFTER stride so iter_var is in tile-start form)
        #   GenerateTileIterationMask   (mask reflects the final tile shape; runs after
        #                                stride + widening so it sees the canonical body)
        passes += [
            MarkTileDims(widths=widths_t),
            StrideMapByTileWidths(widths=widths_t),
        ]
        # Pre-emit: split a body-NSDFG boundary connector whose propagated
        # outer subset has a non-tiled dim of extent > 1 (heat3d-style
        # stencil pattern: ``j, k`` tiled, ``i`` carries 3-point stencil at
        # Walker-primary tiling (replaces SplitMultiSliceBoundaryConnectors + PromoteNSDFGBodyToTiles
        # + EmitTileOps boundary emission). The walker stages every non-transient AccessNode inside
        # tile-tagged body NSDFGs through TileLoad / TileStore based on the per-dim lattice
        # (CONSTANT -> Scalar bridge; LINEAR/AFFINE/REPLICATE/MODULAR -> tile bridge; GATHER ->
        # materialised _idx_<k> + gather_dims). The per-lane idx materialiser is folded into
        # WidenAccesses (step 5); InsertTileLoadStore wires the resulting tiles into the
        # ``_idx_<k>`` connectors of TileLoad / TileStore.
        # NOTE: ``fuse_overlapping_loads`` is intentionally NOT honoured here. Under the multi-dim
        # design ``ExpandNestedSDFGInputs`` widens every body-NSDFG boundary memlet to the full
        # source-array subset (section 2.4), so every inner TileLoad already reads the same
        # full-array connector -- there are no per-tile windows to fuse. The constructor
        # docstring documents this; the knob is kept on the constructor for harness parity with
        # the legacy 1D path but has no effect on the multi-dim pipeline.
        passes += [
            # Unified WidenAccesses pass (user direction 2026-06-10/11) --
            # replaces InferBodyTransientShapes + WidenScalarsToTiles +
            # PreparePerLaneIndices. Single pass widens non-transient boundary
            # subsets (``A[ii]`` -> ``A[ii:ii+W]``, both ``subset`` AND
            # ``other_subset``), widens lane-dep transient descriptors
            # (Scalar / (1,) Array -> tile), and materialises per-lane idx
            # tiles for every GATHER per-dim. Symmetric on gather (read) and
            # scatter (write) sides. InsertTileLoadStore then wires the
            # materialised tiles into the _idx_<k> connectors directly.
            WidenAccesses(widths=widths_t),
            GenerateTileIterationMask(widths=widths_t),
            InsertTileLoadStore(widths=widths_t),
            # Converter sees the walker's lib nodes + the mask in scope; it sets
            # has_mask=True + wires _mask onto Tile{Binop, Unop, ITE, Reduce} lib nodes.
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
        # ExpandNestedSDFGInputs runs as a Pass embedded in the pipeline (see
        # ``_RunExpandNestedSDFGInputs`` constructed at __init__) -- it has to fire AFTER
        # ``NestInnermostMapBodyIntoNSDFG`` (which mints the body NSDFG) and BEFORE
        # ``MarkTileDims`` / the walker. Embedding it as a Pass lets the standard
        # ``ppl.Pipeline.apply_pass`` machinery run the whole list in order without the
        # caller needing to split-and-rerun.
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
            # Sweep any per-lane SDFG symbols that the indirect-access (gather)
            # lowering emitted as named intermediates but that have no remaining
            # use post-expansion. The friendlier
            # :class:`RemoveUnusedPerLaneSymbols` supersedes the legacy
            # :class:`ClearPerLaneIndexSymbols` hard-fail audit -- per-lane
            # symbols are intentional in the gather lowering (per user design
            # 2026-06-10), and the surviving ones are the populate-tasklet
            # references that the audit can't tell apart from accidental leaks.
            RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
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
        # Per user direction 2026-06-10 (architectural audit): under the
        # staging-first design, ``RefineNestedAccess`` is REDUNDANT and
        # actively harmful:
        #
        # * ``LoopToMap`` emits whole-array boundary memlets (``e[0:N]``).
        # * ``RefineNestedAccess`` narrows them to per-iter (``e[i]``).
        # * ``ExpandNestedSDFGInputs`` (next in the pipeline) widens them
        #   back to whole-array (``e[0:N]``) -- contradicting RefineNestedAccess.
        #
        # Worse, ``RefineNestedAccess`` substitutes parent-Map iter-vars into
        # interstate assignments inside body NSDFGs. For the canonical gather
        # / scatter form (``__sym = idx[i]`` -- exactly the @dace.program
        # output that section 3.8 of the design names as canonical), it
        # rewrites the assignment to ``__sym = idx[0]``, destroying the
        # per-lane index semantics that the materialiser's inline lift
        # (folded GatherLift, in :func:`materialise_per_lane_index_tile`)
        # depends on.
        #
        # The old "tile-load widening expects per-iter boundary memlets" rationale
        # for RefineNestedAccess is OBSOLETE under the staging-first design
        # (``StageGlobalArrayThroughScalars`` + tile-bridge staging handle the
        # whole-array boundary form directly).
        #
        # So: skip RefineNestedAccess entirely on the K-dim path. The
        # downstream pipeline handles whole-array boundary memlets natively
        # via ExpandNestedSDFGInputs + the staging chain.

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
