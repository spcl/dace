# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeMultiDim`` — device-parameterized orchestrator for K-dim masked tile-op vectorization.

Threads locked knob config through prep/emit passes, runs ``expand_library_nodes()`` to lower tile
lib nodes to ``pure``; tail audit asserts no per-lane scalar leaked. ``device`` knob (CPU/GPU) picks
reduction-lift form (CPU horizontal SIMD fold vs GPU-placed ``Reduce``) + lifted-libnode finalize
target. Thin subclasses: :class:`VectorizeCPUMultiDim` (device=CPU), :class:`VectorizeGPUMultiDim`
(device=GPU, ``target_isa='CUDA'``, ``assume_even=True``, GPU-schedules first).

Locked knobs (constructor has full semantics):

* ``target_isa`` ∈ ``{AUTO, AVX512, AVX2, ARM_SVE, ARM_NEON, SCALAR, CUDA}`` (K=1 backend; AUTO
  detects host ISA at expansion).
* ``widths`` — innermost-last, len ∈ ``{1,2,3}``, powers of 2.
* ``remainder_strategy`` — ``masked_tail`` (default), ``full_mask``, ``scalar_postamble``.
* ``branch_mode`` — ``merge`` (default) or ``fp_factor``.

Every other combo → ``NotImplementedError``.
"""
import copy
import warnings
from typing import Literal, Optional, Tuple

import sympy

import dace
from dace import properties, symbolic
from dace.dtypes import DeviceType
import dataclasses
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.length_one_array_scalar_conversion import (
    ConvertLengthOneArraysToScalars, )
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.utils.pass_invariants import (no_wcr_in_map_body,
                                                                            no_wcr_inside_nested_sdfgs)
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
from dace.transformation.passes.vectorization.flatten_branches import FlattenBranches
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.resolve_mixed_dtype_binops import ResolveMixedDtypeBinops
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
    RemoveMathCall,
    RewriteModuloToPyMod,
    StripPowerExponentCast,
)
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.vectorization.utils.arrays import demote_connector_views
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (SetSymbolNonnegativeAssumptions,
                                                                                insert_assumption_guards)
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.vectorization.stride_map_by_tile_widths import (
    StrideMapByTileWidths, )
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
try:
    from dace.transformation.passes.normalize_wcr import NormalizeWCR
except ImportError:  # transition: canonicalize's shared reduction-normalize still lives in normalize_nested_reduction
    from dace.transformation.passes.normalize_nested_reduction import NormalizeWCR
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeStridedMaps
from dace.transformation.passes.vectorization.predicate_masked_reduction import PredicateMaskedReduction
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import SplitMapForTileRemainder
# Walker-primary pipeline. The walker (InsertTileLoadStore + PreparePerLaneIndices) stages tile
# transients and emits the TileLoad / TileStore / TileMaskGen boundary; ConvertTaskletsToTileOps
# then rewrites the raw tasklets between staged tiles into TileBinop / TileITE / TileReduce.
from dace.transformation.dataflow import MapCollapse, MapFission, WCRToAugAssign
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.interstate import (InlineMultistateSDFG, InlineSDFG, LoopToMap, RefineNestedAccess,
                                            StateFusionExtended)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.vectorization.split_multi_output_tasklets import SplitMultiOutputTasklets
from dace.transformation.passes.vectorization.normalize_masked_write_tasklets import NormalizeMaskedWriteTasklets
from dace.libraries.tileops.nodes import (TileBinop, TileLoad, TileMaskGen, TileITE, TileReduce, TileStore, TileUnop)
from dace.libraries.tileops._dispatch import select_tile_implementation

#: Tile lib-node types -- all of them, used by the implementation selector.
_TILE_NODE_TYPES = (TileBinop, TileLoad, TileMaskGen, TileITE, TileReduce, TileStore, TileUnop)


class VectorizeUnsupported(Exception):
    """A kernel the K-dim tile pipeline cannot soundly vectorize.

    Raised by the pre-tiling soundness gates when the kernel carries a shape the tile widener would
    mis-lower -- a loop-carried / nested-reduction body WCR that would race the lanes (the
    ``no_wcr_in_map_body`` / ``no_wcr_inside_nested_sdfgs`` invariants), or a prep pass that could
    not lower the kernel to a valid tileable form. :meth:`VectorizeMultiDim.apply_pass` catches it,
    restores the pre-vectorization SDFG, and returns without tiling, leaving the kernel as its
    correct, un-tiled input dataflow.

    This turns a genuine capability limit into a clean *refusal to vectorize* rather than a hard
    crash (or a silently mis-tiled, wrong-numeric result). The underlying invariant CHECK is kept
    intact -- it is the detector -- per the maintainer direction: keep the soundness guard, but let
    it decline the kernel instead of aborting the whole run. Only the safe (e.g. perfectly-nested,
    lane-disjoint) reductions the widener DOES lower pass the guard and tile; every unsound shape is
    declined here.
    """


def restore_sdfg_in_place(target: dace.SDFG, source: dace.SDFG) -> None:
    """Overwrite ``target``'s graph contents with ``source``'s while preserving ``target``'s Python
    identity (the caller still holds a reference to ``target``).

    Used to roll a caller-owned SDFG back to a pre-vectorization snapshot after a
    :class:`VectorizeUnsupported` refusal, so the caller's reference keeps pointing at a valid,
    correct SDFG. ``source`` is consumed (its blocks are re-parented onto ``target``); pass a
    throwaway deep copy. Mirrors the parent-pointer fix-up :meth:`dace.SDFG.__deepcopy__` performs
    (``reset_cfg_list`` + ``FixNestedSDFGReferences``) so nested-SDFG parent references stay coherent.

    :param target: The caller-owned SDFG to overwrite in place.
    :param source: A standalone (throwaway) SDFG whose contents ``target`` adopts.
    """
    from dace.transformation.passes.fusion_inline import FixNestedSDFGReferences
    for key, value in list(source.__dict__.items()):
        if key in ('_parent', '_parent_sdfg', '_parent_nsdfg_node', '_cfg_list', 'guid'):
            continue
        setattr(target, key, value)
    target._parent = None
    target._parent_sdfg = None
    target._parent_nsdfg_node = None
    target._cfg_list = []
    for block in target.nodes():
        block._sdfg = target
        block._parent_graph = target
    target.reset_cfg_list()
    FixNestedSDFGReferences().apply_pass(target, {})


def _expandable_during_vectorization(node) -> bool:
    """Library nodes the vectorizer's ``expand_library_nodes`` may lower: ONLY its own tile-op
    nodes (:data:`_TILE_NODE_TYPES`), nothing else (user 2026-07-09).

    Every non-tile library node -- frontend ``Reduce`` / ``Einsum`` / ``MatMul`` / ``Transpose``
    (including the ones the vectorizer itself lifts) and the scatter-guard opaque primitives
    (``ScatterConflictCheck``, ``IntegerSort``) -- is left for ``compile()`` / codegen to expand on
    the final SDFG. Expanding a frontend library node mid-pipeline re-enters the Python frontend on a
    partially-lowered SDFG (fragile: the conv/newaxis-View broadcast ``IndexError``) and is
    unnecessary -- codegen expands them anyway. ``_finalize_lifted_library_nodes`` still stamps the
    lifted ``Reduce``/``Einsum`` implementation before returning so the deferred expansion is correct.
    """
    return isinstance(node, _TILE_NODE_TYPES)


def _wcr_output_is_injective_rmw(graph, map_exit, array: str, params) -> bool:
    """True iff every inner per-element write of ``array`` into ``map_exit`` indexes with
    EVERY enclosing map param -- an injective per-element read-modify-write (s212
    ``a[i] *= c[i]``) that writes each element exactly once, NOT a cross-iteration reduction.

    A genuine reduction / contraction (gesummv ``tmp[i] += A[i,j]*x[j]``) omits a reduced param
    (``j``) from the write index, so many iterations write the same element -- that is what
    :class:`_MultiOutputReductionMapFission` must separate into one contraction per output for
    ``LiftEinsum`` / the reduction lift. An injective RMW is plain elementwise dataflow the tiler
    widens as an ordinary ``TileStore``; fissioning it is unnecessary AND unsound here: the split
    detaches the in-place ``a`` write from the anti-dependence snapshot that ordered it
    (``a_split_snap = a`` upstream of the fused map), so codegen then overwrites ``a`` before the
    snapshot reads it -- miscompiling ``b[i] += a_snap[i+1]*d[i]``. So such maps stay fused.

    Conservative: an unresolvable inner write (no matching in-edge / no subset) counts as a
    reduction (fission-eligible), never masking a genuine one.
    """
    param_syms = {str(p) for p in params}
    inner = [
        e for e in graph.in_edges(map_exit)
        if e.data is not None and e.data.data == array and e.data.subset is not None
    ]
    if not inner:
        return False
    for e in inner:
        idx_syms = {str(s) for s in e.data.subset.free_symbols}
        if not param_syms.issubset(idx_syms):
            # A map param is reduced over (absent from this write) → cross-iteration reduction.
            return False
    return True


class _MultiOutputReductionMapFission(MapFission):
    """:class:`MapFission` restricted to maps separating ≥2 distinct WCR (reduction/contraction) outputs.

    Fissions a fused multi-output reduction map — e.g. gesummv ``tmp(+)= A[i,j]*x[j]; y(+)=
    B[i,j]*x[j]`` (two matvecs in one map, split to single-output tasklets by
    :class:`SplitMultiOutputTasklets`) — into one single-contraction map per output so
    ``LiftEinsum``/reduction-lift match each. Gate ≥2 distinct WCR outputs: plain
    :class:`MapFission` would fragment a single-output elementwise chain (arc_distance) into per-op
    maps, undoing base ``MapFusion`` + tripping a copy-scope codegen bug.
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Base ``MapFission.can_be_applied`` can raise (KeyError seen on adi) on a map it can't
        # handle. Can't decide → DECLINE, don't propagate: ``apply_transformations_repeated``
        # downgrades the raise to a warning, leaving the map un-fissioned → opaque codegen error later.
        try:
            if not super().can_be_applied(graph, expr_index, sdfg, permissive):
                return False
        except Exception:
            return False
        map_exit = graph.exit_node(self.map_entry)
        # Count only GENUINE reduction / contraction outputs. An INJECTIVE per-element RMW WCR
        # (write indexed by every map param, each element written once — s212 ``a[i] *= c[i]`` /
        # ``b[i] += a_snap[i+1]*d[i]``) is elementwise, not a cross-iteration reduction, so it needs
        # no fissioning; and fissioning it detaches the in-place ``a`` write from its anti-dependence
        # snapshot, miscompiling s212 (see ``_wcr_output_is_injective_rmw``). Only a write that
        # reduces over some map param (gesummv ``tmp[i] += A[i,j]*x[j]``: ``j`` absent) is a
        # contraction ``LiftEinsum`` needs separated.
        params = self.map_entry.map.params
        wcr_arrays = {
            e.data.data
            for e in graph.out_edges(map_exit)
            if e.data is not None and e.data.data is not None and e.data.wcr is not None
            and not _wcr_output_is_injective_rmw(graph, map_exit, e.data.data, params)
        }
        if len(wcr_arrays) < 2:
            return False
        # Do NOT fission a map whose multiple WCR outputs are all independent pure-scalar
        # reductions (azimint ``s += a[j]; cnt += 1``): ``LiftMapReductionToReduce`` lifts
        # each in place, and codegen emits one ``reduction(op:var)`` clause / thread-block
        # fold per accumulator. Fissioning them would have to duplicate a shared body read
        # (the mask) and can hoist it out of the map-param scope. Fission stays for the
        # CONTRACTION case (gesummv ``tmp[i] += A[i,j]*x[j]``: param-dependent write, not a
        # pure scalar reduction) that ``LiftEinsum`` needs separated.
        from dace.transformation.passes.vectorization.lift_map_reduction import _recognize_pure_wcr_reductions
        if len(_recognize_pure_wcr_reductions(graph, self.map_entry)) >= 2:
            return False
        return True


#: "AUTO" resolves to the host's best ISA at expansion time
#: (``dace.libraries.tileops._dispatch.detect_host_isa``); the others pin one.
_VALID_ISAS = ("AUTO", "AVX512", "AVX2", "ARM_SVE", "ARM_NEON", "SCALAR", "CUDA")
_VALID_REMAINDER = ("full_mask", "masked_tail", "scalar_postamble")
_VALID_BRANCH = ("merge", "fp_factor")
_VALID_SCALAR_REMAINDER = ("scalar", "tile_k1")

#: Convergence cap for the per-NSDFG ``RefineNestedAccess`` re-check loop.
_MAX_REFINE_ITERS = 8


class _RunExpandNestedSDFGInputs(ppl.Pass):
    """Pipeline-embedded wrapper running :class:`ExpandNestedSDFGInputs` to fixed point.

    Widens body-NSDFG per-iteration boundary memlets to full source-array subset (design 2.4). MUST
    run between :class:`NestInnermostMapBodyIntoNSDFG` (mints the body NSDFG) and the walker; else
    walker classifies inner ``A[0]`` memlets as CONSTANT + stages via Scalar bridges → broken
    numerics. Embedded as Pass so ``ppl.Pipeline.apply_pass`` runs the list in order.
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
        # ``ExpandNestedSDFGInputs`` re-derives each widened connector descriptor by deep-copying
        # the outer array inward (``_replace_desc_and_uncollapse_dims``) and only clearing
        # ``transient`` -- so a frontend reshape/flatten ``View`` parent (e.g. ``C_0`` viewing
        # ``C``) re-introduces an invalid inner ``View`` connector (no viewing edge in the body).
        # Re-demote every connector ``View`` to a plain array so the widened bodies validate.
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.NestedSDFG):
                demote_connector_views(node)
        self._bind_missing_free_symbols(sdfg)
        return applied or None

    @staticmethod
    def _bind_missing_free_symbols(sdfg: dace.SDFG) -> None:
        """Identity-bind every body-NSDFG free symbol still absent from its ``symbol_mapping``.

        ``ExpandNestedSDFGInputs`` widens a per-iteration boundary memlet (``a[i]`` →
        ``a[i*n3 + n1_minus_1]``), pulling the index's free symbols into the inner memlets. It
        propagates those into ``symbol_mapping`` EXCEPT any whose name also names a parent data
        descriptor: a scalar loop parameter (``n3: dace.int64`` → a ``Scalar`` in
        ``sdfg.arrays``) used purely symbolically is filtered out by the array-name guard and
        left unbound, so ``validate`` reports "Missing symbols on nested SDFG". Bind each such
        symbol identity (outer ``n3`` → inner ``n3``); the parent scope already resolves it (the
        strided map range references it), and the type comes from the parent symbol / scalar
        descriptor (default ``int64``).

        :param sdfg: The SDFG whose body NSDFGs to repair in place.
        """
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, dace.nodes.NestedSDFG) or node.sdfg is None:
                continue
            inner = node.sdfg
            parent_sdfg = parent.sdfg
            connectors = set(node.in_connectors) | set(node.out_connectors)
            for sym in inner.free_symbols:
                sym_name = str(sym)
                if sym_name in connectors or sym_name in node.symbol_mapping:
                    continue
                if sym_name in parent_sdfg.symbols:
                    sym_type = parent_sdfg.symbols[sym_name]
                elif sym_name in parent_sdfg.arrays:
                    sym_type = parent_sdfg.arrays[sym_name].dtype
                else:
                    sym_type = dace.dtypes.int64
                if sym_name not in inner.symbols:
                    inner.add_symbol(sym_name, sym_type)
                node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)


class _RunWCRToAugAssign(ppl.Pass):
    """Pipeline-embedded re-run of :class:`WCRToAugAssign` after the trivial-assign
    cleaning, so NO write-conflict resolution survives into the body NSDFG before tiling.

    The tile path assumes no inner WCR (design 3.5: WCR lives only at the outer
    ``AN → MapExit`` boundary, lifted to ``TileReduce``). Canonicalisation lowers in-place
    ``a[i] = a[i] + b[i]`` to an AN→AN WCR copy ``b → [_out=_in] → _wcr_priv -(+=)→ a``;
    ``BypassTrivialAssignTasklets`` cleans the trivial copy while CARRYING the WCR, and this
    re-run converts the surviving ``b -(+=)→ a`` to an explicit RMW tasklet, leaving the
    body WCR-free. Placed after the cleaning so a WCR the cleaning exposes — or that
    ``LoopToMap`` minted after the earlier ``WCRToAugAssign`` — is still eliminated.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        applied = sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
        # Post/pre-condition: no WCR survives inside any body NSDFG (the tile emitters would
        # silently drop it). The allowed scalar-reduction-out form is on the NSDFG → MapExit
        # edge in the PARENT state, outside the nested SDFG, so it is not flagged. A WCR that
        # ``WCRToAugAssign`` could NOT convert (an in-place RMW whose write is not provably
        # lane-injective) is an un-tileable shape -> refuse this kernel cleanly (the orchestrator
        # restores it un-tiled) rather than crash.
        violation = no_wcr_inside_nested_sdfgs(sdfg)
        if violation is not None:
            raise VectorizeUnsupported(f"unresolved WCR inside the body NSDFG before tiling: {violation}")
        return applied or None


class _AssertNoBodyWCR(ppl.Pass):
    """Vectorizer-entry precondition: NO loose WCR survives in the region about to be
    tiled. Checks BOTH invariants — ``no_wcr_in_map_body`` (loose WCR in the flat map body)
    and ``no_wcr_inside_nested_sdfgs`` (self-contained WCR inside the body NSDFG). The only
    WCR allowed past here is a lifted reduction (TileReduce / the scalar-reduction-out
    ``NSDFG → MapExit`` edge), which neither flags; every in-place RMW must already be an
    explicit aug-assign.

    A surviving loose WCR is a reduction shape the widener cannot lower without racing the lanes
    (e.g. a nested-reduction ``_wcr_priv -[wcr]-> inner MapExit -[wcr]-> outer MapExit -> array``
    the split left in a scalar tail). Rather than abort the whole run, raise
    :class:`VectorizeUnsupported` so the orchestrator refuses this one kernel and leaves it
    correct + un-tiled. Read-only (the CHECK is preserved as the soundness guard/detector).
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        for violation in (no_wcr_in_map_body(sdfg), no_wcr_inside_nested_sdfgs(sdfg)):
            if violation is not None:
                raise VectorizeUnsupported(f"loose WCR in the region to be tiled: {violation}")
        return None


def _promote_read_output_connectors_to_inout(sdfg: dace.SDFG) -> None:
    """Promote a single-state NestedSDFG output connector also read internally to a full inout.

    Branch lowering rewrites a same-write-set masked write ``if cond: arr[s] = f(...)`` into
    ``arr[s] = ITE(cond, f(...), arr[s])`` (merge) / ``cond*f(...) + (1-cond)*arr[s]``
    (fp_factor). The else-operand is the PRE-conditional value of the write target ``arr``.
    When the rewrite happens inside the frontend's per-``if`` body NestedSDFG — whose only
    external wiring for ``arr`` is an OUTPUT connector — that else-operand read reads the
    output-connector array directly. Such a connector (declared output, read internally, not
    declared input) is malformed: it relies on in-place aliasing of the output buffer,
    ``InlineSDFG`` refuses it (``sdfg_nesting`` output-read-in-scope guard), and left nested
    it escapes the tiler (the walker does not descend into a body-internal NestedSDFG, so the
    masked compute would run once at the tile base — lane 0 only, 7/8 lanes unwritten).

    Add ``arr`` as an INPUT connector reading the pre-conditional value from the write
    target's own outer source at the write subset — the in-place ``arr = f(arr)`` read of
    ``arr`` for its RHS, an in-edge already boundary-routed through the scope entry. A masked
    write whose fused output name/subset diverges from any live input (no in-place read; e.g.
    a per-lane rename) has no such source and is left nested — correct where its result feeds
    the checked output, else numerically stale (lane 0 only) until a general re-route lands.
    """
    for node, parent in list(sdfg.all_nodes_recursive()):
        if not isinstance(node, dace.nodes.NestedSDFG) or not isinstance(parent, dace.SDFGState):
            continue
        if len(node.sdfg.nodes()) != 1:
            continue
        istate = node.sdfg.nodes()[0]
        for oc in list(node.out_connectors):
            if oc in node.in_connectors:
                continue
            read_internally = any(
                isinstance(n, dace.nodes.AccessNode) and n.data == oc and istate.out_degree(n) > 0
                for n in istate.nodes())
            if not read_internally:
                continue
            out_edge = next((e for e in parent.out_edges(node) if e.src_conn == oc), None)
            if out_edge is None or out_edge.data.data is None:
                continue
            template = next((e for e in parent.in_edges(node)
                             if e.data.data == out_edge.data.data and str(e.data.subset) == str(out_edge.data.subset)),
                            None)
            if template is None:
                continue
            node.add_in_connector(oc)
            parent.add_edge(template.src, template.src_conn, node, oc, copy.deepcopy(out_edge.data))


class _RunInlineBranchLoweredNSDFGs(ppl.Pass):
    """Fuse the branch-lowered states, promote masked-write out-connectors to inout, then
    inline the now-straight-line body NestedSDFGs into their map bodies.

    ``NormalizeStridedMaps`` / ``normalize_loop_nests`` inlined the loop-nest wrapper
    NestedSDFGs up front, but the frontend's per-``if`` body NestedSDFG still held a
    ``ConditionalBlock`` (control flow) then, so it was left nested. After the branch front
    lowers that control flow to straight-line dataflow, this pass gives those NestedSDFGs the
    same inline treatment so the tiler reaches the (now unconditional) masked compute — else,
    left nested, it runs once at the tile base (lane 0 only) and 7/8 lanes go unwritten.

    ``StateFusionExtended`` collapses the compute-then / compute-else / apply-ITE arms into a
    single state (the shape ``InlineSDFG`` requires); the inout promotion makes the write
    target well-formed so the inline is accepted.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        sdfg.apply_transformations_repeated(StateFusionExtended, permissive=False, validate=False)
        _promote_read_output_connectors_to_inout(sdfg)
        applied = sdfg.apply_transformations_repeated([InlineSDFG, InlineMultistateSDFG],
                                                      permissive=False,
                                                      validate=False)
        return applied or None


def _is_power_of_two(n: int) -> bool:
    """True iff ``n > 0`` and ``n`` is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _validate_knobs(widths: Tuple[int, ...], target_isa: str, remainder_strategy: str, branch_mode: str,
                    scalar_remainder_emit: str) -> None:
    """Reject unsupported knob combinations with one ``NotImplementedError``.

    Replaces an 8-deep ``if ...: raise`` cascade with a single table
    pass. See :class:`VectorizeMultiDim` constructor for semantics.
    """
    # Checks list is built eagerly, so the AVX-512 ``widths[-1]`` access must be empty-safe:
    # ``widths=()`` (K=0) would raise IndexError before the ``1 <= len(widths) <= 3`` check
    # reports the real reason. Sentinel last-width 8 (valid AVX alignment) lets K=0 fall
    # through to the K-range check.
    last_w = widths[-1] if widths else 8
    checks = [
        (scalar_remainder_emit in _VALID_SCALAR_REMAINDER,
         f"scalar_remainder_emit {scalar_remainder_emit!r} not in {_VALID_SCALAR_REMAINDER}"),
        (scalar_remainder_emit != "tile_k1" or remainder_strategy == "scalar_postamble",
         f"scalar_remainder_emit='tile_k1' requires remainder_strategy='scalar_postamble'; "
         f"got remainder_strategy={remainder_strategy!r}"),
        (1 <= len(widths) <= 3, f"K={len(widths)} not in {{1, 2, 3}}; got widths={widths!r}"),
        (target_isa in _VALID_ISAS, f"target_isa {target_isa!r} not in {_VALID_ISAS}"),
        (all(_is_power_of_two(w) for w in widths), f"every width must be a power of 2; got {widths!r}"),
        (target_isa != "AVX512" or last_w % 8 == 0, f"AVX-512 requires widths[-1] % 8 == 0; got widths[-1]={last_w}"),
        (target_isa != "CUDA"
         or last_w % 2 == 0, f"CUDA (half2 FP16x2) requires widths[-1] % 2 == 0; got widths[-1]={last_w}"),
        (remainder_strategy
         in _VALID_REMAINDER, f"remainder_strategy {remainder_strategy!r} not in {_VALID_REMAINDER}"),
        (branch_mode in _VALID_BRANCH, f"branch_mode {branch_mode!r} not in {_VALID_BRANCH}"),
    ]
    for ok, msg in checks:
        if not ok:
            raise NotImplementedError(f"VectorizeMultiDim: {msg}")


def normalize_loop_nests(sdfg: dace.SDFG) -> None:
    """Normalise loop-nest map bodies so a K-dim tile spans K genuine map dims.

    ``LoopToMap`` wraps each parallelised body in an NSDFG, so nested ``for j: for i:``
    becomes ``j-map → NSDFG → i-map`` — non-adjacent maps ``MapCollapse`` cannot fuse. So:

    1. Inline the wrapper NSDFGs (and any single-state leaf body) via ``InlineSDFG`` /
       ``InlineMultistateSDFG`` so the maps become adjacent. A multi-state body, or one
       with an inout connector (cloudsc ``zqlhs`` RMW / ``vbor`` reused-scalar chains), is
       left intact for the walker.
    2. ``MapCollapse`` the now-adjacent perfectly-nested single-param maps into one
       multi-param map.

    Memlet propagation is intentionally NOT run: ``ExpandNestedSDFGInputs`` later widens
    every body-NSDFG boundary memlet to the full source-array subset (design 2.4), so a
    tighten-here step would be undone. The inner classifier reads inner memlets directly.

    :param sdfg: SDFG to normalise in place.
    """
    sdfg.apply_transformations_repeated([InlineSDFG, InlineMultistateSDFG], permissive=False, validate=False)
    sdfg.apply_transformations_repeated(MapCollapse, permissive=False, validate=False)
    _resolve_body_nsdfg_symbol_aliases(sdfg)


def _resolve_body_nsdfg_symbol_aliases(sdfg: dace.SDFG) -> None:
    """Inline pure-rename symbol aliases carried on a body NSDFG's ``symbol_mapping``.

    ``MapFusion`` can fuse two nests whose bodies used different iter-var names, recording
    the equivalence as a ``symbol_mapping`` alias (``_loop_it_2 → _loop_it_0``) instead of
    renaming the inner body. The alias persists through ``normalize_loop_nests`` when the
    NSDFG is multi-state / has an inout connector. The walker keys per-lane access off the
    OUTER map's iter-var names, so an inner access in the aliased name (``src[_loop_it_2,
    _loop_it_3]``) mis-classifies as loop-invariant → scalar broadcast. Inline each
    bare-symbol rename so the body references the outer symbol directly. Only fires on a
    non-identity ``inner → bare-symbol`` mapping (an offset / affine expr is left alone).
    """
    for node, _parent in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue
        inner = node.sdfg
        renames = {}
        for inner_sym, mapped in list(node.symbol_mapping.items()):
            mapped_expr = symbolic.pystr_to_symbolic(str(mapped))
            # A bare-symbol mapping to a DIFFERENT name is a pure rename; a literal /
            # offset / identity is not.
            if not isinstance(mapped_expr, sympy.Symbol) or str(mapped_expr) == inner_sym:
                continue
            renames[inner_sym] = str(mapped_expr)
        if not renames:
            continue
        # Replace in nodes / edges only (``replace_keys=False``): the target already exists
        # in the table (its own identity mapping), so renaming the KEY would collide; the
        # stale source symbol is dropped explicitly below.
        inner.replace_dict(renames, replace_in_graph=True, replace_keys=False)
        for inner_sym, outer_sym in renames.items():
            if inner_sym in inner.symbols:
                inner.remove_symbol(inner_sym)
            node.symbol_mapping.pop(inner_sym, None)
            # Rebind the target identically so the NSDFG still declares it (int-typed iter
            # var). Usually already an inner symbol; this only fills a genuinely-new name.
            if outer_sym not in inner.symbols:
                inner.add_symbol(outer_sym, dace.int64)
            node.symbol_mapping[outer_sym] = symbolic.pystr_to_symbolic(outer_sym)


@properties.make_properties
class VectorizeMultiDim(ppl.Pipeline):
    """Drive the device-parameterized K-dim masked tile-op pipeline.

    The constructor validates the locked knob row (raises
    ``NotImplementedError`` on every unsupported combo) and assembles
    the prep + emit passes into a :class:`ppl.Pipeline`. Library-node
    expansion runs after the pipeline finishes; the audit fires last.

    The ``device`` knob selects CPU vs. GPU reduction/finalize behavior;
    :class:`VectorizeCPUMultiDim` / :class:`VectorizeGPUMultiDim` are the thin
    device-fixed entry points. ``target_isa='CUDA'`` implies ``device=GPU``.
    """

    CATEGORY: str = "Vectorization"

    def __init__(self, config: VectorizeConfig):
        """Build the orchestrator from a :class:`VectorizeConfig`.

        :param config: Every vectorizer knob bundled into one dataclass -- tile ``widths``,
            ``target_isa`` (:class:`ISA`), ``remainder_strategy`` (:class:`RemainderStrategy`),
            ``branch_mode`` (:class:`BranchMode`), ``device``, and the behavioural flags
            (``scalar_remainder_emit``, ``loop_to_map_permissive``,
            ``expand_tile_nodes``, ``validate``, ``validate_all``, ``assume_even``). See
            :class:`VectorizeConfig` for the per-field documentation.
        :raises NotImplementedError: On any disallowed knob combination (a K=1-only knob at
            K>=2, or fp_factor with a masked remainder).
        """
        self._config = config
        widths = config.widths
        target_isa = config.target_isa
        remainder_strategy = config.remainder_strategy
        branch_mode = config.branch_mode
        loop_to_map_permissive = config.loop_to_map_permissive
        scalar_remainder_emit = config.scalar_remainder_emit
        expand_tile_nodes = config.expand_tile_nodes
        validate = config.validate
        validate_all = config.validate_all
        assume_even = config.assume_even
        device = config.device
        _validate_knobs(widths, target_isa, remainder_strategy, branch_mode, scalar_remainder_emit)
        # K-dependent knob support: K=1 and K≥2 both support every (branch, remainder)
        # combo — the iter_mask only gates stores, and fp_factor lowering reduces to
        # single-op tile binops (after ``SplitTasklets``) classified the same for any K.

        widths_t = tuple(widths)
        # ``target_isa='CUDA'`` implies the GPU device (mirrors ``self._device`` below);
        # compute now so ``MarkTileDims`` tiles only GPU-resident innermost maps (half2
        # __device__ intrinsics need a GPU kernel).
        is_gpu_device = device == DeviceType.GPU or target_isa == "CUDA"
        # Front passes (target-agnostic normalization):
        #   * ConvertLengthOneArraysToScalars — length-1 arrays → true Scalars so the per-tile
        #     classifier sees CONSTANT-only sources as the Scalar operand kind (6.2).
        #   * NormalizeWCRSource — every WCR sink sourced by an AccessNode (design 3.5 locks WCR
        #     to the outside-NSDFG AN → MapExit boundary; inner state has none after staging).
        #   * BypassTrivialAssignTasklets — design 3.6: every staged copy is a direct AN → AN edge,
        #     no ``_out = _in`` tasklet. Early so the classifier / staging see clean edges.
        passes = [
            # Normalise every ``%`` (tasklet bodies, ranges, conditions, subsets, edges) to
            # ``py_mod`` for Python/NumPy modulo semantics (C ``%`` miscompiles negative
            # operands and is ill-formed for floats).
            RewriteModuloToPyMod(),
            ConvertLengthOneArraysToScalars(recursive=True, transient_only=True),
            NormalizeWCRSource(),
            BypassTrivialAssignTasklets(),
            # Strip any WCR the cleaning exposed or LoopToMap minted — the tile path must
            # see NO inner-NSDFG WCR (design 3.5).
            _RunWCRToAugAssign(),
        ]
        if branch_mode == "fp_factor":
            # FP-factor branch lowering (K=1, pairs with scalar_postamble): canonicalise
            # every same-write-set ``ConditionalBlock`` to ``ITE(c, t, e)`` tasklets (the
            # merge-mode path), then fold those into ``c*t + (1-c)*e`` arithmetic via
            # :class:`LowerITEToFpFactor`; ``SplitTasklets`` later splits the multi-op RHS
            # into single-op binops → ``TileBinop`` (no TileITE). ``EliminateBranches`` runs
            # LAST as a safety net for residual disjoint-write / single-arm shapes.
            #
            # NON-permissive (default): ``BranchElimination.condition_has_map_param`` refuses ONLY
            # a condition that uses a map param DIRECTLY (``if i < N`` -- an iteration guard whose
            # elimination to straight-line arithmetic would fabricate OOB reads); a DATA
            # conditional ``if a[i] < 0`` is allowed, because ``get_free_syms_outside_calls``
            # excludes symbols that appear only inside an array subscript. So permissive is NOT
            # needed for the data-conditional kernels (``s273`` ...) -- they lower here regardless;
            # a genuine ``i < N`` iteration guard is left for the later full-mask pass instead of
            # being unsoundly force-eliminated (user directive: no permissive in the vectorizer
            # beyond the LoopToMap scatter knob).
            passes += [
                FlattenBranches(),
                SameWriteSetIfElseToITECFG(),
                BranchNormalization(),
                LowerITEToFpFactor(),
                EliminateBranches(),
                LowerInterstateConditionalAssignmentsToTasklets(),
            ]
        else:
            # Merge branch lowering: rewrite a same-write-set ``if/else`` into
            # compute-then / compute-else / apply-merge dataflow states carrying
            # ``ITE(c, t, e)`` tasklets that lower to a per-lane TileITE select (K-dim
            # analogue of the 1D ``vector_select`` blend), gated by the tile iteration mask.
            # SameWriteSetIfElseToITECFG emits per-target ITE tasklets for two-arm
            # same-write-set if/else; BranchNormalization flattens any residual single-arm /
            # disjoint-write two-arm ConditionalBlocks (recursing through nested ones).
            passes += [FlattenBranches(), SameWriteSetIfElseToITECFG(), BranchNormalization()]
        # Branch lowering (both modes) rewrote each same-write-set ``if arr[s] = f(...)`` into
        # ``arr[s] = ITE(cond, f(...), arr[s])`` (merge) / ``cond*f + (1-cond)*arr[s]`` (fp_factor)
        # INSIDE the frontend's per-``if`` body NestedSDFG, whose only external wiring for the
        # write target ``arr`` is an OUTPUT connector. The else-operand read of the pre-conditional
        # value then reads that output-connector array directly — a form that both blocks
        # ``InlineSDFG`` (an out-connector read internally but not declared an input) AND, left
        # nested, escapes the tiler (the walker does not descend into a body-internal NestedSDFG,
        # so the masked compute would run once at the tile base — lane 0 only). Promote the read
        # to a proper inout connector, then inline the now-straight-line NestedSDFG into the map
        # body so its compute tiles alongside its siblings. Mode-agnostic (same NestedSDFG shape
        # under both branch modes).
        passes.append(_RunInlineBranchLoweredNSDFGs())
        # Full prep before tiling (so the tile path handles every kernel the frontend emits):
        #   * RemoveEmptyStates — tidy the CFG after branch lowering.
        #   * PowerOperatorExpansion — ``x**2`` → ``x*x``; ``x**c`` → ``exp(c*log(x))``.
        #   * SplitTasklets — one op per tasklet (also splits expanded power / fp_factor
        #     arithmetic) so the tile emitter can classify each.
        #   * RemoveMathCall — drop the ``math.`` prefix so ``math.exp``/``log`` match
        #     TileUnop's ``exp``/``log``.
        # (``WCRToAugAssign``, ``LoopToMap``, ``RefineNestedAccess``, ``MapCollapse`` run in
        # ``apply_pass``. ``InlineSDFGs`` is intentionally NOT run — it would flatten the
        # body NSDFGs the walker traverses.)
        passes += [
            # dtype casts (``dace.float64(x)`` ...) are NOT stripped: kept as 1-input cast
            # tasklets, split by ``SplitTasklets`` (so the feeding integer arithmetic stays
            # its natural width) and lowered to ``TileUnop(op='cast')`` — an explicit per-lane
            # convert keeping binop/unop operands uniform-dtype.
            # A size / exponent symbol carries no sign in DaCe, so a power's exponent cannot be
            # proven ``>= 0``. Set the nonnegativity assumption on signed-integer free symbols
            # (the offset/size contract) so the tile emitter classifies ``x ** N`` (N a
            # nonnegative int symbol) as ``ipow``, and so the assumption persists to the copy's
            # codegen where the global RelaxIntegerPowers relaxes the SIZE powers (``2**s`` in
            # stockham-FFT strides).
            SetSymbolNonnegativeAssumptions(),
            # Strip the frontend's redundant float cast on a power exponent
            # (``base ** float64(N)`` -> ``base ** N``): value-preserving, and it exposes the
            # integer exponent so the tile emitter can classify the power as ``ipow``. ``**`` is
            # kept as ``**`` (NOT expanded to ``pow`` / a product): the tile emitter decides
            # ``pow`` vs ``ipow`` per operand at emission time from the exponent.
            StripPowerExponentCast(),
            SplitTasklets(),
            # After splitting each body into a single primitive op, unify any mixed-dtype
            # binop by inserting cast tasklets (NumPy promotion). The tile pipeline locks
            # one dtype per lib node (design 6.2), so no ``TileBinop`` may see operands of
            # differing dtype -- resolve it here rather than refusing at conversion.
            ResolveMixedDtypeBinops(),
            RemoveMathCall(),
            # Clean empty states from branch lowering + body rewrites so the tiling passes
            # see a tidy CFG. (``ppl.Pipeline`` forbids duplicate pass types, so this single
            # end-of-prep cleanup covers both the branch-front and AST-rewrite output.)
            RemoveEmptyStates(),
        ]
        # ``assume_even`` (GPU half2): the caller guarantees every tiled map extent is an
        # exact multiple of its width, so NO remainder. Mark every eligible map ``__tile_main``
        # (no peel) — the single ``0:N:W`` map covers the range, and the ``__tile_main`` marker
        # makes GenerateTileIterationMask skip its mask (no mismatched GPU thread-block sizes).
        if assume_even:
            passes.append(SplitMapForTileRemainder(widths=widths_t, assume_even=True))
        elif remainder_strategy in ("masked_tail", "scalar_postamble"):
            # Split each K-dim tile map into a provably-divisible interior (marked
            # ``__tile_main`` → GenerateTileIterationMask skips its mask → lowered with
            # has_mask=False, the fast path) plus boundary remainder regions. Before
            # MarkTileDims so the replicated boundary maps are tagged + tiled. ``masked_tail``
            # keeps the boundary as W-strided masked slabs; ``scalar_postamble`` marks them
            # ``__scalar_tail`` (plain step-1 loops every tile prep pass skips).
            if remainder_strategy == "scalar_postamble":
                tail_mode = "tile_k1" if scalar_remainder_emit == "tile_k1" else "scalar"
            else:
                tail_mode = "masked"
            passes.append(SplitMapForTileRemainder(widths=widths_t, tail_mode=tail_mode))
        # Restore the "no bare-if in a Python tasklet" invariant for the TILED bodies only.
        # The frontend lowers ``A[mask] = value`` to a bare-if tasklet ``if __in_cond:
        # __out = value`` (newast.py); ``SplitTasklets`` leaves it intact and
        # ``SplitMapForTileRemainder`` copied it into any remainder tail. Rewrite it to the
        # first-class ``__out = IT(__in_cond, value)`` conditional-write (analysed like
        # ``ITE``, no old-value read; lowered to a masked ``TileStore``) — but ONLY in the
        # tiled bodies; scalar-tail scopes stay step-1 loops keeping the valid bare-if. So it
        # MUST run AFTER the remainder split.
        passes.append(NormalizeMaskedWriteTasklets())
        # Always-on under walker-primary: every innermost map body must be nested in a body
        # NSDFG so the walker (InsertTileLoadStore) — the only emit path — has something to
        # traverse. A flat body without an NSDFG wrapper would leave the walker idle: the map
        # step strides to W but no per-tile body is produced → silently wrong numerics.
        # ``nest_provably_divisible=True`` disables the legacy single-dim divisibility skip.
        passes.append(NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True))
        # Nesting a flat ``acc = sum(A)`` body pulls the accumulator WCR inside the body
        # NSDFG. ``NormalizeWCR`` (canonicalize's shared reduction-normalize, reused since the
        # vectorizer can run WITHOUT canonicalize) rewrites it to the boundary shape: the body
        # outputs the ADDEND (widens to a tile the walker folds via ``TileReduce``), and the
        # reduction lives on the protected ``NSDFG → _nnr_out ─[wcr:op]→ MapExit → acc`` chain.
        # The scalar-across-tiles residual is backend-emitted (CPU ``reduction(op:var)`` /
        # GPU block-reduce + one atomic per block) — no product buffer, no ``Reduce`` libnode.
        passes.append(NormalizeWCR())
        # Vectorizer-entry precondition: the just-nested region carries no loose WCR (in-place
        # RMW must already be an explicit aug-assign; genuine reductions are lifted). Asserts
        # BOTH checkers (flat-body + inside-NSDFG) so no stray WCR reaches the tile emitters.
        passes.append(_AssertNoBodyWCR())
        # Embedded wrapper — expands body NSDFG boundary memlets to the full source-array
        # subset (design 2.4). MUST run between Nest and the walker (see the class docstring).
        passes.append(_RunExpandNestedSDFGInputs())
        # Stage every ``Tasklet → non-transient → Tasklet`` bridge through per-subset
        # transient scalars (user 2026-06-10). Width-independent: one scalar per distinct
        # subset, RMW subsets fold onto one, sibling write/read pairs join via W×R dep edges.
        # After this, no global access node mediates intermediate computation — every
        # non-transient is a boundary source or sink.
        passes.append(StageGlobalArrayThroughScalars())
        # Index-subset propagation (user direction): the frontend promotes a computed index
        # ``i + offset`` to a scalar then symbol ``__sym`` used in the subset (``A[__sym]``),
        # hiding the iter-var from the classifier. Undo it so the subset reads ``A[i + offset]``
        # directly and widens to a dense load. Order: SymbolPropagation (``__sym`` → the
        # scalar) → PropagateIndexSubsets (scalar → i+offset, crossing the defining tasklet;
        # leaves data-dependent gather indices) → RemoveUnusedSymbols (sweep dead symbols).
        passes += [
            SymbolPropagation(),
            PropagateIndexSubsets(),
            RemoveUnusedSymbols(),
        ]
        # Conceptual order (user 2026-06-09):
        #   MarkTileDims              (tag the outer map with TileDimSpec)
        #   StrideMapByTileWidths     (map step 1 → W; iter_var now means "tile start")
        #   WidenAccesses             (widen memlets + transient descriptors, AFTER stride)
        #   GenerateTileIterationMask (mask reflects the final tile shape, after stride+widen)
        passes += [
            # GPU path: tile only innermost maps inside a GPU kernel (GPU_Device-scheduled,
            # or Sequential under a GPU_Device parent across NSDFG boundaries); host-side maps
            # are skipped so their half2 __device__ intrinsics never leak into host code.
            MarkTileDims(widths=widths_t, require_gpu_resident=is_gpu_device),
            StrideMapByTileWidths(widths=widths_t),
        ]
        # Walker-primary tiling: the walker stages every non-transient AccessNode inside
        # tile-tagged body NSDFGs through TileLoad / TileStore per the per-dim lattice
        # (CONSTANT → Scalar bridge; LINEAR/AFFINE/REPLICATE/MODULAR → tile bridge; GATHER →
        # materialised _idx_<k> + gather_dims). WidenAccesses folds in the per-lane idx
        # materialiser; InsertTileLoadStore wires the tiles into the ``_idx_<k>`` connectors.
        # No overlapping-load fusion: ``ExpandNestedSDFGInputs`` widens every boundary memlet
        # to the full source-array subset (2.4), so every inner TileLoad reads the same
        # full-array connector — no per-tile windows to fuse.
        passes += [
            # Unified WidenAccesses (user 2026-06-10/11): single pass widens non-transient
            # boundary subsets (``A[ii]`` → ``A[ii:ii+W]``, both ``subset`` and
            # ``other_subset``), widens lane-dep transient descriptors (Scalar / (1,) Array →
            # tile), and materialises per-lane idx tiles for every GATHER per-dim — symmetric
            # on gather (read) and scatter (write). InsertTileLoadStore then wires the tiles
            # into the _idx_<k> connectors.
            WidenAccesses(widths=widths_t),
            GenerateTileIterationMask(widths=widths_t),
            InsertTileLoadStore(widths=widths_t),
            # Converter sees the walker's lib nodes + the mask in scope; sets has_mask=True +
            # wires _mask onto Tile{Binop, Unop, ITE, Reduce}.
            ConvertTaskletsToTileOps(widths=widths_t),
        ]
        super().__init__(passes)
        self._widths = widths_t
        self._target_isa = target_isa
        # ``target_isa='CUDA'`` (GPU tile backend) implies device=GPU; an explicit
        # ``device=GPU`` also selects it. Everything else is CPU.
        self._device = DeviceType.GPU if (device == DeviceType.GPU or target_isa == "CUDA") else DeviceType.CPU
        self._remainder_strategy = remainder_strategy
        self._branch_mode = branch_mode
        self._loop_to_map_permissive = loop_to_map_permissive
        self._expand_tile_nodes = expand_tile_nodes
        self._validate = validate
        self._validate_all = validate_all
        self._assume_even = assume_even

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results) -> Optional[int]:
        """Run the prep + emit pipeline, then expand lib nodes + audit.

        The K-dim tile is taken over the last ``K`` params of one innermost map. A realistic
        kernel (``@dace.program`` → ``LoopToMap`` → ``simplify``) leaves perfectly-nested
        single-param maps (``for i: for j:`` → ``i`` map wrapping ``j`` map). We collapse
        those into one multi-param map first: for K≥2 the tile then spans K dims; for K==1
        the merged ``(i, j)`` map iterates outer dims and ``MarkTileDims`` tiles only the
        innermost param (last-``K`` slice). ``MapCollapse`` fires only on a perfectly-nested
        same-schedule pair, so a sequential/carried-dep inner loop (a ``LoopRegion``, not a
        map) is left alone.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Carry-in from any enclosing pipeline.
        :returns: Whatever the inner pipeline returned (count of rewrites).
        """
        # Snapshot the caller's SDFG BEFORE any mutation so a later ``VectorizeUnsupported``
        # refusal (an un-tileable body-WCR reduction, or a prep pass that could not lower the
        # kernel to a valid tileable form) can roll the caller-owned object back to this pristine,
        # correct input and leave it un-tiled -- a clean refusal instead of a crash or a
        # half-transformed SDFG. Cheap relative to the compile that follows; taken once per call.
        snapshot = copy.deepcopy(sdfg)
        # Always simplify first (user direction): callers may hand us an un-simplified SDFG
        # (``to_sdfg(simplify=False)``) with FunctionCallRegions / redundant states / un-inlined
        # wrappers. Up-front simplify gives every downstream pass a canonical flat-state body;
        # already-simplified inputs are a near no-op.
        # Skip ``ArrayElimination`` (mirrors canonicalize's terminal SimplifyPass): it eliminates
        # an anti-dependence SNAPSHOT copy (``a_split_snap = a``, read as ``a_split_snap[i+1]`` while
        # ``a`` is updated in place) as a "redundant" array, redirecting the read back to ``a`` --
        # which destroys the anti-dep break and miscompiles (e.g. s212 ``b[i]+=a[i+1]*d[i]``). The
        # only cost of skipping it is a dead transient left uncleaned, which the tiler ignores.
        sdfg.simplify(validate_all=self._validate_all, skip={'ArrayElimination'})
        # Infer connector types + assign default schedules/storage at the START of vectorization
        # (user 2026-07-10). This gives every map a concrete schedule -- ``Sequential`` vs
        # ``CPU_Multicore`` vs ``GPU_Device`` -- BEFORE the tile pipeline, so a downstream pass that
        # must distinguish a sequential (loop-carried) reduction from a parallel one
        # (``PrivatizeSequentialMapReductionAccumulator``) sees the real schedule, and so the tiler
        # can tile ANY schedule (CPU_Multicore / GPU_Device / Sequential) uniformly. Idempotent and
        # re-run after the reduction lifts mint new library nodes (see ``_finalize_lifted_library_nodes``).
        self._assign_default_schedules(sdfg)
        # Fold every frontend reshape / flatten ``View`` (e.g. ``C_0`` viewing ``C[0:XN, 0:YN]``
        # as flat ``C[0:XN*YN]``) into a direct access. Runs UNCONDITIONALLY -- the multi-dim
        # corpus enters via ``base_pipeline`` (simplify + LoopToMap + MapFusion), NOT canonicalize,
        # so no earlier ``RemoveViews`` ran; and even a canonicalized input is re-cleaned cheaply.
        # A surviving ``View`` gets deep-copied inward as a NestedSDFG connector descriptor by
        # ``NestInnermostMapBodyIntoNSDFG`` / ``ExpandNestedSDFGInputs`` -- a connector ``View`` has
        # no viewing edge in the body, so ``validate()`` rejects it ("Ambiguous or invalid edge
        # to/from a View access node"). Removing views here dissolves that at the root.
        RemoveViews().apply_pass(sdfg, {})
        # Fold any non-unit map step into the index (``a[i]`` under ``0:N:2`` → ``a[2*k]``
        # under ``0:int_floor(N-1,2)+1:1``) so the tiler — which requires unit-step maps
        # (``MarkTileDims`` / ``LiftMapReductionToReduce`` bail on ``step != 1``) — sees a
        # dense map whose strided index its ``dim_strides`` machinery vectorizes unchanged.
        # Before the reduction lifts below (they need a unit-step map to size the product
        # buffer). Value-preserving; no-op on unit-step maps.
        NormalizeStridedMaps().apply_pass(sdfg, {})
        # Scalar-localize an array-slot WCR reduction (``s[3] += a[i]*b[i]`` -- a map-exit WCR into
        # one element of a multi-element array) into a private scalar accumulator + init + writeback
        # so the widener folds it to per-lane partials + a ``TileReduce``. Without this the array
        # slot has no scalar VARIABLE to give K per-lane copies of, the ``no loose WCR in the map
        # body`` precondition fires, and the vectorizer bails. Runs after NormalizeStridedMaps
        # (needs unit-step maps) and BEFORE PredicateMaskedReduction / NormalizeWCR (so it sees the
        # clean top-level ``tasklet -[wcr]-> MapExit -> arr[c]`` shape). No-op on scalar / length-1
        # accumulators (already widenable), non-associative WCRs, and non-reduction shapes.
        PrepareReductionForWidening().apply_pass(sdfg, {})
        # Predicate a masked reduction (``if mask: acc op= f``) into an unconditional
        # select-addend reduction (``acc op= ITE(mask, f, identity)``) while the mask is
        # still a ConditionalBlock in the CFG. Must run BEFORE NormalizeWCR (which buries
        # the reduction across the nsdfg boundary) and before WCRToAugAssign / the reduction
        # lifts (which refuse or cannot fold a WCR trapped inside a conditional). Bit-exact:
        # the false lanes contribute the op identity. No-op on unmasked kernels.
        PredicateMaskedReduction().apply_pass(sdfg, {})
        # Normalize masked in-nsdfg write-only WCR reductions FIRST, on the frontend shape,
        # before any WCR-consuming lift below. The frontend emits ``if mask: acc += x`` (in a
        # ``dc.map``) as a WCR edge INSIDE a body NestedSDFG writing a write-only connector;
        # left as-is, ``WCRToAugAssign`` severs it (reads the write-only seed) and
        # ``MapFusionVertical`` double-counts it. ``NormalizeWCR`` (canonicalize's shared
        # reduction-normalize, the ONLY one — the vectorizer-private ``NormalizeMapReduction``
        # is retired) rewrites both masked and unmasked in-nsdfg reductions to the
        # seeded-body-local + ``_nnr_out ─[wcr:op]→ MapExit → acc`` boundary shape the walker
        # folds via ``TileReduce``. Idempotent + no-op on non-reduction kernels.
        NormalizeWCR().apply_pass(sdfg, {})
        # The pure-WCR boundary reduction (``acc = sum(A)`` — scalar ``CR:op`` at MapExit, no
        # carry-in read) is kept as a map-exit WCR (not lifted to a buffer + ``Reduce``) so
        # codegen lowers it directly (CPU ``reduction(op:var)``; GPU block-reduce + one atomic
        # per block); ``LiftMapReductionToReduce(pure_wcr_only=True)`` is the opt-in buffer form.
        # ``vectorized`` selects the CPU horizontal-SIMD fold for the RMW lift below; the GPU
        # path lifts UN-vectorized so a lifted RMW ``Reduce`` stays device-selectable.
        reduce_vectorized = self._device != DeviceType.GPU
        # Materialise nested pure-WCR reductions NOW, while the WCR is intact -- before
        # ``WCRToAugAssign`` (866) rewrites/mangles the boundary edge the pure-WCR recogniser
        # needs, and before the fission/einsum prep. A body-NSDFG map may fold SEVERAL
        # independent scalar accumulators (azimint ``s += a[j]; cnt += 1``); each is lifted in
        # place to a product buffer + WCR-free ``Reduce``, so no in-NSDFG WCR survives and no
        # fission (which would duplicate a shared body read out of map-param scope) is needed.
        # The later ``nested_only`` call (below) catches reductions whose map the vectorizer's
        # own ``LoopToMap`` mints after this point.
        from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
        LiftMapReductionToReduce(vectorized=reduce_vectorized,
                                 pure_wcr_only=True,
                                 nested_only=True,
                                 wcr_free_output=True).apply_pass(sdfg, {})
        # Prep for the einsum / reduction lifts: fission fused compute so each contraction /
        # reduction is a single-output map the lifts can match. A fused multi-output tasklet
        # (gesummv's ``ot = A[i,j]*x[j]; oy = B[i,j]*x[j]``) blocks MapFission (one component)
        # and LiftEinsum (two contractions). SplitMultiOutputTasklets → one single-output
        # tasklet per output; MapFission then separates the components into clean
        # single-contraction maps. BEFORE WCRToAugAssign so the reduction WCR stays intact.
        SplitMultiOutputTasklets().apply_pass(sdfg, {})
        sdfg.apply_transformations_repeated(_MultiOutputReductionMapFission, permissive=False, validate=False)
        # Lift tensor-contraction maps (matmul / matvec ``c(+)[i,j] = alpha*a[i,k]*b[k,j]``,
        # ``ij,j->i``, ...) to ``Einsum`` nodes BEFORE ``WCRToAugAssign`` rewrites the
        # contraction's WCR into an in-place RMW ``LiftEinsum`` can no longer match. A
        # contraction is not a per-lane broadcast, so it can't tile as an elementwise op;
        # lifting removes the map (and its WCR) from the tile path, and the node carries its
        # own fast (BLAS / ``pure``) expansion, selected in ``_finalize_lifted_library_nodes``.
        # No-op on non-contraction kernels (``LiftEinsum`` needs ≥2 tensor operands).
        PatternMatchAndApplyRepeated([LiftEinsum()]).apply_pass(sdfg, {})
        # WCRToAugAssign converts every WCR memlet that isn't a recognised reduction into an
        # in-place RMW tasklet. Recognised tile-path reductions land as ``tile → scalar
        # -[wcr]→ sink`` and are left for TileReduce; everything else converts so no stray
        # WCR survives into the body.
        sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
        # LoopToMap parallelises data-parallel `for` loops; RefineNestedAccess tightens the body's
        # outer memlet to the per-iteration slice (LoopToMap on its own emits whole-array body edges).
        self._refine_loop_to_map_bodies(sdfg, LoopToMap, RefineNestedAccess)
        # Inline wrapper NSDFGs + collapse adjacent perfectly-nested single-param maps so the K-dim
        # tile spans K genuine map dims.
        normalize_loop_nests(sdfg)
        # Re-infer schedules: the maps LoopToMap just minted (e.g. atax ``tmp[i] = sum_j``, gesummv)
        # were created with ``ScheduleType.Default``; the start-of-pass inference could not see them.
        # A downstream schedule-dependent pass (``PrivatizeSequentialMapReductionAccumulator``) must
        # distinguish a sequential from a parallel reduction on THESE maps, so re-assign concrete
        # schedules here rather than leaving Default until finalize. Idempotent on already-set maps.
        self._assign_default_schedules(sdfg)
        # Lift a loop-carried scalar reduction over an innermost map (acc threaded through
        # entry/exit via an associative op) to a product-fill map + a ``Reduce`` node with the
        # "vectorized" implementation (ExpandReduceVectorized: horizontal_reduce_<op> + scalar
        # tail). The product-fill map is ordinary elementwise dataflow the tiler strides; the
        # Reduce carries its own fold, so no remainder-map split is needed. No-op on
        # non-reductions.
        from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
        LiftMapReductionToReduce(vectorized=reduce_vectorized, rmw_only=True).apply_pass(sdfg, {})
        # A pure-WCR reduction (``acc = sum_j ...``, no carry-in) trapped INSIDE a body NSDFG
        # -- the outer loop was parallelised by ``LoopToMap`` (atax ``tmp[i] = sum_j A[i,j]*x[j]``
        # consumed in-body; trmm/symm/gramschmidt scalar folds) -- cannot keep its map-exit WCR:
        # that boundary lands inside the nested SDFG, and the multi-dim invariant forbids any loose
        # in-NSDFG WCR (the tile emitter silently drops it). The top-level ``rmw_only`` call above
        # deliberately leaves such reductions as a boundary WCR, correct ONLY when the boundary is
        # at top level. So materialise every remaining nested pure-WCR reduction to a product-fill
        # map + ``Reduce`` with a WCR-FREE explicit ``acc = acc <op> fold`` read-back: no WCR
        # survives, the product-fill map is ordinary elementwise dataflow the tiler strides, and
        # the read-back reproduces the original ``acc (op)= ...`` for any prior ``acc``. No-op at
        # top level (``nested_only``) and on non-reductions.
        LiftMapReductionToReduce(vectorized=reduce_vectorized,
                                 pure_wcr_only=True,
                                 nested_only=True,
                                 wcr_free_output=True).apply_pass(sdfg, {})
        # ExpandNestedSDFGInputs runs as an embedded Pass (see ``_RunExpandNestedSDFGInputs``);
        # it fires after Nest and before MarkTileDims / the walker.
        #
        # A pre-tiling soundness gate (or a prep pass that cannot produce a valid tileable form)
        # raises ``VectorizeUnsupported`` for a kernel the tile widener would mis-lower -- e.g. a
        # nested-reduction body WCR the remainder split leaves in a scalar tail. Rather than abort
        # the whole run, refuse THIS kernel: restore the pristine input (so the caller keeps a
        # valid, correct SDFG) and return without tiling. Warned, not silent, so a refusal is
        # visible in the log and never mistaken for a successful vectorization.
        try:
            result = super().apply_pass(sdfg, pipeline_results)
        except VectorizeUnsupported as unsupported:
            warnings.warn(f"VectorizeMultiDim: refusing to vectorize {sdfg.name!r}; leaving it "
                          f"un-tiled (correct, un-optimized): {unsupported}")
            restore_sdfg_in_place(sdfg, copy.deepcopy(snapshot))
            return None
        # Stamp ``target_isa`` + the concrete implementation on every tile lib node
        # UNCONDITIONALLY, even when expansion is deferred: a deferred SDFG
        # (``expand_tile_nodes=False``) is expanded later by the caller / ``compile()``, so its
        # nodes must already carry the chosen ISA (e.g. ``cuda``) to avoid the ``pure`` default.
        # Cheap + idempotent.
        self._select_tile_implementations(sdfg)
        # Finalize the lifted NON-tile lib nodes (Einsum, Reduce) UNCONDITIONALLY, even when
        # tile-node expansion is deferred: a deferred GPU SDFG must return with its ``Reduce``
        # GPU-placed + ``GPUAuto``/cub-stamped so the caller's later ``expand_library_nodes()``
        # lowers a real GPU reduction, not the host pure fallback. No-op with no non-tile node.
        self._finalize_lifted_library_nodes(sdfg)
        # ``expand_tile_nodes=False`` defers ``sdfg.expand_library_nodes()`` to the caller
        # (SDFG returns with tile lib nodes present). The per-lane-symbol audit (design 10.6)
        # only runs after expansion (it walks lowered tasklets), so skip it on the deferred path.
        if self._expand_tile_nodes:
            sdfg.expand_library_nodes(predicate=_expandable_during_vectorization)
            # Sweep per-lane SDFG symbols the gather lowering emitted as named intermediates
            # but now unused post-expansion. ``RemoveUnusedPerLaneSymbols`` supersedes the
            # legacy ``ClearPerLaneIndexSymbols`` hard-fail audit — per-lane symbols are
            # intentional in the gather lowering (user 2026-06-10), and the survivors are
            # populate-tasklet references the audit can't tell from accidental leaks.
            RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
        # Runtime nonnegativity guard (Piece 3 of the symbol-canonicalization contract, mirrors
        # canonicalize's terminal ``AssumeSymbolConstraints``): ``SetSymbolNonnegativeAssumptions``
        # (line ~493) made every signed-integer free symbol ``nonnegative=True`` at COMPILE time so
        # the tile emitter's offset/size reasoning (``x ** N`` -> ipow, full-tile store size
        # ``end - begin + W``) is sound; that assumption is only valid if the caller actually passes
        # ``s >= 0``. Prepend one dominating start state whose per-symbol ``__builtin_trap`` aborts
        # on a negative value so the contract is CHECKED at the boundary, not silently assumed. Emit
        # LAST -- after expansion + the per-lane audit -- so nothing reshapes the start block after
        # (the same "runs last" rule the canonicalize guard follows to avoid orphaning its state).
        # No-op when the SDFG has no signed-integer free symbols (fixed-size kernels).
        insert_assumption_guards(sdfg)
        # Final validate (gated on the ``validate`` knob, default on): the core passes
        # (WidenAccesses + tile-lib insertion) leave the SDFG transiently invalid, so the
        # per-subpass gate skips them; by here they've all completed (and, on the expand path,
        # lowered to tasklets), so the SDFG must be valid again. This is the whole-pipeline
        # validity check; ``validate_all`` additionally checks between subpasses.
        if self._validate:
            sdfg.validate()
        return result

    #: Passes after which the SDFG is intentionally TRANSIENTLY invalid (a later pass in the
    #: same structural sequence repairs it), so the per-subpass validate gate must NOT
    #: validate right after them: ``WidenAccesses`` onward widen memlets / insert tile lib
    #: nodes before the lib nodes consume them — not valid again until
    #: ``ConvertTaskletsToTileOps`` completes. ``NestInnermostMapBodyIntoNSDFG`` is NOT
    #: listed: it clears the stale scalar-staging ``other_subset`` on its boundary edges, so
    #: it leaves a VALID SDFG. The final ``sdfg.validate()`` re-checks the end state.
    _SKIP_VALIDATE_AFTER = (WidenAccesses, GenerateTileIterationMask, InsertTileLoadStore, ConvertTaskletsToTileOps)

    def apply_subpass(self, sdfg: dace.SDFG, p, state):
        """Run a pipeline subpass, then ``sdfg.validate()`` for the cleaning / structural
        passes that leave a valid SDFG.

        Validate after each such pass (user 2026-06-14) so a malformation is reported RIGHT
        AFTER the pass that produced it — with the pass name — instead of surfacing as a
        cryptic failure in a later consumer. Skips the passes that deliberately leave the SDFG
        transiently invalid (:data:`_SKIP_VALIDATE_AFTER`); the final validate in
        :meth:`apply_pass` re-checks the end state.
        """
        r = super().apply_subpass(sdfg, p, state)
        if self._validate_all and not isinstance(p, self._SKIP_VALIDATE_AFTER):
            try:
                sdfg.validate()
            except Exception as ex:  # noqa: BLE001 - a prep pass left the SDFG un-tileable
                # A prep pass that cannot lower this kernel to a VALID form (e.g. the branch front
                # leaving a nested-if body NSDFG with an unbound staged-read symbol) is a capability
                # limit for this kernel, not a run-ending fault. Signal ``VectorizeUnsupported`` so
                # ``apply_pass`` refuses this one kernel -- restoring the pristine input un-tiled and
                # warning -- instead of aborting the sweep. The pass name rides along for the log.
                raise VectorizeUnsupported(f"preprocessing pass {type(p).__name__!r} left the SDFG in a form "
                                           f"that cannot be tiled: {ex}") from ex
        return r

    @staticmethod
    def _assign_default_schedules(sdfg: dace.SDFG) -> None:
        """Give every map a concrete schedule (Sequential / CPU_Multicore / GPU_Device), replacing
        ``ScheduleType.Default``. Idempotent -- already-set maps are untouched -- so it is safe to
        re-run after a pass mints new maps (LoopToMap) or library nodes."""
        from dace.sdfg import infer_types
        infer_types.set_default_schedule_and_storage_types(sdfg, None)

    def _refine_loop_to_map_bodies(self, sdfg: dace.SDFG, loop_to_map, refine_nested_access) -> None:
        """Parallelise data-parallel loops with ``LoopToMap``.

        ``RefineNestedAccess`` is intentionally NOT run here (see body): under the
        staging-first design it is redundant (``ExpandNestedSDFGInputs`` re-widens the
        boundary memlets) and harmful (it rewrites per-lane gather indices ``idx[i] → idx[0]``).

        :param sdfg: SDFG to transform in place.
        :param loop_to_map: The ``LoopToMap`` transformation class.
        :param refine_nested_access: The ``RefineNestedAccess`` transformation class (unused).
        """
        pre = {id(n) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}
        # DEFAULT non-permissive (``_loop_to_map_permissive`` defaults False): permissive
        # ``LoopToMap`` would parallelise a scatter FOR-loop (``a[idx[i]] = ...`` -- a write the
        # iter var does not provably index uniquely) on an unchecked assumption the index is
        # injective, an unsound auto-parallelisation. In production a genuine data-parallel
        # scatter must arrive already a ``Map`` (the frontend ``dace.map``, where the author
        # asserts injectivity); the tile path then lowers it via gather/scatter tile ops (see
        # ``kernels/test_forced_scatter_gather_scatter.py``). The knob stays honored so a
        # dedicated scatter/gather tile-lowering test can opt in (``loop_to_map_permissive=True``)
        # to force its for-loop scatter into a Map and exercise the tile path directly.
        sdfg.apply_transformations_repeated(loop_to_map, permissive=self._loop_to_map_permissive, validate=False)
        # ``RefineNestedAccess`` is skipped on the K-dim path (user 2026-06-10 audit): under
        # the staging-first design it is redundant and harmful:
        # * It narrows ``LoopToMap``'s whole-array memlets (``e[0:N]``) to per-iter (``e[i]``),
        #   which ``ExpandNestedSDFGInputs`` (next) widens straight back.
        # * Worse, it substitutes parent-Map iter-vars into body-NSDFG interstate assignments,
        #   rewriting the canonical gather/scatter ``__sym = idx[i]`` to ``__sym = idx[0]`` and
        #   destroying the per-lane index semantics the materialiser's inline lift depends on.
        # The downstream pipeline handles whole-array boundary memlets natively via
        # ExpandNestedSDFGInputs + the staging chain.

    def _finalize_lifted_library_nodes(self, sdfg: dace.SDFG) -> None:
        """Select native BLAS implementations for any NON-tile library node lifted during
        prep (``Einsum`` → GEMM/GEMV, ``Reduce``, ...) so it expands to a fast library call.

        Uses :func:`~dace.transformation.auto.auto_optimize.set_fast_implementations` directly
        (user direction). Unlike canonicalize's finalize selector, it does NOT override a
        small-constant-dimension matmul to the inlined ``pure`` nest — the vectorizer prefers
        the BLAS call when available (``pure`` only when no fast impl is registered).
        ``infer_connector_types`` / ``set_default_schedule_and_storage_types`` run first
        (a library node may expand into further ones whose connectors / schedules need
        resolving). No-op with no non-tile node. The tile lib nodes are re-pinned separately
        by :meth:`_select_tile_implementations`, which runs AFTER this.
        """
        from dace.sdfg import infer_types
        from dace.transformation.auto.auto_optimize import set_fast_implementations

        has_nontile_libnode = any(
            isinstance(node, dace.nodes.LibraryNode) and not isinstance(node, _TILE_NODE_TYPES)
            for node, _ in sdfg.all_nodes_recursive())
        if not has_nontile_libnode:
            return
        # Finalize for the TARGET device, not always CPU: a GPU-path reduction must pick the
        # GPU ``Reduce`` expansion (cub / GPUAuto), else its expansion emits host code touching
        # GPU_Global buffers. CPU multi-dim stays CPU.
        device = self._device
        if device == DeviceType.GPU:
            self._gpu_place_reductions(sdfg)
        set_fast_implementations(sdfg, device)
        infer_types.infer_connector_types(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)

    def _gpu_place_reductions(self, sdfg: dace.SDFG) -> None:
        """GPU-place every lifted top-level ``Reduce`` so ``GPUAuto`` / cub applies.

        The lift emits a product-fill map into a transient buffer + a ``Reduce`` over it.
        ``GPUAuto`` requires its input ``GPU_Global`` and the node ``GPU_Device``-scheduled,
        else it silently falls back to the ``pure`` (host) expansion whose ``reduce_init`` map
        touches a ``GPU_Global`` output from host code. So schedule each ``Reduce`` (not
        already inside a GPU kernel) ``GPU_Device`` and move its transient input buffer(s) to
        ``GPU_Global`` (the fill map is already GPU-scheduled — the strided tile map).

        :param sdfg: SDFG to place in (GPU path only).
        """
        from dace.libraries.standard.nodes.reduce import Reduce
        gpu = dace.dtypes.StorageType.GPU_Global
        host = (dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.Default, dace.dtypes.StorageType.Register)
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, Reduce):
                    continue
                node.schedule = dace.dtypes.ScheduleType.GPU_Device
                for e in state.in_edges(node):
                    if e.data is None or e.data.data is None:
                        continue
                    desc = sdfg.arrays[e.data.data]
                    if desc.transient and desc.storage in host:
                        desc.storage = gpu

    def _select_tile_implementations(self, sdfg: dace.SDFG) -> None:
        """Stamp ``target_isa`` on every emitted tile lib node and resolve its concrete
        implementation before expansion.

        The choice depends only on ``target_isa`` + the node's ``K`` (both known now), so set
        ``node.implementation`` directly rather than via an ``'Auto'`` re-dispatch.
        ``select_tile_implementation`` returns ``'pure'`` for ``K ≥ 2`` and falls back to
        ``'pure'`` when the per-ISA expansion is not yet defined.

        :param sdfg: SDFG whose tile lib nodes are resolved in place.
        """
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, _TILE_NODE_TYPES):
                node.target_isa = self._target_isa
                node.implementation = select_tile_implementation(node, parent)


def _has_gpu_device_map(sdfg: dace.SDFG) -> bool:
    """True iff any map in the SDFG (recursively) is ``GPU_Device``-scheduled."""
    return any(
        isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        for n, _ in sdfg.all_nodes_recursive())


class VectorizeCPUMultiDim(VectorizeMultiDim):
    """CPU entry point for the K-dim masked tile-op pipeline.

    A thin wrapper that pins ``device=DeviceType.CPU`` on :class:`VectorizeMultiDim`;
    every other knob (``widths``, ``target_isa`` in the CPU ISA set, ``branch_mode``,
    ...) is forwarded unchanged. Kept as the stable public name / import path the
    corpus harness and tests use.
    """

    def __init__(self, config: VectorizeConfig):
        super().__init__(dataclasses.replace(config, device=DeviceType.CPU))


class VectorizeGPUMultiDim(VectorizeMultiDim):
    """GPU (CUDA half2 / FP16x2) entry point for the K-dim masked tile-op pipeline.

    A thin wrapper that fixes the GPU knob row on :class:`VectorizeMultiDim`:

    * ``device = DeviceType.GPU`` — CPU horizontal folds are replaced by GPU-placed
      ``Reduce`` nodes; lifted library nodes finalize for the GPU.
    * ``target_isa = "CUDA"`` — the innermost tile op lowers to ``dace/tile_ops/cuda.h``
      (native ``__hadd2`` / ``__hmul2`` / ... half2 intrinsics, 2 lanes per instruction;
      fp8 computes through ``float``).
    * ``widths = (2,)`` default — a half2 packs exactly two fp16 lanes.
    * ``assume_even = True`` — a GPU kernel emits NO remainder loop: the map extent is
      an exact multiple of 2, so it is a single ``0:N:2`` strided map with no masked
      tail (which would otherwise split into two GPU_Device maps of different
      thread-block sizes). The caller guarantees the even extent.

    The tile ops must run inside a GPU kernel for the ``__device__`` half2 intrinsics to
    apply, so the target map must ALREADY be ``GPU_Device``-scheduled (or a ``Sequential`` map
    nested through scopes / NestedSDFG boundaries inside a ``GPU_Device`` scope). This pass does
    NOT offload / schedule: it assumes the caller -- the canonicalize-GPU pipeline
    (``finalize_for_target(sdfg, 'gpu')`` -> :func:`~dace.transformation.passes.canonicalize.finalize.offload_to_gpu`)
    -- moved the SDFG onto the device FIRST, and only vectorizes the already-resident maps
    (:class:`MarkTileDims` with ``require_gpu_resident=True`` skips host maps). It never calls
    ``apply_gpu_transformations`` -- device offload is the caller's job, never the vectorizer's,
    so running the vectorizer is idempotent w.r.t. offloading. On an un-offloaded (host) SDFG it
    finds no GPU-resident innermost map and no-ops.
    """

    def __init__(self, config: VectorizeConfig):
        """Build the GPU orchestrator from a :class:`VectorizeConfig`.

        The GPU row is pinned regardless of the config: ``device=GPU``, ``target_isa=CUDA``,
        ``assume_even=True`` (a half2 kernel emits no remainder loop). ``widths`` (innermost
        must be even) and the remaining flags come from ``config`` -- GPU callers typically
        pass ``VectorizeConfig(widths=(2,), expand_tile_nodes=False)``.

        The input SDFG must ALREADY be GPU-offloaded (its maps ``GPU_Device``-scheduled). This
        pass never offloads / schedules the SDFG itself (no ``apply_gpu_transformations``): the
        canonicalize-GPU pipeline offloads BEFORE the vectorizer runs.

        :param config: The vectorizer configuration; its ``device`` / ``target_isa`` /
            ``assume_even`` are overridden with the GPU values.
        """
        super().__init__(dataclasses.replace(config, device=DeviceType.GPU, target_isa=ISA.CUDA, assume_even=True))
