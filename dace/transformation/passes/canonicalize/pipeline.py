# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline.

Rewrites an SDFG into a deterministic canonical form so later passes (fusion,
vectorization, scheduling, equivalence checks) observe one shape per
computation. See ``DESIGN.md`` for the rationale and ordering constraints.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from dace import SDFG, properties
from dace.transformation import transformation
from dace.transformation import pass_pipeline as ppl

from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.lower_ite_to_fp_factor import LowerITEToFpFactor
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary

from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy)
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern)
from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import (
    CleanTaskletToScalarSliceToAccessNodePattern)
from dace.transformation.passes.scalar_fission import PrivatizeScalars, ScalarFission
from dace.transformation.passes.accumulator_to_map_and_reduce import AccumulatorToMapAndReduce
from dace.transformation.passes.parallelization_prep import (BestEffortLoopPeeling, ShortLoopUnroll,
                                                             DEFAULT_UNROLL_LIMIT)
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import MaterializeLoopExitSymbols
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.scatter_to_guarded_maps import ScatterToGuardedMaps
from dace.transformation.passes.privatize_scatter_reduction import PrivatizeScatterReduction
from dace.transformation.passes.parallelize_under_constraint import ParallelizeUnderConstraint
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.buffer_expansion import BufferExpansion
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.early_exit_to_find_index import EarlyExitToFindIndex
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination

from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended


def _structural_cleanup(label: str) -> List[Tuple[str, ppl.Pass]]:
    """Between-phase structural cleanup (never ``SimplifyPass`` mid-pipeline):
    fuse adjacent states, flatten nested SDFGs, then drop empty states, so each
    phase starts from a tidy state machine.

    Order: ``StateFusionExtended`` (a strict superset of ``StateFusion``;
    accepts everything the base accepts and additionally fuses across
    happens-before dependencies, emitting empty-memlet ordering edges as
    needed) collapses adjacent states; both inliners flatten nestings --
    ``InlineSDFG`` a single-``SDFGState`` NestedSDFG, ``InlineMultistateSDFG``
    the control-flow-bearing NestedSDFGs that map->loop lowering produces
    (a NestedSDFG wrapping a ``LoopRegion``/``ConditionalBlock``); without
    the latter those nestings are permanent, burying loops so
    ``MoveIfIntoLoop`` and cross-nest fusion cannot see them.
    ``EmptyStateElimination`` then removes the empty states fusion/inlining
    leave behind. None of these changes the computation; they only normalize
    structure.

    The non-extended ``StateFusion`` is intentionally NOT called here -- it
    only runs inside ``SimplifyPass`` (the end-of-canonicalize Simplify
    invocation and any caller-driven Simplify). Every shape it can fuse, the
    extended variant can fuse; running both back-to-back used to mask gaps
    in the extended matcher.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs for the cleanup, in order.
    """
    # Order rationale:
    # * ``StateFusionExtended`` -- collapse adjacent states first.
    # * ``InlineMultistateSDFG`` + ``InlineSDFG`` -- flatten NestedSDFG
    #   nestings so all subsequent cleanup passes can see across the
    #   boundary.
    # * ``RemoveViews`` (PR #2335) -- folds View access nodes into the
    #   viewed array's address map: composing the view edge memlet's
    #   affine mapping into every downstream memlet (and Python
    #   tasklet subscript) eliminates the View node. Runs AFTER
    #   inlining so views surfacing from a just-flattened NSDFG also
    #   get folded.
    # * Scalar-slice fold passes -- collapse the
    #   ``AccessNode -> scalar slice -> Tasklet`` (``A``) and the
    #   inverse ``Tasklet -> scalar slice -> AccessNode`` (``A^-1``)
    #   bridges so a gather chain like ``d_index = d[i]`` reads as
    #   ``d[i]`` directly. Wired here so downstream matchers (e.g.
    #   ``EarlyExitToFindIndex`` 's cond read-analysis) see the
    #   underlying array names rather than synthetic transients. The
    #   folds had previously regressed ~13 TSVC kernels (branched
    #   min/max s314-s316, gather-sum s4115/s4116, multi-state-chain
    #   s3111/s31111/s352, etc.) by stripping a load-bearing scalar
    #   that ``LoopToReduce`` / ``LoopToScan`` matched on; those
    #   matchers have since been hardened to match through the folded
    #   form.
    # * ``EmptyStateElimination`` -- drop empty states left behind.
    return [(label, PatternMatchAndApplyRepeated([StateFusionExtended()])),
            (label, PatternMatchAndApplyRepeated([InlineMultistateSDFG()])),
            (label, PatternMatchAndApplyRepeated([InlineSDFG()])), (label, RemoveViews()),
            (label, CleanAccessNodeToScalarSliceToTaskletPattern()),
            (label, CleanTaskletToScalarSliceToAccessNodePattern()), (label, EmptyStateElimination())]


@properties.make_properties
class _PrivatizeScalarsStage(ppl.Pass):
    """Self-contained adapter for ``PrivatizeScalars`` in the recipe.

    ``PrivatizeScalars`` resolves its analysis dependencies itself when applied
    with empty results, but it is unhashable so it cannot be wrapped in a
    ``Pipeline`` (whose dependency graph keys on the pass). Adapting it here keeps
    the self-contained-stage invariant (:func:`_assert_self_contained`) honest.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Any]:
        # ``PrivatizeScalars`` resolves a ``FindAccessNodes`` analysis (keyed by
        # ``cfg_id``) and a reachability analysis that calls ``reset_cfg_list`` mid-
        # pipeline; a stale control-flow-region list (left by a prior stage's inliner)
        # then lets that reset reassign ``cfg_id`` under the cached ``FindAccessNodes``
        # result -> ``KeyError``. Refresh the list up front so both analyses agree.
        sdfg.reset_cfg_list()
        return PrivatizeScalars().apply_pass(sdfg, {})


# Per-target knob presets. ``canonicalize(..., target='cpu'|'gpu')`` picks one
# of these and any explicit knob arg overrides the preset value. Verdicts are
# justified by the perf tests in ``tests/ab_perf/``; cite the test ID alongside
# every knob that has a per-target asymmetry.
#
# ``interchange_carry_with_map`` (LoopToScan): the post-L2M ``LoopRegion[jk]
# { Map[jl] }`` shape is interchanged in place to ``Map[jl] { LoopRegion[jk] }``
# so the carry runs SEQUENTIAL per-thread.
#   - CPU win (``tests/ab_perf/test_for_1133_ab.py``): 4.6-5.2x at klev in {90,
#     96}, klon=20480, both fp32 and fp64. Fewer parallel-Map invocations and
#     no kernel-launch-style per-jk barrier; the column carry runs in a
#     register on a single thread.
#   - GPU loss (same test): 0.73-0.80x. The kernel is BW-bound and Variant A's
#     stream of short, contiguously-coalesced kernels saturates BW better than
#     Variant B's single kernel where each thread carries the accumulator
#     across ``klev`` global-memory loads with a true RAW dep chain.
# ``peel_limit`` (BestEffortLoopPeeling): TSVC corpus
# (``tests/corpus/measure_parallelization.py`` ``_PEEL_LIMIT = 4``) is the
# coverage anchor. ``peel_limit=4`` lifts every boundary-conflict TSVC
# kernel that's lift-able with peeling; higher values add cost without
# adding lifts on the corpus. Same value on both targets -- the per-loop
# search runs at canonicalize time, not in the kernel.
#
# ``break_anti_dependence`` (BreakAntiDependence): snapshot-renames pure
# read-ahead anti-dep loops so LoopToMap lifts them at the cost of a
# transient + one copy per call. AB verdict
# (``tests/ab_perf/test_canon_knobs_ab.py``, N=1M, kernel ``A[i] =
# A[i+1] + B[i]`` -- single-pass, low arithmetic intensity):
#   - CPU: off=117ms, on=136ms (off wins 1.16x). The CPU auto-vectoriser
#     handles the sequential read-ahead well; the snapshot copy
#     out-costs the parallel speedup on THIS trivial kernel.
#   - GPU: off=12490ms, on=151ms (on wins 82.6x).
# Default is ON for both: realistic chained-anti-dep graphs (many
# anti-dep loops + arithmetic per element) benefit from parallelization
# more than this single-pass benchmark shows; the 16% CPU loss on the
# trivial case is the worst case, not the typical case.
# ``scatter_to_guarded_maps`` (ScatterToGuardedMaps): inserts a sort +
# duplicate-count guard around scatter-shaped loops and lifts them to
# parallel Maps. AB verdict (``tests/ab_perf/test_scatter_ab.py``,
# N=1M, permutation idx):
#   - CPU: B (guarded) ~1.04x A (unguarded). Sort overhead is small and
#     the sorted-idx pattern is cache-friendlier; B also handles
#     non-permutation idx safely. On.
#   - GPU: B (guarded) ~1.03x A. Same reasoning. On.
# Both targets default to True; the knob exists so the AB harness can
# measure off-vs-on without resorting to pre-canonicalize hand-wiring.
# ``privatize_scatter_reductions`` (PrivatizeScatterReduction): surface a
# data-dependent scatter reduction (``hist[bin[i]] (+)= w[i]``) to a whole-buffer
# map WCR so CPU codegen privatises the accumulator with an OpenMP array-section
# ``reduction(op:hist[0:n])`` clause instead of a contended per-element atomic
# (azimint_hist: ~200x -> ~1x vs numpy). CPU-only: the clause path is gated on
# ``openmp_array_reductions`` (an OpenMP feature); on GPU a scatter accumulate stays
# an ``atomicAdd`` and this privatisation does not apply, so the knob defaults off.
_CPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': True,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': True,
}
_GPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': False,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': False,
}
_TARGET_DEFAULTS: Dict[str, Dict[str, Any]] = {'cpu': _CPU_DEFAULTS, 'gpu': _GPU_DEFAULTS}


def _resolve_target_default(target: str, knob: str, explicit: Optional[Any], fallback: Any) -> Any:
    """Pick ``explicit`` if not ``None``, else the per-target preset, else
    ``fallback``. Used to resolve every per-target knob in one place."""
    if explicit is not None:
        return explicit
    return _TARGET_DEFAULTS.get(target, {}).get(knob, fallback)


def _build_stages(unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                  peel_limit: int = 4,
                  break_anti_dependence: bool = True,
                  interchange_carry_with_map: bool = True,
                  scatter_to_guarded_maps: bool = True,
                  privatize_scatter_reductions: bool = True,
                  assume_parallel_guards: bool = False,
                  target: str = 'cpu',
                  lift: bool = True,
                  lift_copy: bool = True,
                  semantic_lifting: bool = True) -> List[Tuple[str, ppl.Pass]]:
    """Build the loop-centric canonicalization recipe as one flat list.

    :param unroll_limit: Fully unroll constant-trip loops with at most this many
                         iterations before the reduction/parallelize stages
                         (``ShortLoopUnroll``; 0 disables).
    :param peel_limit: Best-effort loop peeling before ``parallelize``
                       (``BestEffortLoopPeeling``); 4 (default), 0 disables it. The
                       per-loop-isolated, can-be-applied-pre-filtered search only
                       fires on loops ``LoopToMap`` already refused, so it no-ops on
                       the mappable majority; on by default to maximize parallelism.
    :param break_anti_dependence: Snapshot-rename pure read-ahead anti-dependence
                                  loops before ``parallelize`` (``BreakAntiDependence``);
                                  on by default (it adds a transient + a copy, but
                                  unlocks read-ahead WAR loops for ``LoopToMap``).
    :param interchange_carry_with_map: ``LoopToScan`` knob (see
                                       ``_CPU_DEFAULTS`` / ``_GPU_DEFAULTS``
                                       above): relocate the carry LoopRegion
                                       INTO the per-column Map so the scan runs
                                       sequential-per-thread. On for CPU, off
                                       for GPU.

    Every map is lowered to a ``LoopRegion`` up front so all canonicalization
    runs on a single representation (one fission/normalize/reduce path, no
    map/loop duplication, no hybrids); ``LoopToMap`` recovers parallelism near
    the end, then maps are fused. Returns ``(stage_label, pass)`` pairs with
    fresh instances each call.

    ``SimplifyPass`` runs at the very start, after the cleaning passes (unique
    loop iterators, split tasklets, trivial-tasklet cleanup), once more right
    after ``ShortLoopUnroll`` to collapse the redundant straight-line code an
    unroll produces, and once at the end -- never otherwise between transforming
    stages. Between-stage structural cleanup is ``StateFusionExtended`` +
    ``InlineSDFG`` instead.
    ``LoopStridePermutation`` is an explicit no-op so the pipeline shape is
    honest and slottable.
    """
    s: List[Tuple[str, ppl.Pass]] = []

    # Canonicalization runs UniqueLoopIterators with the post-value epilogue
    # OFF: it is a Fortran-frontend convenience (materialise ``<i> = post``
    # so downstream reads of the un-renamed name still see the counted-DO
    # exit value), but canonicalize already rewrites every use site to the
    # unique ``_loop_it_<N>`` name, so the epilogue would be a dead-state
    # assignment that keeps the original symbol declaration live across
    # NestedSDFG boundaries and re-introduces the alias hazard the pass
    # exists to remove.
    _uniq = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    _uniq2 = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    _uniq_fis = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    _uniq_unroll = UniqueLoopIterators(assign_loop_iterator_post_value=False)

    # clean: unique loop iterators -> split tasklets -> the single SimplifyPass
    # (only here and at the end). Trivial-tasklet elimination now opens the
    # 'reduce' recipe (after simplify), not here.
    # NormalizeNegativeStride runs first so every downstream matcher
    # (LoopToMap's affine subset classifier, LoopToScan's ``stride != 1``
    # refusal, RerollUnrolledLoops) only ever sees positive-stride loops.
    # ContinueToCondition runs explicitly after the initial cleanup passes (it is
    # also inside SimplifyPass, but running it here lifts ``continue`` -> guarding
    # condition before the structural transforms, the same way the break lift is
    # applied early). A no-op on kernels without a ``continue`` (e.g. the current
    # TSVC corpus emits none); it hardens the pipeline for kernels that do.
    # RewriteModuloToPyMod runs first: normalise ``a % b`` -> ``py_mod(a, b)`` up
    # front so the canonicalized reference, every downstream tasklet split, and the
    # base codegen all carry Python/NumPy modulo semantics (cppunparse lowers a bare
    # ``%`` to C's dividend-sign ``%``, which miscompiles negative operands).
    # StateFusionExtended runs as an early cleaning pass (after SimplifyPass's
    # own non-extended StateFusion): merging adjacent states up front collapses
    # multi-state loop/branch bodies into the single-state shape the main
    # LoopFission path (and the reduction lifts) require, so a body that was
    # split across states becomes fissionable / liftable instead of being left
    # alone. The later _structural_cleanup runs only have to mop up what the
    # transforms re-introduce.
    # RemoveViews runs at the very FRONT of the pipeline -- before every semantic and
    # structural pass (loop_to_symm, normalize_reduction, and the clean block below).
    # Folding View access nodes into their backing array up front means no downstream
    # matcher (the symm / reduction lifts, LoopToMap's subset classifier, StateFusion's
    # memlet rename) ever has to reason through a slice/reshape view. Library-node operand
    # views are preserved (see ``RemoveViews._is_library_node_operand``), so BLAS/MatMul
    # expansions still see their squeezed 2-D operands. The later ``_structural_cleanup``
    # RemoveViews calls only have to mop up views the transforms (re)introduce.
    s += [('clean', RemoveViews())]

    # loop_to_symm (semantic lift, BEFORE normalize_reduction): the hand-written
    # symmetric matrix-multiply nest (polybench symm) is recognised on its raw
    # frontend shape -- a 2-D map whose NestedSDFG boundary carries a triangular
    # self-scatter ``C[0:i, j]`` plus a point-write ``C[i, j]`` fed by a symmetric
    # operand -- and replaced by a ``Symm`` BLAS node (vendor dsymm / cublasDsymm).
    # It must run before normalize_reduction, which would otherwise rewrite that
    # boundary WCR into a seeded-local reduction the recogniser no longer matches. A
    # strict, no-op-on-any-deviation match (gated on the semantic-lifting knobs, like
    # LiftEinsum), so the vectorizer path (semantic_lifting=False) leaves symm as the
    # plain reduction nest.
    if semantic_lifting and lift:
        s += [('loop_to_symm', LoopToSymm())]

    # privatize_scatter (BEFORE normalize_reduction): surface a data-dependent scatter
    # reduction (``hist[bin[i]] (+)= w[i]`` -- the azimint histogram) onto the whole-buffer
    # ``NestedSDFG -> MapExit -> accumulator`` WCR edge chain, so CPU codegen privatises the
    # accumulator with an OpenMP array-section ``reduction(op:hist[0:n])`` clause (each thread
    # a private copy, uncontended accumulate, runtime tree-merge) instead of the contended
    # per-element atomic (~200x on azimint_hist). Must run BEFORE ``NormalizeWCR``: surfacing
    # the WCR here makes the outer edge non-plain, so ``NormalizeWCR`` skips the scatter and
    # its unsound drop-WCR shortcut (which turns ``oc[bin] (+)= w`` into a partial plain
    # ``oc[bin] = w`` over a per-iteration whole-array buffer, reading the rest uninitialised)
    # never fires. CPU-only (the clause path needs OpenMP array reductions); off for GPU.
    if privatize_scatter_reductions:
        s += [('privatize_scatter', PrivatizeScatterReduction())]

    # normalize_reduction (FIRST, on the frontend shape): a masked reduction emitted as an
    # in-nsdfg WCR into a write-only output connector (plain map-exit edge) is rewritten to
    # the seeded-local + map-exit-WCR shape the frontend already emits for the equivalent
    # polybench reduction (symm). Downstream then treats it like any map-exit reduction:
    # WCRToAugAssign keeps the scalar WCR, MapToForLoop's map-exit-WCR refusal keeps it a
    # parallel map, MapFusionVertical's seeded-reduction guard fires -- so it is neither
    # severed nor double-counted. Idempotent, so the vectorizer can also run it standalone.
    s += [('normalize_reduction', NormalizeWCR())]

    # A loop with a ``break`` / ``continue`` is not splittable and its induction variable
    # is not closed-form (the trip count is data-dependent), so SplitStatements / IVS below
    # cannot handle it. Lift the early exit to a find-first index + clipped range HERE --
    # before those stages -- so they only ever see the resulting break-free, clipped loop.
    s += [('clean', RewriteModuloToPyMod()), ('clean', NormalizeNegativeStride()), ('clean', _uniq),
          ('clean', SplitTasklets()), ('clean', LowerITEToFpFactor()), ('clean', ContinueToCondition()),
          ('clean', EarlyExitToFindIndex()), ('clean', SimplifyPass()),
          ('clean', PatternMatchAndApplyRepeated([StateFusionExtended()]))]

    # prep (still maps): push guarding conditionals into maps, then split
    # statements -- replicate a conditional / gather-scatter NestedSDFG per
    # independent output so it can fission later (SplitStatements subsumes the
    # former ConditionalComponentFission and also handles forward-read anti-deps).
    s += [('prep', PatternMatchAndApplyRepeated([MoveIfIntoMap()])), ('prep', SplitStatements())]

    # lift_reduce (GPU only): rewrite an accumulator loop into the "per-iteration
    # deltas into a buffer + Reduce libnode" form (while the reduction is still a Map --
    # once ``lower`` rewrites it to a LoopRegion+NestedSDFG the clean shape is gone; the
    # Reduce libnode carries its own identity). This buffer+Reduce form is the
    # GPU-efficient reduction: the Reduce node expands to a warp/block tree-reduce,
    # whereas a WCR scalar on GPU lowers to a contended per-thread ``atomicAdd``.
    #
    # On CPU we deliberately do NOT lift to buffer+Reduce: the WCR-on-map form produced
    # later by ``reduction_to_wcr_map`` (``LoopToReduce(prefer='wcr-scalar')``) lowers to
    # an OpenMP ``reduction(op:var)`` clause (per-thread privatization + tree-reduce),
    # which is the CPU-efficient form and needs no intermediate buffer.
    #
    # TODO: implement scalar -> MapExit WCR reduction lowering for GPU (hierarchical
    # warp/block reduce) so the GPU path can lower the WCR form directly, matching the
    # CPU OMP-reduction path, instead of relying on this buffer+Reduce fallback.
    if target == 'gpu':
        s += [('lift_reduce', AccumulatorToMapAndReduce())]

    # WCRToAugAssign BEFORE lower: rewrite every conflict-free (injective) WCR back to
    # an explicit RMW while maps are still maps; what stays WCR is a genuine reduction
    # that MapToForLoop then refuses to lower (kept parallel -> OMP reduction), so the
    # in-state producer->consumer edge is never severed by the map->loop round-trip.
    s += [('lower', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]
    # lower: every map -> LoopRegion (MapToLoop = reuse MapToForLoop), then
    # structural cleanup (no SimplifyPass).
    s += [('lower', PatternMatchAndApplyRepeated([MapToForLoop()]))]
    s += _structural_cleanup('lower')
    # MapToForLoop leaves empty *_pre_state / *_post_state boundary states;
    # inside a guard branch they make the body look like a heterogeneous
    # [empty, empty, loop] chain and send MoveIfIntoLoop down its imperfect
    # path to wrap *empty* states. Splice them out so the guarded body is the
    # bare loop -> MoveIfIntoLoop's clean single-loop path applies.
    s += [('lower', EmptyStateElimination())]

    # reroll: re-roll a hand-unrolled lane chain (a step-``S`` loop whose body is
    # ``m`` lanes at equally-spaced offsets ``{0, g, ..., (m-1)g}``) back to a
    # step-``g`` loop, so the lanes do not survive normalization as a strided
    # ``S*i + k`` access that blocks LoopToMap. Runs right after the maps are
    # lowered to loops, while the loop is still in step-``S`` form (before
    # normalize rescales it).
    s += [('reroll', RerollUnrolledLoops())]

    # reduce: front-loaded reduction + parallelization-prep recipe, applied right
    # after lowering. Order follows the classical recipe:
    #
    #   trivial_tasklet_elim  -- drop ``__out = __inp`` copies, exposing the bare RMW
    #     spine of accumulators (so downstream passes see the canonical shape);
    #   WCRToAugAssign        -- normalise WCR writes back into explicit ``a = a + b``
    #     augmented assignments, so every reduction reaches loop_to_reduce in one shape;
    #
    #   --- "specialize -> unroll -> IV -> LICM -> simplify" block ---
    #   PrivatizeScalars + SymbolProp + ConstProp -- specialize the symbols and fold
    #     constants into bounds/guards (visible accumulator initializers, concrete
    #     trip counts for unroll);
    #   ShortLoopUnroll       -- fully unroll tiny constant-trip loops to straight-
    #     line code (now that ConstProp has revealed the constant trip counts);
    #   _uniq_unroll          -- give the loops that survive (and any the unroll
    #     cloned) unique ``_loop_it_<N>`` names before reduction passes read them;
    #   InductionVariableSubstitution -- collapse single-tasklet 'acc = acc OP const'
    #     loops to their O(1) closed form (the classical IV / scalar-evolution shape;
    #     red-dragon-book Ch. 9.6). Runs BEFORE LICM so IV-eliminated loops don't
    #     hold up LICM-eligible expressions in their bodies;
    #   LoopInvariantCodeMotion -- hoist loop-invariant tasklets out of loop bodies
    #     and map scopes to the preheader (red-dragon-book Ch. 9.5);
    #   SimplifyPass          -- clean up the staged hoists, fused states, and the
    #     IV-eliminated-loop placeholders before loop_to_reduce reads the body;
    #
    #   loop_to_reduce        -- lift the augmented-assignment accumulator loops
    #     to ``Reduce`` library nodes.
    #
    # AugAssignToWCR is intentionally NOT in this recipe: reductions are handled
    # via loop_to_reduce -> Reduce nodes, not WCR-on-Map. PrivatizeScalars is
    # adapted (_PrivatizeScalarsStage) so its analysis dependencies resolve.
    s += [('reduce', PatternMatchAndApplyRepeated([TrivialTaskletElimination()])),
          ('reduce', PatternMatchAndApplyRepeated([WCRToAugAssign()])), ('reduce', _PrivatizeScalarsStage()),
          ('reduce', SymbolPropagation()), ('reduce', ConstantPropagation())]
    # UntileLoops (BEFORE ShortLoopUnroll): collapse manually-tiled two-level
    # nests (``for i in range(0, N, K): for ii in range(0, K): body[i+ii]`` or
    # ``for ii in range(i, i+K): body[ii]``) back to a single ``for k in
    # range(N)``. Must run BEFORE ``ShortLoopUnroll`` because the small fixed-
    # trip inner would otherwise be straight-line-unrolled into ``K`` copies,
    # re-baking the tile into the body. Memlet audit refuses bodies whose
    # accesses don't use only ``i + ii`` / ``ii`` -- a bare reference to the
    # outer iterator alone would change semantics under collapse.
    s += [('reduce', UntileLoops())]
    if unroll_limit > 0:
        s += [('reduce', ShortLoopUnroll(unroll_limit)), ('reduce', _uniq_unroll)]
    # scalar fission (after unroll + unique-loop-iterators): unrolling and iterator
    # privatization expose transient scalars / size-1 arrays that a dominating write
    # fully redefines; fissioning them into separate containers per dominated scope
    # breaks the false write/write dependence that otherwise blocks LoopToMap and
    # confuses later value analyses. Wrapped in a Pipeline so its
    # ``ScalarWriteShadowScopes`` analysis dependency is resolved.
    s += [('reduce', ppl.Pipeline([ScalarFission()]))]
    # PromoteConstantIndexAccess + BufferExpansion: both privatize loop-carried
    # false dependences that block ``LoopToMap``. PCIA promotes ``arr[c]``
    # constant-index slot writes-then-reads on a SHARED array to a per-iteration
    # scalar; BufferExpansion adds a per-iteration dimension to a transient
    # SCRATCH buffer that is fully (re)written then read on every iteration.
    # Both run AFTER ``ShortLoopUnroll`` because unrolling can make the
    # constant-index / scratch-buffer pattern concrete (loop-variable indices
    # become literals; trip-1 trivial loops collapse). They are otherwise no-ops
    # on loops ``LoopToMap`` already accepts -- a built-in ``can_be_applied``
    # pre/post probe inside each pass leaves the SDFG untouched when the
    # privatization wouldn't unblock a refusal.
    s += [('reduce', PromoteConstantIndexAccess()), ('reduce', BufferExpansion())]
    # HoistInductionVariableUpdates runs BEFORE InductionVariableSubstitution: it
    # fissions IV-eligible updates out of compound bodies into sibling single-statement
    # loops so the IVSub matcher (which requires a single tasklet in the body) catches
    # them. Together they turn O(N) recurrences with surrounding loop work into O(1)
    # straight-line plus the surviving body. MaterializeLoopExitSymbols then handles
    # the surviving body-defined IV symbols (``k = k + step`` on an interstate edge)
    # whose final value is read after the loop: it materialises the closed-form exit
    # under a fresh ``_loop_exit_<sym>_<N>`` symbol and rewrites every downstream
    # reader so the "loop-defined symbol used after the loop" refusal disappears.
    # Reduce PREP only (the loop-lifting LoopTo* passes moved AFTER fission +
    # LoopStridePermutation -- see 'loop_to_x' below -- so the pipeline shape is
    # LoopFission -> LoopStridePermutation -> LoopToX -> LoopToMap).
    # PromoteConstInputs runs FIRST: a read-only integer scalar argument used
    # purely for indexing (e.g. a loop-stride ``inc``) is a non-transient scalar
    # the default promotion skips. Promoting it to a symbol -- and unwrapping the
    # frontend's defensive ``k + dace.int64(inc)`` cast -- lets the following
    # SimplifyPass collapse the secondary-IV update into a clean symbolic
    # ``k := k + inc`` iedge, which InductionVariableSubstitution then closes to
    # ``a[k + (i-1)*inc]`` so the strided argmax (TSVC s318) becomes liftable.
    _promote_const_inputs = ScalarToSymbolPromotion()
    _promote_const_inputs.readonly_inputs = True
    _promote_const_inputs.unwrap_integer_casts = True
    s += [('reduce', _promote_const_inputs), ('reduce', SimplifyPass()), ('reduce', HoistInductionVariableUpdates()),
          ('reduce', InductionVariableSubstitution()), ('reduce', MaterializeLoopExitSymbols()),
          ('reduce', LoopInvariantCodeMotion()), ('reduce', SimplifyPass())]

    # cascade_iedges_up (post-reduce): lift invariant interstate-edge assignments
    # (e.g. ``kfdia_plus_1 = kfdia + 1``) past every enclosing loop (all-or-nothing
    # upward, see ``CASCADE_UP_DESIGN.md``) so the later body-assigns-range-symbol
    # refuse-check sees the cleaned-up shape.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # distribute (BEFORE loop_to_symmetrize / loop_to_x): split a linear-chain loop
    # across a forward per-iteration producer->consumer dependence -- atax's two
    # matvecs coupled through ``tmp``, and covariance's per-column normalize ->
    # transpose-mirror -- so the LoopToEinsum / LoopToSymmetrize lifts below each see
    # a single-contraction / pure-copy loop. Placed here so the mirror is already its
    # own triangular loop when LoopToSymmetrize matches.
    s += [('distribute', DistributeProducerConsumerLoop())]

    # loop_to_symmetrize (BEFORE break_antidep): lift a triangular in-place
    # matrix-symmetrization nest (``for i: for j in i+1:M: X[j,i] = X[i,j]``) to a
    # ``Symmetrize`` library node whose expansion is the parallel triangular copy.
    # It runs here, before break_antidep / fission, so the in-place symmetric
    # read/write is recognised as one semantic op rather than snapshot-renamed by
    # BreakAntiDependence into a whole-matrix copy + plain map. Canonicalize does
    # NOT expand the node -- it stays a library node for codegen to lower.
    s += [('loop_to_symmetrize', LoopToSymmetrize())]

    # peel / break_antidep (optional knobs, off by default): last-resort attempts to
    # unblock loops LoopToMap would refuse, run BEFORE move_if / fission so the
    # transform sees the whole guarded loop. Peeling splits a boundary iteration off
    # and prunes the now-dead boundary guard (e.g. ``if i == 0: A[N-1] += 1``),
    # leaving a disjoint-write remainder; break_anti_dependence snapshot-renames a
    # pure read-ahead WAR (``a[i] = a[i+1]``). Both target loops that FAIL
    # parallelization (peel via a LoopToMap-can-apply pre-filter, break_antidep via
    # WAR detection), so they no-op on already-mappable loops, and both only PROBE
    # ``can_be_applied`` -- the actual LoopToMap is the 'parallelize' stage. (Loop
    # reversal is intentionally NOT a separate pass: reversing a loop only changes a
    # dependence's direction, never removes one, so it cannot make a dependent loop
    # parallelizable while preserving values -- clearing the anti-dependence does.)
    if peel_limit > 0:
        s += [('peel', BestEffortLoopPeeling(peel_limit))]
    if break_anti_dependence:
        s += [('break_antidep', BreakAntiDependence())]
    # Re-prep the freshly-unblocked loops: peel/break-antidep PROBE mappability with
    # the prep recipe but only APPLY the peel / snapshot-rename, so the peeled
    # remainder (and any body-assigned range symbol the peel introduced) still needs
    # scalar fission + symbol/constant propagation -- the same prep the reduce stage
    # ran -- before LoopToMap can map it. Only runs when a knob is enabled.
    if peel_limit > 0 or break_anti_dependence:
        s += [('peel', _PrivatizeScalarsStage()), ('peel', SymbolPropagation()), ('peel', ConstantPropagation())]

    # move_if_into_loop: push guarding conditionals into loop bodies. The genuine
    # inner imperfect nest (a bare tasklet beside an inner loop) takes the
    # free-state path: the bare sibling is wrapped in a trivial single-iteration
    # loop, spliced out again by 'untrivialize' before LoopToMap.
    s += [('move_if_into_loop', MoveIfIntoLoop())]

    # cascade_iedges_up (post-move-if): MoveIfIntoLoop may bury an invariant
    # iedge assignment inside the loop it pushed the guard into; lift it back out.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # fission: loop distribution + block-level perfect-loop-nesting. Fission clones
    # a loop into siblings that keep the same ``_loop_it_<N>`` name; re-running
    # UniqueLoopIterators disambiguates those duplicates so the later LoopToMap is
    # not blocked by a sibling appearing to read the shared iterator.
    # break_antidep (before fission): a forward-read anti-dependence -- a body that
    # reads ``a[i+1]`` off the same array a sibling statement writes at ``a[i]`` (s1244
    # ``d[i] = a[i] + a[i+1]``) -- is a cross-iteration bridge LoopFission cannot sever,
    # so the whole body stays one sequential loop. BreakAntiDependence's per-edge mixed
    # break (arrays its whole-array rename skips because a sibling read is RAW)
    # snapshots the array and redirects ONLY the read-ahead access to the snapshot
    # (same-index / read-behind reads keep their live-array RAW value), leaving just
    # per-iteration bridges for LoopFission to distribute into siblings. It runs here,
    # on the now single-compute-state body, in addition to the earlier 'break_antidep'
    # stage (which sees the loop before its slice states fuse and so only breaks the
    # whole-array pure-WAR loops). Gated on the same knob as that stage.
    if break_anti_dependence:
        s += [('fission', BreakAntiDependence())]
    s += [('fission', PerfectLoopNesting()), ('fission', _uniq_fis)]

    # normalize: dropped from the pipeline. ``NormalizeLoopsAndMaps`` rewrites
    # ``for i in b:e:s`` into ``for j in 0:(e-b)//s:1`` with body
    # ``i -> b+s*j``. In a corpus-wide measurement (TSVC, 151 kernels) this
    # rewrite blocks ``LoopToMap`` on every stride loop it touches -- L2M no
    # longer recognises ``a[b+s*j]`` as uniquely indexed by ``j`` (it expected
    # the original ``a[i]`` with stride encoded in the range), and 0 kernels
    # gained anything from running it. Net: -1 parallel map (s172). Kept the
    # standalone ``NormalizeLoopsAndMaps`` for callers that want it; just not
    # wired into the canonicalize pipeline.

    # loop_stride_permutation (after LoopFission, before every LoopTo* lift):
    # interchange a perfect loop nest so a unit-stride DOALL loop is innermost.
    # For a recurrence kernel (``aa[j,i] = aa[j-1,i] + ...`` with ``i`` the
    # unit-stride parallel axis) this turns ``for i: for j:`` into ``for j(seq):
    # for i(parallel):`` -- the inner ``i`` becomes a contiguous map and ``j``
    # stays a plain sequential loop, so NO ``Scan`` libnode (over a strided
    # apply) is needed. Soundness rests on moving only DOALL loops (a parallel
    # loop is freely interchangeable); see the pass docstring.
    s += [('loop_stride_permutation', LoopStridePermutation())]

    # fuse_consecutive_loops (right before the LoopTo* lifts): re-join a
    # hand-tiled main-body loop and its step-1 remainder -- two directly
    # consecutive, identical-bodied, unit-stride loops over adjacent index
    # ranges ``[A, B)`` then ``[B, C)`` -- into one loop over ``[A, C)``. Left
    # split, a reduction tiled this way lifts to two ``Reduce`` nodes writing
    # the same accumulator whose seed does not chain (the remainder drops the
    # main partial sum); fused, a single ``Reduce`` over the whole range lifts
    # correctly. Runs after re-roll has already collapsed the tiled main body to
    # a unit-stride single-accumulator loop, and before ``loop_to_x`` lifts it.
    s += [('fuse_consecutive_loops', FuseConsecutiveLoops())]

    # lift_copy_loops (BEFORE loop_to_x / LoopToReduce): a plain contiguous copy /
    # zero loop -- ``for i: dst[i] = src[i]`` / ``for i: dst[i] = 0`` -- is lifted to a
    # Copy / Memset library node here, before the reduction/scan detection runs, so it is
    # recognised as pure data movement instead of being mis-analysed as a (degenerate)
    # reduction or left as a naive loop. The earlier structural cleanup has already folded
    # the frontend ``AccessNode -> scalar-slice -> Tasklet`` bridge into the ``_out = _in``
    # form the detector matches. Gated on the same ``lift_copy`` knob as the post-parallelize
    # map lift below; the vectorizer (``semantic_lifting=False``) skips it so copy loops stay
    # raw loops it can lower. Structural cleanup tidies the spliced-in states afterwards.
    if semantic_lifting and lift_copy:
        s += [('lift_copy_loops', AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += _structural_cleanup('lift_copy_loops')

    # loop_to_x (moved here from the 'reduce' stage so the order is
    # LoopFission -> LoopStridePermutation -> LoopToX -> LoopToMap): lift the
    # accumulator / scan / argmax / find-index / conditional-reduce shapes that
    # LoopStridePermutation did NOT turn into a sequential-loop + parallel-map.
    # The reduce PREP (LICM / SimplifyPass / IV substitution / ...) already ran
    # above; these are the lifting passes only.
    # EarlyExitToFindIndex is NOT re-run here: it runs once in the early 'clean'
    # prep (before SplitStatements / IVS), which is the only place it is needed
    # -- the break -> find-first-index + clipped-range lift must precede those
    # stages, and re-running it in loop_to_x lifted nothing the early pass had
    # not already handled. LoopToSymmetrize likewise runs earlier (its own stage,
    # before break_antidep).
    # LoopToEinsum runs FIRST (before LoopToReduce): a contraction loop nest
    # (matvec / matmul / transpose) must be claimed as a single Einsum node before
    # LoopToReduce lifts its reduction axis to a Reduce. It probes on a throwaway
    # copy and is a clean no-op on any nest that does not collapse to one Einsum.
    s += [('loop_to_x', LoopToEinsum()), ('loop_to_x', LoopToReduce()),
          ('loop_to_x', LoopToScan(interchange_carry_with_map=interchange_carry_with_map)), ('loop_to_x', ArgMaxLift()),
          ('loop_to_x', LoopToConditionalReduce())]

    # untrivialize: splice out the single-iteration trivial-loop scaffold (the
    # wrappers MoveIfIntoLoop put around bare siblings) *while still a LoopRegion*,
    # before LoopToMap turns it into a sticky NestedSDFG.
    s += [('untrivialize', PatternMatchAndApplyRepeated([TrivialLoopElimination()]))]

    # cascade_iedges_up (pre-parallelize): re-run after fission / normalize rewrite
    # the CFG; MUST precede LoopToMap. Re-unique the iterators (ssa) so the
    # distributed siblings are independent.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp()), ('ssa', _uniq2)]

    # NOTE: MoveLoopInvariantIfUp is deliberately NOT wired here. It is the dual of
    # the earlier ``MoveIfIntoLoop`` stage, so hoisting guards back out here would
    # undo that work and ping-pong. The terminal ``hoist_guards`` stage runs it
    # once, AFTER fuse, where the fusion it would otherwise undo has happened.

    # wavefront_skew (pre-parallelize): apply the ``t = i + j, p = j`` skew to
    # 2-D perfect nests whose body has a backward dependence pattern (TSVC
    # ``s2111`` and friends). The outer ``t`` axis stays sequential (diagonal
    # sweep); the inner ``p`` axis becomes parallel and is then mapped by
    # ``LoopToMap`` in the ``parallelize`` stage. The rewrite is a relabelling
    # of the iteration space + an in-body substitution; symbolic offsets get a
    # runtime ``__builtin_trap`` non-positivity guard planted in a pre-state.
    s += [('wavefront_skew', WavefrontSkew())]

    # loop_to_scan (late, post-fission + post-skew): a second LoopToScan pass
    # catches prefix-scan recurrences that only emerged AFTER ``LoopFission``
    # isolated the recurrence statement (TSVC ``s221``:
    # ``a[i] = a[i] + c[i]*d[i]; b[i] = b[i-1] + a[i] + d[i]`` -> two fissioned
    # loops, the ``b`` loop is a clean scan). The earlier in-``reduce``
    # LoopToScan handles single-statement scan bodies that don't need fission
    # (``s242``, ``s1221``); running it again here also lifts the post-fission
    # ones without harming the already-lifted shapes.
    s += [('loop_to_scan', LoopToScan(interchange_carry_with_map=interchange_carry_with_map))]

    # parallelize: the canonical (fissioned / normalized) loops -> parallel maps.
    s += [('parallelize', PatternMatchAndApplyRepeated([LoopToMap()]))]

    # parallelize_guarded: loops that ``LoopToMap`` refused but would accept
    # permissively, where the blocker is an algebraic side condition (TSVC s171's
    # symbolic-stride in-place update ``a[i*inc] = a[i*inc] + b[i]``, injective iff
    # ``inc != 0``). Emit a runtime ``ConditionalBlock`` -- parallel Map when the
    # constraint holds, sequential loop otherwise -- rather than assuming it
    # (unsound) or leaving it as a WCR-map the vectorizer rejects. Runs BEFORE
    # ``reduction_to_wcr_map`` so the guarded loop is split out before that stage
    # would otherwise lift it to a (non-vectorizable, WCR-carrying) reduction map.
    # ``assume_parallel_guards`` drops the runtime check and lifts unconditionally
    # (caller asserts the constraint always holds); default keeps the sound guard.
    s += [('parallelize_guarded', ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards))]

    # reduction_to_wcr_map: full "scalar accumulator loop -> parallel WCR-map
    # with a true scalar accumulator" pipeline. Loops that survived parallelize
    # as multi-tasklet 'compute then accumulate' shapes (LoopToReduce keeps
    # narrow on these -- e.g. ``s += a[i] * b[i]`` for s313/vdotr, ``s += a[i]
    # * b[ip[i]]`` for the gather-sum s4115 family) become parallel WCR-maps
    # via ``AugAssignToWCR`` (frontend copy-wrapped RMW -> WCR write) +
    # ``LoopToMap``. Then the post-L2M NestedSDFG body is inlined and adjacent
    # states fused so the WCR edge is visible at the top-level MapExit, and
    # ``PrivatizeReductionAccumulator`` swaps the array-element WCR target for
    # a transient ``Scalar`` (with init + writeback states) -- the shape the
    # downstream WCR codegen can lower to a clean OMP ``reduction(op:scalar)``
    # clause. Folded into one stage because the four steps (AugWCR, L2M,
    # inline+fuse, privatize) form an atomic logical transformation.
    # FuseChainedScalarReductions FIRST: a loop that accumulates into the SAME scalar
    # more than once per iteration (TSVC s319 ``sum_val += a[i]; sum_val += b[i]``) reaches
    # here as a chained ``acc -> (+incA) -> acc -> (+incB) -> acc`` dataflow whose
    # intermediate read-back defeats the single-accumulation matcher below, leaving the
    # reduction loop sequential. Re-associating the chain into one ``acc += (incA + incB)``
    # (sound by associativity of +/*) exposes the single accumulation that
    # ``LoopToReduce(wcr-scalar)`` then lifts to a parallel WCR-map.
    s += [('reduction_to_wcr_map', FuseChainedScalarReductions())]
    s += [('reduction_to_wcr_map', LoopToReduce(prefer='wcr-scalar'))]
    s += [('reduction_to_wcr_map', PatternMatchAndApplyRepeated([LoopToMap()]))]
    # ``LoopToMap`` splits the loop body into per-iteration NestedSDFG
    # states whose intermediate scalar transients share names across
    # siblings. Running ``PrivatizeScalars`` here renames each scope's
    # transient so the downstream structural cleanup's same-name candidate
    # list is short -- defence-in-depth for the StateFusionExtended same-
    # name writer-merge guard.
    s += [('reduction_to_wcr_map', _PrivatizeScalarsStage())]
    s += _structural_cleanup('reduction_to_wcr_map')

    # scatter: ``ScatterToGuardedMaps`` inserts a runtime ``IntegerSort + WCR-summed
    # adjacent-equal collision count + post-region trap`` guard on each scatter
    # ``idx`` array, then permissively lifts the now-safe loops. Parallelizes the TSVC
    # scatter family (s491, vas, s4113) and additionally catches the cases LoopToMap
    # refused conservatively in the preceding ``parallelize`` stage (+27 maps on the
    # 151-kernel corpus: 89L/82M/3R -> 52L/109M/3R). The sink-tasklet shape that
    # previously blocked wiring is gone: the comparison map writes via WCR ``+``
    # into a ``int64`` counter, and a separate sequential ``trap_state`` reads the
    # counter as an interstate-edge-bound symbol and traps if positive -- the trap
    # tasklet has no connectors (the symbol-only convention is satisfied).
    # ``assume_parallel_guards`` skips the sort + duplicate-count guard entirely
    # and lifts each scatter unconditionally (caller asserts every idx array is a
    # permutation); default keeps the sound sort-based guard.
    if scatter_to_guarded_maps:
        s += [('scatter', ScatterToGuardedMaps(assume_no_conflicts=assume_parallel_guards))]

    # post_l2m: insert assign tasklets at map boundary, then structural
    # cleanup (state fusion + inline SDFG) -- after LoopToMap.
    s += [('post_l2m', InsertAssignTaskletsAtMapBoundary())]
    s += _structural_cleanup('post_l2m')

    # lift_copy (cleaning, post-parallelize): now that loops are maps, extract pure
    # data-movement out of them -- a contiguous element-wise copy -> a Copy library
    # node, a constant-zero write -> a Memset node (the map is fissioned first if it
    # mixes compute with data movement). This is the proper home for a unary copy /
    # transpose (einsum 'i->i' / 'ij->ji'): LiftEinsum deliberately skips those (it
    # requires >=2 tensor operands), so without this pass durbin's ``y[i] = z[i]``
    # would stay a naive map. Must run AFTER the loop->map lifting (it only matches
    # MapEntry nodes) and before the compute-map transforms / the einsum lift.
    if semantic_lifting and lift_copy:
        s += [('lift_copy', AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += _structural_cleanup('lift_copy')

    # interchange (post-parallelize, both modes): a sequential loop that survived
    # parallelize but wraps a parallel map (e.g. a recurrence sweep ``for t {
    # map[i] }``) is interchanged to ``map[i] { for t }`` so the parallel axis is
    # outer and the carry runs sequential-per-thread. Target-gated: always on GPU
    # (one kernel instead of one launch per loop iteration); on CPU only when it
    # lowers the innermost iterated stride (see MoveLoopIntoMapGated). This is the
    # loop<->map stride minimizer -- the map-only (MinimizeStridePermutation) and
    # loop-only (LoopStridePermutation) passes cannot cross that boundary. The
    # produced ``map { nsdfg { loop } }`` is flattened by the following cleanup.
    s += [('interchange', MoveLoopIntoMapGated(target=target))]
    s += _structural_cleanup('interchange')

    # TODO(perfect-nesting sift-down; GPU-oriented): a pass that turns an
    # *imperfect* nest into a perfect one so it can then be interchanged /
    # collapsed for more parallelism. Given
    #     for i: pre_body(i); for j: map_body(i, j); post_body(i)
    # sift the pre/post statements INTO the inner loop guarded by the boundary
    # iteration (the loop-region analog of the condition sift MoveIfIntoLoop
    # already does):
    #     for i: for j: { if j==0: pre_body(i); map_body(i, j); if j==M-1: post_body(i) }
    # yielding a perfect nest that MinimizeStridePermutation / MapCollapse /
    # MoveLoopIntoMapGated can then interchange. Mechanics: if the outer body is
    # not already a NestedSDFG, wrap it as one (nest_state_subgraph); then
    # collect the pre/post blocks, deep-copy them into guarded ConditionalBlocks
    # as the nested CFG's first/last blocks. Works for both Maps and LoopRegions
    # (map -> for-loop first). NOT an automatic transform for loops with real
    # cross-iteration dependencies (interchange is unsound there) -- it is a
    # parallelism-enabling rewrite, most valuable on GPU (perfect nest -> one
    # kernel), less so on CPU. Needs unit tests. See PerfectLoopNesting (the
    # data-independent statement-fission analog) for the group-analysis prior art.

    # TODO: privatize_reduction -- PrivatizeReductionAccumulator rewrites
    # WCR-on-array-element reductions to WCR-on-scalar + init + writeback so
    # the eventual WCR codegen can emit a clean ``#pragma omp parallel for
    # reduction(op:scalar)`` clause. Standalone-tested correct on s313, but
    # interacts badly with the trailing _structural_cleanup in the full
    # pipeline (StateFusion/InlineSDFG re-fuse the new init/writeback states
    # with the map state in a way that drops the map). Needs further work --
    # likely a smarter cleanup-skip mechanism for the privatize-introduced
    # states, or a different ordering w.r.t. the cleanup. Once stable:
    #
    # s += [('privatize_reduction', PrivatizeReductionAccumulator())]
    # s += _structural_cleanup('privatize_reduction')

    # reorder: permute the now-parallel map nests for unit stride (the loops
    # that were LoopToMap-eligible). Symbolic-safe: undeducible strides ->
    # no permutation.
    s += [('reorder', MinimizeStridePermutation())]

    # collapse: fold a perfect parallel map nest (``map i: { map j }``, the
    # shape maximal LoopFission leaves for a fully-parallel statement) into one
    # multi-dimensional map (``map[i, j]``). This is the canonical form for a
    # fully-parallel nest, and -- being N-dimensional -- it no longer matches a
    # sibling 1-D map for horizontal fusion, so the perfect loop nesting that
    # maximal fission produced for differently-parallel statements (e.g. a
    # parallel ``map[i, j]`` beside a carried ``map i: { loop j }``) survives
    # the fuse stage instead of being re-merged into one mixed-parallelism map.
    s += [('collapse', PatternMatchAndApplyRepeated([MapCollapse()]))]

    # fuse: first recombine adjacent identical-condition ConditionalBlocks
    # (ConditionFusion -- the inverse of branch-replicated fission, so maps
    # split across replicated guards become co-located). ConditionFusion
    # emits the full cartesian product of branch combinations; LiftTrivialIf
    # drops the provably-unsatisfiable ones (``c and not c``) so the guards'
    # maps actually co-locate. Then structural-clean so the recombined
    # branch's maps share a state, then vertical+horizontal map fusion in one
    # fixpoint (vertical priority; horizontal can expose further vertical
    # opportunities; no FindSingleUseData).
    s += [('fuse', PatternMatchAndApplyRepeated([ConditionFusion()]))]
    s += [('fuse', LiftTrivialIf())]
    s += _structural_cleanup('fuse')
    s += [('fuse', PatternMatchAndApplyRepeated([MapFusionVertical(), MapFusionHorizontal()]))]

    # normalize_map_body (post-fuse): MapFusion co-locates independent guarded
    # computations under one map but leaves each as its own NestedSDFG
    # (``map: { nsdfg1, nsdfg2 }``), which traps two same-condition guards in
    # separate control-flow graphs where ConditionFusion cannot reach them.
    # NormalizeMapBody sequences the sibling NestedSDFGs into ONE, so the guards
    # become consecutive ConditionalBlocks; the follow-up ConditionFusion folds
    # them (``if c: {s1}; if c: {s2}`` -> ``if c: {s1; s2}``), and the single
    # merged guard can then hoist out of the map at the terminal hoist_guards
    # stage. Structural cleanup tidies the spliced states.
    s += [('fuse', NormalizeMapBody())]
    s += [('fuse', PatternMatchAndApplyRepeated([ConditionFusion()]))]
    s += _structural_cleanup('fuse')

    # lift: recognize a tensor-contraction map (``map[i, k, j]: c(+)[i, j] =
    # alpha * a[i, k] * b[k, j]``) as an ``Einsum`` library node so a chain of
    # matmuls (2mm/3mm/gemm) lowers to one BLAS GEMM per contraction at finalize
    # instead of a naive WCR loop nest. Runs AFTER fuse (the contraction map is in
    # final shape; the pipeline deliberately leaves 3-input WCR contractions
    # un-reduced -- LoopToReduce refuses them -- precisely so they survive to here)
    # and BEFORE the WCR-normalization stages (LiftEinsum cancels the map's WCR and
    # folds it into the Einsum's beta, so it must precede normalize_wcr). A runtime
    # scalar coefficient (gemm's ``alpha``) is wired as the Einsum's explicit
    # ``_alpha`` connector; ``finalize_for_target`` selects the node's implementation
    # (fast BLAS if available, else a pure contraction SDFG) and codegen expands it.
    # Non-contraction maps do not match.
    # ``lift=False`` skips this optimization entirely (the matmul stays a correct WCR
    # loop nest) -- a correctness-safe escape hatch while the Einsum lowering is hardened.
    # ``semantic_lifting=False`` (set by the vectorizer) skips BOTH map->library-node
    # lifts -- this einsum lift and the lift_copy memset/memcpy above -- so the residual
    # stays as raw maps the vectorizer can lower (a library node is not vectorizable).
    if semantic_lifting and lift:
        s += [('lift', PatternMatchAndApplyRepeated([LiftEinsum()]))]

    # licm: hoist loop-invariant code (after LoopToMap, on maps).
    s += [('licm', LoopInvariantCodeMotion())]

    # hoist_guards (terminal): hoist any still-invariant guard out past every
    # enclosing loop (all-or-nothing upward, ``require_full_hoist=True``). Run
    # AFTER fuse -- the dual ``MoveIfIntoLoop`` (prep stage) has already pushed
    # guards in to enable sibling fusion, so a terminal hoist of a guard that
    # is STILL invariant w.r.t. the whole remaining loop nest does not undo
    # that fusion; it lifts the surviving config-flag guards (ICON ``istep ==
    # 1``, cloudsc ``IWARMRAIN`` etc.) to the cheapest scope. MoveLoopInvariantIfUp's
    # dead-outside-branch match lifts past per-iteration iedge assignments
    # (``start = jb // 4``), which stay in the now-guarded loop body.
    s += [('hoist_guards', MoveLoopInvariantIfUp(require_full_hoist=True))]

    # By this point fully-parallel guarded nests are collapsed maps, so the
    # surviving invariant guard sits inside a map body (not a loop) where
    # MoveLoopInvariantIfUp cannot reach it. MoveMapInvariantIfUp is its map
    # analogue / the inverse of MoveIfIntoMap: a guard whose condition does not
    # depend on the map parameters is lifted out of the map, one map copy per
    # branch (``map[i, j]: { if c: A else B }`` -> ``if c: { map[i, j]: A } else
    # { map[i, j]: B }``), so each branch is a clean unconditional parallel map.
    s += [('hoist_guards', MoveMapInvariantIfUp())]

    # normalize_wcr: WCR edges sourced from a Tasklet/NestedSDFG get an intermediate
    # private AccessNode inserted, so every WCR edge sources from an AccessNode (the
    # canonical reduction shape the downstream codegen recognises). Necessary because
    # the codegen's WCR if-branch only fires for scalar-typed CodeNode outputs:
    # vectorization-style map-body NSDFGs (pointer-typed output) would otherwise lose
    # the reduction and produce a parallel race.
    s += [('normalize_wcr', NormalizeWCRSource())]

    # revert_nonreduction_wcr: WCRs that never became a genuine reduction (left in
    # sequential loops, or injective in-place updates) go back to explicit aug-assigns;
    # WCRToAugAssign's injectivity gate keeps real in-map reductions + scatters as WCR.
    s += [('revert_nonreduction_wcr', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]

    # end: the final SimplifyPass.
    #
    # ``ArrayElimination`` is intentionally skipped: its source-merge
    # (``merge_access_nodes`` with the ``in_degree == 0`` predicate) collapses
    # two distinct source AccessNodes for the same data into one with
    # ``out_degree >= 2``. When the body of a loop carries a *different*
    # transient scalar via a read-AN-at-top / write-AN-at-bottom pair, the
    # implicit codegen ordering between those two roots is what kept the
    # carrier's RAW order intact -- merging the source breaks that ordering
    # and the carrier reads the *new* value instead of the previous one.
    # TSVC s254 / s255 (single- / two-step scalar lookback) divergence comes
    # from this exact mechanism. The Phase 2.4a guard refuses the merge when
    # the merged data itself has writes in the state, but s254's ``b`` is
    # read-only -- the carrier (``x``) is the affected transient. Until the
    # guard is widened to account for sibling carrier transients, the safer
    # disposition is to skip the pass at end-of-canonicalize. Other Simplify
    # sub-passes still run.
    #
    # relax_powers (before the terminal simplify): freeze a provable non-negative integer
    # ``base ** exp`` to the exact integer ``ipow`` on the size / subscript / bound sites WHILE
    # the loop-iterator ranges that prove the exponent non-negative are still live. This must be
    # a canonicalization pass, NOT deferred to codegen: SimplifyPass folds ``R**i * R**(K-i-1)``
    # (both exponents range-nonnegative inside the enclosing loop) into ``R**(K-1)``, and by
    # codegen that size is a persistent state-struct allocation OUTSIDE any loop -- the range
    # that proved ``K-1 >= 0`` is gone, so the codegen-time relax leaves a ``dace::math::pow``
    # (double) size that is not integral (``new complex128[...]`` -> compile error, stockham_fft).
    # (This does NOT harm ``N**2``-style sizes: freezing ``N**2 -> ipow(N, 2)`` is value-exact;
    # an earlier suspicion that it miscompiled gramschmidt was a misdiagnosis -- that was an
    # uninitialized WCR read whose layout the relax merely perturbed.)
    s += [('relax_powers', RelaxIntegerPowers())]
    s += [('end', SimplifyPass(skip={'ArrayElimination'}))]

    # Final parallelize sweep: the symbolic-stride scan specialization
    # (``LoopToScan._specialize_scan_under_stride_guard``) emits its carry-free
    # delta-build loop INSIDE the ``if stride >= 1`` ConditionalBlock branch. The
    # earlier ``parallelize`` LoopToMap stages ran before that loop settled into
    # liftable form (subsets propagated, offsets normalized by the intervening
    # simplify / symbol passes), so it survived as a residual sequential loop even
    # though it is embarrassingly parallel (``scan_strided_sym`` / ``ext_floordiv_offset``
    # / ``fission_dep_sym_offset`` -- the delta-build ``_scan_in[i-K] = x[i]``). Lift
    # any such residual now. LoopToMap only fires on genuinely parallel loops and
    # no-ops otherwise, so this cannot mis-parallelize a real carry. BEFORE
    # AssumeSymbolConstraints, which must stay the terminal stage.
    # Lift loop-carried in-place array reductions (contour_integral's
    # ``for idx: P[i, j] += X[i, j]``) to WCR writes so the terminal LoopToMap
    # parallelizes the enclosing loop. Runs post-canon (loops in fissioned /
    # normalized form) right before the terminal parallelize sweep.
    s += [('end', LiftLoopCarriedReduction())]
    s += [('end', PatternMatchAndApplyRepeated([LoopToMap()]))]

    # Terminal symbol cleanup: after fusion, a fused gather-map body carries
    # duplicate index symbols that map fusion introduced -- ``idx_index`` and
    # ``idx_index_0`` both ``idx[i]`` (``idx[i]`` computed twice). ``SymbolDedup``
    # merges provably-equal interstate-edge symbols; the following
    # ``SymbolPropagation`` + ``ConstantPropagation`` then re-fold the survivors
    # (the merge can expose fresh constant/symbol chains). BEFORE
    # AssumeSymbolConstraints, which must stay the terminal stage.
    s += [('end', SymbolDedup()), ('end', SymbolPropagation()), ('end', ConstantPropagation())]

    # NOTE: fresh WCR accumulators are identity-seeded by ``NormalizeWCRSource`` (the
    # ``normalize_wcr`` stage above), not a separate pass -- codegen never seeds a WCR
    # accumulator, so a reduction into genuinely-uninitialized scratch reads garbage. That pass
    # seeds only a provably-fresh, write-once accumulator: a transient (or an out-only nested
    # connector whose every caller binding is a transient -- never an aliased live array such as
    # gramschmidt's in-place ``__tmp_78 -> &A[j]``), with no plain initializer, whose WCR writes
    # a Map-parameter-indexed slot (so a same-slot fold that continues a live prior -- nussinov's
    # ``_priv_table`` -- is left alone). It does not attempt full cross-nested-SDFG liveness, so a
    # fresh accumulator whose WCR is already AccessNode-sourced before that pass is not covered.

    # assume_constraints (LAST): make the assumptions the pipeline relied on
    # explicit and runtime-checked, by prepending a side-effecting
    # ``__builtin_trap`` start state that aborts when one is violated -- a
    # negative signed-integer free symbol (the offset-sign nonnegativity
    # contract) or a false tracked relation (e.g. the ``K < N`` a modular-wrap
    # split leaned on). Runs AFTER every structural pass + the terminal simplify:
    # a guard prepended earlier is orphaned by any pass that builds its own entry
    # state (LoopToScan's scan-init block, reduction init, ...), which resets the
    # top-level start block and leaves the guard a disconnected source that
    # dominator analyses then KeyError on. Emitting it last -- nothing runs after
    # -- avoids that entirely while still yielding a first-state guard at codegen.
    # The external free-symbol set is unchanged by canonicalization (only loop
    # iterators are renamed, and those are bound).
    s += [('end', AssumeSymbolConstraints())]
    return s


#: A stage factory returns that stage's fresh passes, in order.
StageFactory = Callable[[], List[ppl.Pass]]


def _stage_factory(label: str) -> StageFactory:
    """Return a factory yielding fresh passes for the named stage.

    :param label: The stage label to filter the recipe by.
    :returns: A factory that builds that stage's passes in order.
    """
    return lambda: [p for lbl, p in _build_stages() if lbl == label]


#: Grouped view of :func:`_build_stages`: ``(label,
#: factory)`` per stage, in order, where ``factory()`` builds that stage's
#: fresh passes. ``_build_stages`` (flat ``(label, pass)``) is the source of
#: truth used by the pipeline; this view exists for callers that iterate
#: stage-by-stage (``for name, factory in CANONICALIZE_STAGES:
#: for unit in factory(): ...``).
CANONICALIZE_STAGES: List[Tuple[str, StageFactory]] = [(label, _stage_factory(label))
                                                       for label in dict.fromkeys(lbl for lbl, _ in _build_stages())]


def _assert_self_contained(unit: ppl.Pass):
    """Guard the empty-``pipeline_results`` invariant.

    Every unit is applied with an empty results dict, so it must either have
    no dependencies or be a self-resolving ``Pipeline`` (e.g. ``SimplifyPass``,
    or the ``Pipeline`` wrapping ``FullMapFusion``). A bare dependency-bearing
    pass placed directly in a stage would silently lose its inputs.

    :param unit: The pass about to be applied.
    :raises AssertionError: If ``unit`` has unresolved dependencies.
    """
    assert not unit.depends_on() or isinstance(
        unit, ppl.Pipeline), (f"{type(unit).__name__} has dependencies but is not a self-resolving "
                              f"Pipeline; wrap it so its depends_on() is satisfied.")


@properties.make_properties
@transformation.explicit_cf_compatible
class CanonicalizationPipeline(ppl.Pass):
    """Rewrite an SDFG into its canonical form.

    The recipe (:func:`_build_stages`) is one flat ordered list of passes
    applied once, imperatively, as ``auto_optimize`` does. A single
    :class:`~dace.transformation.pass_pipeline.Pipeline` cannot be used because
    it forbids duplicate pass types and the recipe reuses ``SimplifyPass`` and
    ``PatternMatchAndApplyRepeated`` across stages. Composites that need
    iteration iterate internally; the pipeline itself does not re-run.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after EVERY stage -- a debugging bisect aid, off by
                         default (the final ``validate`` still catches an invalid result). Set True
                         to pinpoint which stage produced an invalid SDFG.
    :param unroll_limit: Fully unroll constant-trip loops with at most this many
                         iterations (0 disables).
    :param peel_limit: Best-effort loop peeling before parallelize (0 disables;
                       off by default -- the per-loop search is expensive).
    :param break_anti_dependence: Snapshot-rename pure read-ahead anti-dependence
                                  loops before parallelize (off by default).
    :param target: ``'cpu'`` (default) or ``'gpu'``. Picks the per-target knob
                   preset (see ``_CPU_DEFAULTS`` / ``_GPU_DEFAULTS``). Any
                   explicit knob argument (e.g. ``interchange_carry_with_map=...``)
                   overrides the preset for that knob.
    :param interchange_carry_with_map: ``LoopToScan`` knob: relocate the carry
                                       ``LoopRegion`` INTO the per-column Map so
                                       the scan runs sequential-per-thread.
                                       ``None`` (default) -> per-target preset.
    :param specialize_constants: Optional ``{symbol: value}`` map (e.g. CloudSC's
                             ``{'nclv': 5}``, or a kernel's shape symbols like
                             ``{'Norb': 3}``) baked into the SDFG via
                             ``specialize_symbol`` (recursively, dropping the symbol)
                             BEFORE canonicalization -- the same specialization the
                             cloudsc parallelization pipeline does. Symbolic trip
                             counts that become concrete then unroll under
                             ``ShortLoopUnroll``, and concrete matmul extents let
                             ``canonicalize_set_fast_implementations`` pick the inlined
                             ``'pure'`` GEMM for known-small dims. ``None`` leaves every
                             symbol symbolic.
    """

    CATEGORY: str = 'Canonicalization'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(
        dtype=bool,
        default=False,
        desc='Validate the SDFG after EVERY stage (a debugging bisect aid: it pinpoints which stage '
        'produced an invalid SDFG). Off by default -- the final ``validate`` still catches an invalid '
        'result, and re-validating the whole SDFG after each of ~140 stages is a large cost on a big '
        'kernel (channel_flow: ~45s). Set True while debugging a pass regression.')
    unroll_limit = properties.Property(dtype=int,
                                       default=DEFAULT_UNROLL_LIMIT,
                                       desc='Unroll constant-trip loops <= this many iterations (0 disables).')
    peel_limit = properties.Property(dtype=int,
                                     default=4,
                                     desc='Best-effort loop peeling before parallelize (0 disables).')
    break_anti_dependence = properties.Property(
        dtype=bool, default=True, desc='Snapshot-rename read-ahead anti-dependence loops before parallelize.')
    target = properties.Property(dtype=str,
                                 default='cpu',
                                 choices=['cpu', 'gpu'],
                                 desc="Per-target knob preset selector ('cpu' or 'gpu').")
    interchange_carry_with_map = properties.Property(
        dtype=bool,
        default=True,
        desc='LoopToScan: relocate the carry LoopRegion INTO the per-column Map (on for CPU, off for GPU).')
    scatter_to_guarded_maps = properties.Property(
        dtype=bool,
        default=True,
        desc='Run ScatterToGuardedMaps in the scatter stage to lift scatter loops with a sort-based guard.')
    privatize_scatter_reductions = properties.Property(
        dtype=bool,
        default=True,
        desc='Run PrivatizeScatterReduction to surface a data-dependent scatter reduction '
        '(azimint histogram) to an OpenMP array-section reduction clause (CPU-only; off for GPU).')
    assume_parallel_guards = properties.Property(
        dtype=bool,
        default=False,
        desc='Assume every parallel-guard condition holds: ParallelizeUnderConstraint + '
        'ScatterToGuardedMaps emit only the parallel Map (no if-else fallback, no sort/trap). '
        'Unsound if a condition is violated at runtime; default keeps the sound guards.')
    lift = properties.Property(
        dtype=bool,
        default=True,
        desc='Lift tensor-contraction (matmul) maps to Einsum library nodes (False keeps them as WCR loop nests).')
    lift_copy = properties.Property(
        dtype=bool,
        default=True,
        desc='Lift contiguous copy/zero-init maps to Copy/Memset library nodes (False keeps them as maps).')
    semantic_lifting = properties.Property(
        dtype=bool,
        default=True,
        desc='Master gate for the post-LoopToMap map->library-node lifts (Einsum + Copy/Memset). '
        'False (set by the vectorizer) keeps the residual as raw maps it can lower.')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 peel_limit: Optional[int] = None,
                 break_anti_dependence: Optional[bool] = None,
                 target: str = 'cpu',
                 interchange_carry_with_map: Optional[bool] = None,
                 scatter_to_guarded_maps: Optional[bool] = None,
                 privatize_scatter_reductions: Optional[bool] = None,
                 assume_parallel_guards: bool = False,
                 specialize_constants: Optional[Dict[str, int]] = None,
                 lift: bool = True,
                 lift_copy: bool = True,
                 semantic_lifting: bool = True):
        if target not in _TARGET_DEFAULTS:
            raise ValueError(f"target must be one of {sorted(_TARGET_DEFAULTS)}; got {target!r}")
        self.validate = validate
        self.validate_all = validate_all
        self.unroll_limit = unroll_limit
        self.target = target
        # Per-target knobs: ``None`` -> preset; explicit value overrides preset.
        self.peel_limit = _resolve_target_default(target, 'peel_limit', peel_limit, fallback=4)
        self.break_anti_dependence = _resolve_target_default(target,
                                                             'break_anti_dependence',
                                                             break_anti_dependence,
                                                             fallback=True)
        self.interchange_carry_with_map = _resolve_target_default(target,
                                                                  'interchange_carry_with_map',
                                                                  interchange_carry_with_map,
                                                                  fallback=True)
        self.scatter_to_guarded_maps = _resolve_target_default(target,
                                                               'scatter_to_guarded_maps',
                                                               scatter_to_guarded_maps,
                                                               fallback=True)
        self.privatize_scatter_reductions = _resolve_target_default(target,
                                                                    'privatize_scatter_reductions',
                                                                    privatize_scatter_reductions,
                                                                    fallback=(target == 'cpu'))
        self.assume_parallel_guards = assume_parallel_guards
        self.lift = lift
        self.lift_copy = lift_copy
        self.semantic_lifting = semantic_lifting
        self._specialize_constants = specialize_constants or {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Canonicalize ``sdfg`` in place.

        :param sdfg: The SDFG to canonicalize.
        :returns: The number of passes applied.
        """
        # Specialize chosen symbols to constants first (e.g. ``nclv = 5``), so the
        # otherwise-symbolic species-loop trip counts become concrete and unroll.
        # ``specialize_symbol`` descends into nested SDFGs (and strips their
        # ``symbol_mapping``); a plain ``replace_dict`` would leave nested-SDFG
        # bodies -- the bulk of a real cloudsc build -- unspecialized.
        if self._specialize_constants:
            from dace.sdfg.utils import specialize_symbol
            for sym, val in self._specialize_constants.items():
                specialize_symbol(sdfg, sym, val)
        stages = _build_stages(unroll_limit=self.unroll_limit,
                               peel_limit=self.peel_limit,
                               break_anti_dependence=self.break_anti_dependence,
                               interchange_carry_with_map=self.interchange_carry_with_map,
                               scatter_to_guarded_maps=self.scatter_to_guarded_maps,
                               privatize_scatter_reductions=self.privatize_scatter_reductions,
                               assume_parallel_guards=self.assume_parallel_guards,
                               target=self.target,
                               lift=self.lift,
                               lift_copy=self.lift_copy,
                               semantic_lifting=self.semantic_lifting)
        for _label, unit in stages:
            _assert_self_contained(unit)
            unit.apply_pass(sdfg, {})
            if self.validate_all:
                sdfg.validate()
        if self.validate:
            sdfg.validate()
        return len(stages)


def canonicalize(sdfg: SDFG,
                 validate: bool = True,
                 validate_all: bool = False,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 peel_limit: Optional[int] = None,
                 break_anti_dependence: Optional[bool] = None,
                 target: str = 'cpu',
                 interchange_carry_with_map: Optional[bool] = None,
                 scatter_to_guarded_maps: Optional[bool] = None,
                 privatize_scatter_reductions: Optional[bool] = None,
                 assume_parallel_guards: bool = False,
                 specialize_constants: Optional[Dict[str, int]] = None,
                 lift: bool = True,
                 lift_copy: bool = True,
                 semantic_lifting: bool = True) -> SDFG:
    """Canonicalize ``sdfg`` in place and return it.

    One-call recipe analogous to ``auto_optimize``.

    :param sdfg: The SDFG to canonicalize.
    :param validate: Validate the SDFG after canonicalization.
    :param validate_all: Validate the SDFG after EVERY stage -- a debugging bisect aid, off by
                         default (the final ``validate`` still catches an invalid result). Set True
                         to pinpoint which stage produced an invalid SDFG.
    :param unroll_limit: Unroll constant-trip loops <= this many iterations (0 disables).
    :param peel_limit: Best-effort loop peeling before parallelize; ``None``
                       (default) -> per-target preset (CPU=4, GPU=4).
    :param break_anti_dependence: Snapshot-rename read-ahead anti-dependence
                                  loops (TSVC s121 shape:
                                  ``a[i] = a[i+1] + b[i]``); ``None``
                                  (default) -> per-target preset
                                  (CPU=True, GPU=True).
    :param target: ``'cpu'`` (default) or ``'gpu'``. Picks the per-target knob
                   preset (see ``_CPU_DEFAULTS`` / ``_GPU_DEFAULTS``). Explicit
                   knob args override the preset.
    :param interchange_carry_with_map: ``LoopToScan`` knob; ``None`` (default) ->
                                       per-target preset (CPU=True, GPU=False).
    :param privatize_scatter_reductions: Surface a data-dependent scatter reduction
                                   (``hist[bin[i]] (+)= w[i]`` -- the azimint histogram)
                                   to an OpenMP array-section ``reduction(op:hist[0:n])``
                                   clause so the accumulator is thread-privatised instead
                                   of hammered with a contended atomic; ``None`` (default)
                                   -> per-target preset (CPU=True, GPU=False).
    :param assume_parallel_guards: Assume every parallel-guard condition holds --
                                   ``ParallelizeUnderConstraint`` and
                                   ``ScatterToGuardedMaps`` emit only the parallel
                                   Map (no ``if cond: par else: seq`` fallback, no
                                   scatter sort/trap). Unsound if a condition is
                                   violated at runtime; ``False`` (default) keeps
                                   the sound guards.
    :param specialize_constants: Optional ``{symbol: value}`` baked in via
                             ``specialize_symbol`` (cloudsc-style, recursive into nested
                             SDFGs) before canonicalization, so symbolic trip counts
                             unroll (e.g. ``{'nclv': 5}``) and concrete matmul extents
                             (e.g. ``{'Norb': 3}``) enable the small-GEMM ``'pure'`` path.
    :param lift: Lift tensor-contraction maps (matmul chains) to ``Einsum`` library
                 nodes for BLAS lowering (default ``True``). Set ``False`` to skip
                 that optimization and keep matmuls as plain WCR loop nests -- a
                 correctness-safe escape hatch.
    :param lift_copy: Lift contiguous element-wise-copy / constant-zero maps to
                      ``Copy`` / ``Memset`` library nodes (default ``True``). Set
                      ``False`` to keep them as plain maps.
    :param semantic_lifting: Master gate for the post-LoopToMap map->library-node
                             lifts (Einsum + Copy/Memset). Default ``True``; the
                             vectorizer sets ``False`` to keep the residual as raw
                             maps (a library node is not vectorizable).
    :returns: The same ``sdfg`` instance, canonicalized.
    """
    CanonicalizationPipeline(validate=validate,
                             validate_all=validate_all,
                             unroll_limit=unroll_limit,
                             peel_limit=peel_limit,
                             break_anti_dependence=break_anti_dependence,
                             target=target,
                             interchange_carry_with_map=interchange_carry_with_map,
                             scatter_to_guarded_maps=scatter_to_guarded_maps,
                             privatize_scatter_reductions=privatize_scatter_reductions,
                             assume_parallel_guards=assume_parallel_guards,
                             specialize_constants=specialize_constants,
                             lift=lift,
                             lift_copy=lift_copy,
                             semantic_lifting=semantic_lifting).apply_pass(sdfg, {})
    # Canonicalized output opts in to OpenMP array-section reduction codegen (whole-buffer
    # WCR accumulators of a parallel map -> ``reduction(op:A[0:n])`` instead of per-element
    # atomics; complex via ``declare reduction``). Off by default elsewhere; only provably
    # contiguous cases take the clause, everything else still falls back to atomics.
    for nested in sdfg.all_sdfgs_recursive():
        nested.openmp_array_reductions = True
    return sdfg
