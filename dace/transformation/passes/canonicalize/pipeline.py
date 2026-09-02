# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline.

Rewrites an SDFG into a deterministic canonical form so later passes (fusion,
vectorization, scheduling, equivalence checks) observe one shape per
computation. See ``DESIGN.md`` for the rationale and ordering constraints.
"""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from dace import SDFG, symbolic, properties
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation import pass_pipeline as ppl

from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.simplification.prune_empty_conditional_branches import (PruneEmptyConditionalBranches)
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation.passes.canonicalize.collapse_noop_cast import CollapseNoOpCast
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.loop_to_reduce import (AccumulatorCopyChainToWCR, LoopToReduce, PinCarriedTopLevelLoops,
                                                       RetargetWCRAccumulator)
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.propagate_memlets import PropagateMemlets
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.pattern_matching import (PatternApplyOnceEverywhere, PatternMatchAndApplyRepeated)
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import (PruneUnreferencedTransients)
from dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination import (
    RedundantOrderingEdgeElimination)
from dace.transformation.passes.fusion_inline import InlineControlFlowRegions
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.canonicalize.reverse_map_traversal import ReverseMapTraversal
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary

from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy)
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.distribute_tasklet_into_map import DistributeTaskletIntoMap
from dace.transformation.dataflow.mapreduce import MapReduceFusion, MapWCRFusion
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.redundant_array import RedundantArray
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern)
from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import (
    CleanTaskletToScalarSliceToAccessNodePattern)
from dace.transformation.passes.scalar_fission import ArrayFission, PrivatizeArrays, PrivatizeScalars, ScalarFission
from dace.transformation.passes.parallelization_prep import (BestEffortLoopPeeling, ShortLoopUnroll,
                                                             DEFAULT_UNROLL_LIMIT)
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import (InductionVariableSubstitution,
                                                                                     LoopCarriedRotationSubstitution)
from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
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
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dace.transformation.passes.canonicalize.forward_store_to_load import ForwardStoreToLoad
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion
from dace.transformation.passes.canonicalize.reconstruct_wavefront_nest import ReconstructWavefrontNest
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.early_exit_to_find_index import EarlyExitToFindIndex
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.loop_to_syrk import LoopToSyrk
from dace.transformation.passes.canonicalize.loop_to_syr2k import LoopToSyr2k
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination

from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.dataflow.prune_connectors import PruneConnectors
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended


def disable_openmp_sections(sdfg: SDFG) -> None:
    """Opt ``sdfg`` and every nested SDFG out of ``#pragma omp parallel sections``.

    ``framecode.py`` wraps a state's independent components in sections whenever
    ``sdfg.openmp_sections`` is on (the property defaults to
    ``compiler.cpu.openmp_sections``), so a caller who flipped the knob globally would
    otherwise get the construct in canonicalized/vectorized output too. It is a loss there:
    the parallelism already lives in the maps, and a map inside a section re-enters OpenMP at
    nesting level 2, where the default ``OMP_MAX_ACTIVE_LEVELS=1`` hands it a team of ONE.
    Local opt-out only -- the generic codegen path stays available to every other caller.
    Call at pipeline entry AND exit: a nested SDFG minted mid-pipeline takes the property
    default, which follows the global config.
    """
    for nested in sdfg.all_sdfgs_recursive():
        nested.openmp_sections = False


def _structural_cleanup(label: str) -> List[Tuple[str, ppl.Pass]]:
    """Tidy symbols, then the state machine, between phases; never ``SimplifyPass`` mid-pipeline.

    Two phases, symbols first. The symbol phase is the established ``end``-stage quartet in its
    established order: ``SymbolDedup`` merges interstate symbols that provably hold one value,
    ``SymbolPropagation`` and ``ConstantPropagation`` re-fold the survivors (a merge exposes fresh
    chains), and ``RemoveUnusedSymbols`` prunes what folding left unreferenced -- propagation
    substitutes a value but leaves its defining name behind, so the prune belongs at the same
    boundary that creates the garbage rather than only at ``end``. Running the quartet at every
    boundary rather than one chosen point is the whole reason it is here: a consumer that compares
    two subsets SYNTACTICALLY reads two names for one address as two locations and silently
    declines, and ``AugAssignToWCR`` doing that to an indirect accumulate cost a kernel that
    ABORTED at run time. One placement only protects the consumers that happen to sit after it.

    The structural phase then decides what the states are. ``StateFusionExtended`` applies ONCE
    everywhere it matches rather than to a fixpoint: the design is cheap-per-boundary repeated
    often, not a fixpoint at each of ~15 boundaries.
    ``RedundantOrderingEdgeElimination`` runs last of all -- it is the only member that works
    inside a state, and fusing two states is precisely what turns an ordering edge that was
    load-bearing on its own into one the merged dataflow already implies; there is also no point
    reducing the edges of a state ``DeadStateElimination`` is about to delete.

    Fusion unions the interstate assignments of the states it merges, so it is itself a
    duplicate-minting producer and a duplicate it mints at one boundary is not cleaned until the
    symbol phase of the NEXT one. That is safe only while no syntactic-comparison consumer runs in
    between; ``scatter_accum_dup`` is the canary for it, and pins the WCR that a stale duplicate
    would cost.

    ``SymbolDedup`` runs TWICE, and the second one is LAST in the phase -- after the prune, not
    before it. Both facts are measured. Propagation and constant folding rewrite the assignments
    the first dedup merged, which exposes fresh equal-RHS pairs it could not have seen; and
    ``RemoveUnusedSymbols`` then DELETES assignments, which can make two previously-different
    edge sets identical and so mint merge opportunities of its own (dedup merges only symbols
    assigned on exactly the same set of edges). Closing the phase before the prune leaves those:
    7 kernels still held a mergeable pair, ``scatter_accum_dup`` among them. Closing it after the
    prune leaves none. ``SymbolDedup`` calls ``remove_symbol`` itself, so running it last costs no
    dead descriptors.

    A duplicate that survives the phase is not cosmetic: two names for one address is exactly what
    makes ``AugAssignToWCR``'s syntactic same-slot test answer "different slots", which turned an
    indirect accumulate into a guarded scatter that ``std::abort()``ed at run time. The failure is
    severe and silent, and dedup is 0.8% of canonicalize.

    Placement is deliberate and few, not every stage boundary. Cleanup is needed where a PHASE
    ends and the next one reads the graph differently, and there are four such points, plus one
    terminal:

    * ``coalesce`` (x2) -- between the opening phases. The second is not a repeat: map fusion
      rebuilds bodies as fresh single-state NestedSDFGs, and an un-inlined body hides its
      per-element memlets behind a whole-array boundary memlet (polybench seidel_2d).
    * ``lower`` -- the canonical representation is established here; every map is a LoopRegion.
    * ``loop_to_scan`` -- closes the semantic-lifting band (``lift_inv`` / ``normalize_reduction``
      / ``loop_to_symm`` / ``loop_to_scan``), before ``parallelize`` starts asking dependence
      questions of what lifting left behind.
    * ``reduction_to_wcr_map`` -- after the LoopToMap that turns the surviving loops into maps.
    * ``fuse`` (x2) -- parallelization and map fusion. The first is load-bearing in its own
      right: it is what puts the recombined branch's maps in one state for fusion to see.
    * ``end`` -- the optimization tail (terminal LoopToMap, terminal fuse, redundant-array,
      remat) is the one band whose output nothing else tidies.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, SymbolDedup()), (label, SymbolPropagation()), (label, ConstantPropagation()),
            (label, RemoveUnusedSymbols()), (label, SymbolDedup()),
            (label, PatternApplyOnceEverywhere([StateFusionExtended()])), (label, EmptyStateElimination()),
            (label, DeadStateElimination()), (label, RedundantOrderingEdgeElimination())]


def run_structural_cleanup(sdfg: SDFG) -> None:
    """Apply the between-phase structural cleanup once, to a finished SDFG.

    The recipe's own helper, for callers that run their own stage after the pipeline returns
    (:func:`~dace.transformation.passes.canonicalize.finalize.offload_to_gpu` does) and need the
    same tidy-up on the graph they were handed. One source of truth for what "structural cleanup"
    means, rather than a second copy of the list.

    :param sdfg: The SDFG to clean up in place.
    """
    for _label, unit in _structural_cleanup('structural_cleanup'):
        if isinstance(unit, PatternMatchAndApplyRepeated):
            unit.progress = False
        unit.apply_pass(sdfg, {})


def _inline_single_state(label: str) -> List[Tuple[str, ppl.Pass]]:
    """Flatten single-state NestedSDFG bodies; un-inlined, they report whole-array memlets and
    every dependence test refuses on the box (seidel_2d). ``PruneConnectors`` shares the fixpoint
    because a dead connector is a hard ``InlineSDFG`` refusal.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, PatternMatchAndApplyRepeated([PruneConnectors(), InlineSDFG()]))]


def _fold_scalar_slices(label: str) -> List[Tuple[str, ppl.Pass]]:
    """Fold the frontend scalar-slice bridge; behind the transient a matcher has no index to shift,
    which costs tsvc s252 its ``_remat`` clone and its map.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, CleanAccessNodeToScalarSliceToTaskletPattern()),
            (label, CleanTaskletToScalarSliceToAccessNodePattern())]


def _coalesce() -> List[Tuple[str, ppl.Pass]]:
    """Graph preparation for maximal map fusion, run after the first ``LoopToMap``.

    Two maps fuse only if they share a state, so everything that keeps states
    apart has to go first. The recipe removes each blocker in turn, cheapest
    and most-enabling first, because every removal exposes work for the next:

    1. ``CascadeInterstateEdgeAssignmentsUp`` -- an assignment-bearing
       interstate edge blocks ``StateFusionExtended``. Sifting the assignments
       towards the graph entry frees the edges between the compute states.
       This must re-run *here*: the earlier invocations are all pre-parallelize,
       and ``InlineMultistateSDFG`` lifts fresh assignments into the top-level
       region every time it flattens a lowered map body.
    2. ``EmptyStateElimination`` -- splices out the empty states left between
       them, merging the assignments the cascade could not lift onto the bypass
       edge (rather than letting a single assignment pin two maps apart).
    3. ``TrivialMapElimination`` -- a single-iteration map is not a parallel
       scope, only a wrapper; dropping it lifts its body to the top level where
       it can fuse with its neighbours.
    4. ``EmptyLoopElimination`` -- the loops those rewrites empty out.
    5. ``MoveIfIntoMap`` -- a guard *outside* a map keeps it in its own
       ``ConditionalBlock``, unreachable for fusion; pushing the guard in
       co-locates the map with its siblings. (``ConditionFusion``, later, only
       merges guards that are already adjacent -- it cannot push one inward.)
    6. structural cleanup -- fuse the states the steps above just freed and
       inline the nestings, so the maps genuinely share a state.
    7. ``ReverseMapTraversal`` then ``MinimizeStridePermutation`` then ``MapCollapse``
       -- BEFORE fusing, in
       that order, for two separate reasons. The permuter only walks chains of
       single-parameter maps (``_collect_perfect_nest`` breaks on a multi-param
       map and ``_reorder_nest`` needs two levels), so collapsing first would
       hide every nest it exists to reorder. And collapsing before fusing is
       what keeps differently-parallel statements apart: an N-dimensional map
       no longer matches a sibling 1-D map for horizontal fusion, so a parallel
       ``map[i, j]`` beside a carried ``map i: { loop j }`` survives instead of
       being re-merged into one mixed-parallelism map.
    8. ``DistributeTaskletIntoMap`` then ``MapFusionVertical`` / ``MapFusionHorizontal``
       -- the payoff; the first clears a free tasklet that would block the pair.
    9. ``MapCollapse`` again -- fusion can leave a freshly-perfect nest; folding
       it to one N-dimensional map is the canonical fully-parallel form.

    :returns: ``(stage_label, pass)`` pairs for the phase, in order.
    """
    s: List[Tuple[str, ppl.Pass]] = [('coalesce', CascadeInterstateEdgeAssignmentsUp()),
                                     ('coalesce', EmptyStateElimination()),
                                     ('coalesce', PatternMatchAndApplyRepeated([TrivialMapElimination()])),
                                     ('coalesce', EmptyLoopElimination()),
                                     ('coalesce', PatternMatchAndApplyRepeated([MoveIfIntoMap()]))]
    s += _inline_single_state('coalesce')
    s += _structural_cleanup('coalesce')
    # Direction before order: a reversed source loop reaches here as an ascending parameter over
    # DESCENDING addresses, and the permuter scores unit coefficients -- so orient first and it
    # scores the accesses the emitted code will actually make.
    s += [('coalesce', ReverseMapTraversal())]
    s += [('coalesce', MinimizeStridePermutation())]
    s += [('coalesce', PatternMatchAndApplyRepeated([MapCollapse()]))]
    s += [('coalesce',
           PatternMatchAndApplyRepeated([DistributeTaskletIntoMap(),
                                         MapFusionVertical(),
                                         MapFusionHorizontal()]))]
    s += [('coalesce', PatternMatchAndApplyRepeated([MapCollapse()]))]
    # 10. structural cleanup AGAIN -- map fusion rebuilds map bodies as fresh single-state
    #     NestedSDFGs, and an un-inlined body hides its precise per-element memlets behind a
    #     whole-array boundary memlet. Every downstream dependence test then reads the bounding
    #     box instead of the real subset and refuses (polybench seidel_2d: FuseLoops saw
    #     ``A[0:N, 0:N]`` where the body writes ``A[i, j+1]``). Leaving the phase tidy is this
    #     helper's stated contract; the earlier call at step 6 predates the fusion that dirties it.
    s += _inline_single_state('coalesce')
    s += _structural_cleanup('coalesce')
    return s


#: Cap on the IV-substitution / statement-fission alternation.
#:
#: TWO, because the second round is the IDEMPOTENCE CHECK, not a second attempt at optimizing.
#: A canonical form is by definition a fixpoint: the output must be a graph the round cannot
#: change again. At one round that property is assumed; at two it is observed, and the loop below
#: exits the moment the observation comes back clean.
#:
#: Measured over three corpora -- 205 kernels of tsvc, polybench and npbench, comparing loops,
#: maps, library nodes and surviving symbols -- 48 kernels reached a second round (the early exit
#: refused it for the other 157) and every one of the 48 came out identical. That is the evidence
#: the fixpoint converges at one round, which is exactly what the second round is here to confirm;
#: it is not a reason to stop confirming it. The cost is one no-op round on 23% of compiles.
_IV_SPLIT_MAX_ROUNDS = 2


@properties.make_properties
@transformation.explicit_cf_compatible
class IvSubstitutionFissionFixpoint(ppl.Pass):
    """Alternate induction-variable substitution with statement fission until neither fires.

    A FIXPOINT rather than an ordering, because the two enable each other and neither order
    dominates. ``InductionVariableSubstitution`` closes a counter that would otherwise hold a body
    together -- while an IV is live every statement reads it, so the body is one dependence
    component and no fission is legal (TSVC ``s126``). ``SplitStatements`` in turn produces the
    single-statement bodies the IV matcher requires, which is the entire reason
    ``HoistInductionVariableUpdates`` exists. At any FIXED depth one of the two is left with work
    it could only have done after the other, so the alternation is iterated to a fixpoint instead.

    The prep stays INSIDE the round on purpose. ``s318`` (``k += inc``, ``inc`` an argument) and
    ``s453`` both decline until ``ScalarToSymbolPromotion`` has turned the read-only argument into
    a symbol, so an IV pass placed ahead of the promotion is a no-op that costs compile time and
    finds nothing.

    Stops as soon as a round changes nothing, which is what makes it a fixpoint rather than a
    fixed sequence and doubles as the idempotence check: on a graph the previous round settled,
    the next must be a provable no-op. Measured on a settled graph every member reports no
    change, so the early exit is reachable and not dead code.
    """

    CATEGORY: str = 'Canonicalization'

    max_rounds = properties.Property(dtype=int,
                                     default=_IV_SPLIT_MAX_ROUNDS,
                                     desc='Cap on alternation rounds; the loop breaks earlier when '
                                     'a round changes nothing.')

    def __init__(self, max_rounds: int = _IV_SPLIT_MAX_ROUNDS) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def round_units(self) -> Tuple[ppl.Pass, ...]:
        """The passes one round runs, in order.

        A method rather than a literal buried in ``apply_pass`` because this round OWNS a
        ``SimplifyPass``: a recipe invariant that asks where simplification happens can only be
        checked if the composite says what it contains.
        """
        promote = ScalarToSymbolPromotion()
        promote.transients_only = False
        return (promote, SimplifyPass(), HoistInductionVariableUpdates(), InductionVariableSubstitution(),
                SplitStatements())

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the round up to ``max_rounds`` times, stopping early on a no-op round.

        :returns: the number of rounds actually run, or ``None`` if the first round did nothing.
        """
        units = self.round_units()
        rounds = 0
        for _ in range(self.max_rounds):
            # Each pass reports differently -- a count, a set of promoted names, a results dict --
            # so test each for truthiness rather than summing them.
            changed = False
            for unit in units:
                if unit.apply_pass(sdfg, {}):
                    changed = True
            rounds += 1
            if not changed:
                break
        return rounds or None


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
        return {}

    def privatizer(self) -> ppl.Pipeline:
        """The privatization pipeline this stage adapts.

        :returns: The self-resolving ``Pipeline`` to apply.
        """
        return PrivatizeScalars()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Any]:
        # ``PrivatizeScalars`` resolves a ``FindAccessNodes`` analysis (keyed by
        # ``cfg_id``) and a reachability analysis that calls ``reset_cfg_list`` mid-
        # pipeline; a stale control-flow-region list (left by a prior stage's inliner)
        # then lets that reset reassign ``cfg_id`` under the cached ``FindAccessNodes``
        # result -> ``KeyError``. Refresh the list up front so both analyses agree.
        sdfg.reset_cfg_list()
        return self.privatizer().apply_pass(sdfg, {})


@properties.make_properties
class _PrivatizeArraysStage(_PrivatizeScalarsStage):
    """Array sibling of :class:`_PrivatizeScalarsStage`, paired with it everywhere it runs.

    A transient ARRAY reused as a per-iteration scratch buffer carries the same false
    write/write dependence as a reused scalar, so it needs privatizing at the same points of
    the recipe. The difference is the premise: versioning per dominating write is only
    value-preserving when that write is a must-def of the WHOLE container, which is free for a
    scalar and a proof obligation for an array. ``ArrayWriteShadowScopes`` discharges it and
    reports nothing it cannot prove, so the scalar path never becomes a fallback for an
    unproven array.
    """

    def privatizer(self) -> ppl.Pipeline:
        return PrivatizeArrays()


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
# ``reconstruct_wavefront_nest`` (ReconstructWavefrontNest): rebuilds an imperfect
# Map-plus-LoopRegion stencil body (polybench ``seidel_2d``'s slice-vectorized neighbor
# sum beside its sequential in-row scan) into the single ``LoopRegion`` ``WavefrontSkew``
# requires, committing only when a deepcopy trial proves it unlocks a skew. Off by default
# on both targets: on the corpus kernel it targets, the Map's slice-normalized range and the
# scan's direct-index range do not line up (a real offset, not just a DOALL refusal), so the
# reconstruction never actually fires there yet -- ON is safe (mutate-on-provable-win only)
# but unproven to help; flip once a corpus win is measured.
# ``normalize_loop_and_map_origin`` (NormalizeLoopAndMapOrigin): rebase every Map range /
# ``LoopRegion`` counter to a 0-based begin, KEEPING the stride (``b:e:s`` -> ``0:(e-b):s``,
# ``p -> p + b``) -- unlike ``NormalizeLoopsAndMaps`` (dropped from this pipeline, see the
# ``normalize`` comment below), which folds the stride into the index and cost a parallel map
# on TSVC ``s172``. Two ranges of equal trip count but different origin (a slice-vectorized
# Map ``0:N-3`` beside a sequential scan ``LoopRegion`` ``1:N-2``, polybench ``seidel_2d``)
# become the identical ``0:N-3`` once both are rebased. Corpus verdict (transform-only,
# ``tests/corpus/measure_parallelization.py``, 4 corpora): see the AB note on the knob's wiring
# below for the measured totals and default.
_CPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': True,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': True,
    'reconstruct_wavefront_nest': False,
    'normalize_loop_and_map_origin': False,
}
_GPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': False,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': False,
    'reconstruct_wavefront_nest': False,
    'normalize_loop_and_map_origin': False,
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
                  reconstruct_wavefront_nest: bool = False,
                  normalize_loop_and_map_origin: bool = False,
                  assume_parallel_guards: bool = False,
                  perfect_loop_nesting: bool = True,
                  iv_split_rounds: int = _IV_SPLIT_MAX_ROUNDS,
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
    :param reconstruct_wavefront_nest: Rebuild an imperfect Map-plus-LoopRegion stencil
                                       body into the single loop ``WavefrontSkew`` requires
                                       (``ReconstructWavefrontNest``), right before it in the
                                       ``loop_fuse`` stage; commits only on a proven skew.
                                       Off by default on both targets (see ``_CPU_DEFAULTS``).
    :param normalize_loop_and_map_origin: Rebase every Map range / ``LoopRegion`` counter to a
                                          0-based begin, keeping the stride
                                          (``NormalizeLoopAndMapOrigin``), right before the
                                          ``loop_to_x`` stage -- BEFORE every ``LoopTo*`` lift so
                                          they see the normalized shape. Off by default on both
                                          targets (see ``_CPU_DEFAULTS``).

    Every map is lowered to a ``LoopRegion`` up front so all canonicalization
    runs on a single representation (one fission/normalize/reduce path, no
    map/loop duplication, no hybrids); ``LoopToMap`` recovers parallelism near
    the end, then maps are fused. Returns ``(stage_label, pass)`` pairs with
    fresh instances each call.

    ``SimplifyPass`` runs at the very start, after the cleaning passes (unique
    loop iterators, split tasklets, trivial-tasklet cleanup), and twice in the
    ``reduce`` stage around ``ShortLoopUnroll`` to collapse the redundant
    straight-line code an unroll produces -- never otherwise, and never after
    ``reduce``. Between-stage structural cleanup is ``StateFusionExtended`` +
    ``InlineSDFG`` instead; every stage past ``reduce`` therefore has to stand on
    its own on un-simplified input.
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

    # clean: unique loop iterators -> split tasklets -> the leading SimplifyPass
    # (only here and, twice, in 'reduce'). Trivial-tasklet elimination now opens the
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

    # lift_inv (semantic lift, on the raw frontend shape, gated like loop_to_symm):
    # recognise ``A^-1`` written as ``solve(A, eye(N))`` -- a ``Solve`` library
    # node whose RHS operand is a freshly-built identity matrix
    # (``numpy.eye`` / ``numpy.identity``: a square ``val = 1 if i == j else 0``
    # mapped tasklet) -- and replace the ``Solve`` + identity-construction map
    # with a single ``Inv`` node (getrf + getri, the direct inverse that never
    # materialises an identity RHS). It runs BEFORE ``MapToForLoop`` lowers the
    # identity map to a loop and before the ITE-lowering passes rewrite its
    # ``1 if ... else 0`` body, so the identity is still its raw mapped-tasklet
    # form. A strict no-op on any deviation (non-identity RHS, shifted diagonal,
    # a reused identity), like the other semantic lifts.
    if semantic_lifting and lift:
        s += [('lift_inv', LiftInv())]

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
    # WCRToAugAssign keeps the scalar WCR and MapToForLoop's map-exit-WCR refusal keeps it a
    # parallel map, so it is neither severed nor double-counted. (This used to also credit
    # MapFusionVertical's seeded-reduction guard, which is dead code -- see its docstring.)
    # Idempotent, so the vectorizer can also run it standalone.
    s += [('normalize_reduction', NormalizeWCR())]

    # A loop with a ``break`` / ``continue`` is not splittable and its induction variable
    # is not closed-form (the trip count is data-dependent), so SplitStatements / IVS below
    # cannot handle it. Lift the early exit to a find-first index + clipped range HERE --
    # before those stages -- so they only ever see the resulting break-free, clipped loop.
    #
    # ``CollapseNoOpCast`` leads the block: a Fortran kind coercion / numpy ``astype`` lands as
    # ``__out = dace.float64(__inp)`` where ``__out`` ALREADY has that dtype. Every downstream
    # matcher reads tasklet bodies textually, so the redundant call hides an otherwise-plain
    # assignment from all of them -- and from ``TrivialTaskletElimination`` inside the
    # ``SimplifyPass`` five entries below, whose predicate is the exact string
    # ``out = in`` (``trivial_tasklet_elimination.py:85``) and which separately refuses any
    # endpoint dtype mismatch, so it can never collapse the cast itself. Rewriting to the plain
    # assignment first is what lets that Simplify fold the copy away in the same block. A
    # genuine cast (differing dtypes) fails the pass's own equality check and is kept.
    s += [('clean', CollapseNoOpCast()), ('clean', RewriteModuloToPyMod()), ('clean', NormalizeNegativeStride()),
          ('clean', _uniq), ('clean', ContinueToCondition()), ('clean', EarlyExitToFindIndex()),
          ('clean', SimplifyPass()), ('clean', PatternMatchAndApplyRepeated([StateFusionExtended()]))]

    # loop_to_syrk / loop_to_syr2k (semantic lift, gated like loop_to_symm): the
    # hand-written symmetric rank-k / rank-2k update nests (polybench syrk / syr2k) are
    # recognised -- an outer row loop whose body beta-scales a triangular row slice of a
    # square C and then accumulates onto that same slice in an inner k loop -- and
    # replaced by a ``Syrk`` / ``Syr2k`` BLAS node (vendor dsyrk / dsyr2k, which compute
    # only the referenced triangle: half the flops of the equivalent gemm, and threaded).
    #
    # They run HERE, right after the clean block's SimplifyPass + StateFusionExtended,
    # because -- unlike loop_to_symm, which matches a raw frontend map+NestedSDFG
    # boundary -- these match the *dataflow expression* of a single fused body state.
    # The frontend spreads each slice statement over several states and staging
    # temporaries; StateFusionExtended collapses that into the one-state-per-loop-body
    # shape the recogniser resolves through. They must still run BEFORE 'prep'
    # (SplitStatements) and 'lower' (MapToForLoop) rewrite the body.
    #
    # A strict, no-op-on-any-deviation match (gated on the semantic-lifting knobs, like
    # LiftEinsum / LoopToSymm), so the vectorizer path (semantic_lifting=False) leaves
    # syrk / syr2k as plain reduction nests.
    # LoopToSymm runs a SECOND time here, for the same reason: its npbench slice form
    # (``C[:i, j] += alpha*B[i, j]*A[i, :i]`` plus a ``B[:i, j] @ A[i, :i]`` inner product --
    # the spelling the corpus carries) is a two-level LoopRegion nest whose body statements
    # the frontend spreads over several states, so it only becomes matchable once the clean
    # block's StateFusionExtended has collapsed each body to one state. The earlier
    # 'loop_to_symm' entry stays where it is: the polybench MAP form it matches must be seen
    # before normalize_reduction rewrites that boundary WCR. Each form is a clean no-op for
    # the other, so running the pass at both points lifts whichever spelling is present.
    if semantic_lifting and lift:
        s += [('loop_to_symm', LoopToSymm()), ('loop_to_syrk', LoopToSyrk()), ('loop_to_syr2k', LoopToSyr2k())]

    # prep (still maps): push guarding conditionals into maps, then split
    # statements -- replicate a conditional / gather-scatter NestedSDFG per
    # independent output so it can fission later (SplitStatements subsumes the
    # former ConditionalComponentFission and also handles forward-read anti-deps).
    #
    # ConvertLengthOneArraysToScalars leads the stage: the frontend spells a scalar temporary as a
    # ``(1,)`` transient Array, and every consumer downstream of here -- the statement split's
    # dependency walk, fission, WCR handling -- keys on the descriptor, so the two spellings take
    # different paths for the same value. Normalize to Scalar first. TRANSIENTS ONLY
    # (``preserve_abi`` left clear): a signature-level length-1 array is the caller's contract and
    # is not touched.
    #
    # ForwardStoreToLoad runs immediately before the split, and only there. A body that stores
    # ``a[i]`` and reads it back at the same ``i`` (TSVC ``s323``) makes ``a`` cross between the
    # two statements, and with a second array carried the other way the split has no provable
    # order and refuses. Feeding the in-iteration reader from the stored value -- the store to
    # ``a`` stays -- leaves one crossing array, which the split does order; downstream that is
    # what lets ``LoopToScan`` see the prefix sum. Later is too late (the split has already
    # refused) and earlier buys nothing.
    # Define the reserved thread-count symbol before anything can want it. A pass rather than
    # frame code so it is visible in the IR, inherited by nested SDFGs through symbol_mapping,
    # and emitted by whatever already emits tasklets -- including the standalone MPR frame.
    s += [('prep', SupplyNumThreads())]
    s += [('prep', PatternMatchAndApplyRepeated([MoveIfIntoMap()])), ('prep', ConvertLengthOneArraysToScalars()),
          ('prep', ForwardStoreToLoad()), ('prep', SplitStatements())]
    # Distribute first: the split removes the anti-dependence where reader and writer separate.
    if break_anti_dependence:
        s += [('prep', BreakAntiDependence(forward_reads=True))]

    # WCRToAugAssign BEFORE lower: rewrite every conflict-free (injective) WCR back to
    # an explicit RMW while maps are still maps; what stays WCR is a genuine reduction
    # that MapToForLoop then refuses to lower (kept parallel -> OMP reduction), so the
    # in-state producer->consumer edge is never severed by the map->loop round-trip.
    s += [('lower', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]
    # lower: every map -> LoopRegion (MapToLoop = reuse MapToForLoop), then
    # structural cleanup (no SimplifyPass).
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True  # canon preference, off in the transformation's default contract
    s += [('lower', PatternMatchAndApplyRepeated([lower_maps]))]
    # The pipeline's only ``InlineMultistateSDFG``: lowering mints the nestings here.
    s += [('lower', PatternMatchAndApplyRepeated([PruneConnectors(), InlineMultistateSDFG(), InlineSDFG()]))]
    s += _fold_scalar_slices('lower')
    s += _structural_cleanup('lower')
    # MapToForLoop leaves empty *_pre_state / *_post_state boundary states;
    # inside a guard branch they make the body look like a heterogeneous
    # [empty, empty, loop] chain and send MoveIfIntoLoop down its imperfect
    # path to wrap *empty* states. Splice them out so the guarded body is the
    # bare loop -> MoveIfIntoLoop's clean single-loop path applies.
    s += [('lower', EmptyStateElimination())]

    # NormalizeNegativeStride again, now that every map is a LoopRegion. The
    # pass only ever rewrites LoopRegions, so the earlier 'clean' invocation
    # reached the loops the frontend emitted but not the maps -- a negative-
    # stride Map became a negative-stride loop here, after its only chance to
    # be normalized. Downstream (RerollUnrolledLoops, LoopToScan's stride != 1
    # refusal, LoopToMap's affine subset classifier) all run past this point
    # and do rely on the positive-stride invariant.
    s += [('lower', NormalizeNegativeStride())]

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
          ('reduce', _PrivatizeArraysStage()), ('reduce', SymbolPropagation()), ('reduce', ConstantPropagation())]
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
    # array fission, immediately after: the same false write/write dependence exists on a
    # transient ARRAY reused as a per-iteration scratch buffer (``Tz = np.zeros(...)`` at the top
    # of a loop body). Versioning it is only value-preserving when the dominating write covers the
    # WHOLE array -- for a scalar that is free, for an array it is a proof obligation that
    # ``ArrayWriteShadowScopes`` discharges; anything it cannot prove is left alone. Kept a
    # separate stage rather than widening ``ScalarFission``, so the scalar path never becomes a
    # fallback for an unproven array.
    s += [('reduce', ppl.Pipeline([ArrayFission()]))]
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
    # PromoteConstInputs runs FIRST so any promotable secondary-IV scalar becomes
    # a symbol before the following SimplifyPass collapses the update into a clean
    # ``k := k + inc`` iedge, which InductionVariableSubstitution then closes to
    # ``a[k + (i-1)*inc]`` (the strided-argmax lift).
    s += [('reduce', IvSubstitutionFissionFixpoint(max_rounds=iv_split_rounds))]
    # MaterializeLoopExitSymbols runs AFTER the rounds, not between IVS and LICM as it used to:
    # it materialises the exit value of whatever counters SURVIVE, so it wants the settled graph.
    s += [('reduce', MaterializeLoopExitSymbols()), ('reduce', LoopInvariantCodeMotion()), ('reduce', SimplifyPass())]

    # cascade_iedges_up (post-reduce): lift invariant interstate-edge assignments
    # (e.g. ``kfdia_plus_1 = kfdia + 1``) past every enclosing loop (all-or-nothing
    # upward, see ``CASCADE_UP_DESIGN.md``) so the later body-assigns-range-symbol
    # refuse-check sees the cleaned-up shape.
    # The frontend promotes a computed index (``i * inc``, ``i + M``) to a scalar and then to an
    # interstate symbol used in the subset (``a[__sym_i_times_inc]``). ``SymbolPropagation``
    # deliberately does NOT substitute a loop-variable RHS back into the graph (``replace_dict``
    # sizes descriptors), so the opaque symbol reaches every structural matcher below --
    # ``expr.coeff(loop_var)`` reads 0, the loop variable is "absent", and LoopToMap / LoopToScan
    # / ParallelizeUnderConstraint all decline a loop that is plainly parallel. Recover the direct
    # arithmetic here, before the lifting stages; genuine data-dependent gathers (``a[idx[i]]``)
    # are left alone. ``RemoveUnusedSymbols`` then sweeps the now-dead promotion symbols.
    s += [('index_subsets', PropagateIndexSubsets()), ('index_subsets', RemoveUnusedSymbols())]
    # ``PropagateIndexSubsets`` exposes fresh loop-invariant arithmetic on
    # interstate edges; without a following propagation/cleanup the
    # ``reduction_to_wcr_map`` stage can build WCR memlet subsets that reference
    # nested connector arrays as if they were symbols and then inline the
    # nested SDFG, leaving free ``__tmp_*`` symbols behind
    # (``split_tasklets_test::test_add_missing_symbols_honors_integer_cast``).
    # This narrow cleanup (not a full ``SimplifyPass``) folds those expressions
    # and removes the now-dead dataflow.
    s += [('index_subsets',
           ppl.FixedPointPipeline([SymbolPropagation(),
                                   ConstantPropagation(),
                                   DeadDataflowElimination()]))]

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
        # rotate (BEFORE peel): a loop-carried DELAY LINE (``x = b[i]`` read one iteration later)
        # has no ``a*i + b`` write subset, so LoopToMap refuses a whole loop over a carry whose
        # value is just a shifted array element. Substituting the shifted read and deleting the
        # update makes the body DOALL; the one iteration the shifted read does not cover is peeled
        # off the front, which is why this shares ``peel_limit`` -- and its "runs more often than we
        # peel it" assumption -- instead of owning a knob. Runs before ``BestEffortLoopPeeling`` so
        # the peel search sees the already-unblocked loop and nominates nothing further for it.
        s += [('rotate', LoopCarriedRotationSubstitution(peel_limit))]
        s += [('peel', BestEffortLoopPeeling(peel_limit))]
    if break_anti_dependence:
        s += [('break_antidep', BreakAntiDependence())]
    # Re-prep the freshly-unblocked loops: peel/break-antidep PROBE mappability with
    # the prep recipe but only APPLY the peel / snapshot-rename, so the peeled
    # remainder (and any body-assigned range symbol the peel introduced) still needs
    # scalar + array fission and symbol/constant propagation -- the same prep the reduce
    # stage ran -- before LoopToMap can map it. Only runs when a knob is enabled.
    if peel_limit > 0 or break_anti_dependence:
        s += [('peel', _PrivatizeScalarsStage()), ('peel', _PrivatizeArraysStage()), ('peel', SymbolPropagation()),
              ('peel', ConstantPropagation())]

    # move_if_into_loop: push guarding conditionals into loop bodies. The genuine
    # inner imperfect nest (a bare tasklet beside an inner loop) takes the
    # free-state path: the bare sibling is wrapped in a trivial single-iteration
    # loop, spliced out again by 'untrivialize' before LoopToMap.
    #
    # Removal probed, NOT proven safe: dropping this standalone run leaves the tsvc
    # scorecard identical (223 kernels, 35 loops / 226 maps, per-kernel) and adds no
    # failure to the guard / branchy-polybench suites -- but those already carry two reds
    # at d2be1fde4 (floyd-warshall's single-nest lift, the coexisting-index-guards
    # collapse), so they cannot witness a regression here. PerfectLoopNesting does not
    # subsume it: that pass never runs MoveIfIntoLoop.
    s += [('move_if_into_loop', MoveIfIntoLoop())]

    # cascade_iedges_up (post-move-if): MoveIfIntoLoop may bury an invariant
    # iedge assignment inside the loop it pushed the guard into; lift it back out.
    #
    # Collapse to one run probed, NOT proven safe: dropping this and the post-reduce run
    # (keeping only the pre-parallelize one) leaves the tsvc scorecard untouched
    # (35 loops / 226 maps, per-kernel identical) and the npbench + polybench numerical
    # corpus green. Kept anyway: the post-reduce run exists for the body-assigns-range-symbol
    # refuse-check that runs between it and parallelize, and both comments cite cloudsc
    # shapes that no gate here covers.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # fission: loop distribution + block-level perfect-loop-nesting. Fission clones
    # a loop into siblings that keep the same ``_loop_it_<N>`` name; re-running
    # UniqueLoopIterators disambiguates those duplicates so the later LoopToMap is
    # not blocked by a sibling appearing to read the shared iterator.
    # break_antidep (before fission): a second whole-array pure-WAR rename, on the now
    # single-compute-state body. It runs in addition to the earlier 'break_antidep'
    # stage, which sees the loop before its slice states fuse. The per-edge MIXED shape
    # -- a body that reads ``a[i+1]`` off the same array a sibling writes at ``a[i]``
    # (s1244 ``d[i] = a[i] + a[i+1]``), which the whole-array rename skips because that
    # sibling read is RAW -- is handled earlier, by SplitStatements in 'prep'. Gated on
    # the same knob as that stage.
    if break_anti_dependence:
        s += [('fission', BreakAntiDependence())]
    # PerfectLoopNesting runs BY DEFAULT (user ruling 2026-09-01), reversing the 2026-08-18
    # ruling that kept it opt-in. Both rulings are about the same defect, and it is worth
    # recording precisely because the reversal rests on it being gone:
    #
    # THE DEFECT: the pass used to distribute via ``LoopFission``'s node-level grouping, which
    # carries no dependence distance or direction. A SINGLE application reproduced the CloudSC
    # read-modify-write miscompile bit-for-bit (tendency_loc_a rel=0.13, measured 2026-08-18 on
    # the phase-17 staged snapshot) -- the fault was that grouping itself, not a composed fixpoint.
    #
    # WHAT CHANGED: PerfectLoopNesting no longer calls that grouping at all; the code path named
    # above is severed. It now groups with Allen-Kennedy at BLOCK granularity (SCC-equivalent,
    # emitted in program order) and checks per-parent-iteration disjointness, refusing whatever it
    # cannot pin -- so a group it cannot prove independent is left fused rather than split blind.
    #
    # WHAT IS NOT YET ESTABLISHED: the CloudSC numerics gate has NOT been re-run against the
    # rebuilt pass. The reversal rests on the offending grouping being gone, not on a fresh
    # measurement. Re-run tendency_loc_a before treating the miscompile as closed.
    #
    # ``canonicalize_mixed_parallelism_test`` exercises the collapsed-2D-map contract here.
    if perfect_loop_nesting:
        s += [('fission', PerfectLoopNesting(target=target))]
    s += [('fission', _uniq_fis)]

    # untrivialize: splice out the single-iteration trivial-loop scaffold (the
    # wrappers MoveIfIntoLoop put around bare siblings) *while still a LoopRegion*,
    # before LoopToMap turns it into a sticky NestedSDFG.
    #
    # Runs HERE, right after fission, not just before LoopToMap: every matcher
    # between this point and LoopToMap expects a loop body to be one SDFGState
    # and refuses a body that is a LoopRegion. ``LoopStridePermutation``'s
    # ``_perfect_nests`` would absorb the wrapper as a real nest level and
    # bubble it as if it were an axis; ``AssignmentAndCopyKernelToMemsetAndMemcpy``
    # and the ``LoopTo*`` lifts refuse outright on ``not isinstance(blocks[0],
    # SDFGState)``. Leaving the scaffold in place through those stages silently
    # disabled them. Fission runs first because the fission stage needs the
    # uniform all-siblings-are-loops shape the scaffold provides.
    s += [('untrivialize', PatternMatchAndApplyRepeated([TrivialLoopElimination()]))]

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
    # Copy / Fill library node here, before the reduction/scan detection runs, so it is
    # recognised as pure data movement instead of being mis-analysed as a (degenerate)
    # reduction or left as a naive loop. The earlier structural cleanup has already folded
    # the frontend ``AccessNode -> scalar-slice -> Tasklet`` bridge into the ``_out = _in``
    # form the detector matches. Gated on the same ``lift_copy`` knob as the post-parallelize
    # map lift below; the vectorizer (``semantic_lifting=False``) skips it so copy loops stay
    # raw loops it can lower. The spliced-in states are tidied at the next cleanup phase
    # (``lower``), not here -- see :func:`_structural_cleanup` for why the phases are few.
    if semantic_lifting and lift_copy:
        s += [('lift_copy_loops', AssignmentAndCopyKernelToMemsetAndMemcpy())]

    # normalize_origin (optional knob, off by default): rebase every Map range / LoopRegion
    # counter to a 0-based begin, keeping the stride -- the LAST point before every ``LoopTo*``
    # lift (loop_to_x / loop_to_scan / parallelize), so those matchers see the normalized shape.
    # No Map exists in the SDFG yet at this point ('lower', above, already turned every one into
    # a LoopRegion, and no stage between there and here reintroduces one), so this only ever
    # rebases LoopRegions here -- the Map-rebasing half of the pass matters for a caller that
    # invokes it standalone, or once ``LoopToMap`` (below) creates fresh Maps.
    if normalize_loop_and_map_origin:
        s += [('normalize_origin', NormalizeLoopAndMapOrigin())]

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
    # LoopToConditionalReduce folds a guarded accumulator ``if cond: acc OP= x`` into an
    # UNCONDITIONAL masked reduction -- ``masked = (x if cond else IDENTITY); acc OP= masked``
    # -- the exact "compute then accumulate" shape an unguarded reduction (``s += a[i]*b[i]``)
    # has -- so the later ``reduction_to_wcr_map`` stage lifts it to a parallel WCR-on-scalar
    # Map whose codegen emits the OpenMP ``reduction(op:acc)`` clause (CPU) / block-warp
    # tree-reduce (GPU), instead of the per-passing-thread guarded atomic the raw conditional
    # lowers to.
    # LoopToStreamCompaction runs LAST in loop_to_x, after every reduction-shaped lift has had its
    # refusal: a guarded ``if cond: acc OP= x`` is a masked REDUCTION and belongs to
    # LoopToConditionalReduce, whereas this pass claims the complementary shape where the carried
    # scalar is the write INDEX (``if cond: j += 1; a[j] = f(i)``), which no other lift matches.
    # It must precede ``parallelize``: it emits three loops -- mask, scan, scatter -- and force-lifts
    # the two parallel ones itself (it owns the disjointness proof for the cursor-indexed writes;
    # LoopToMap cannot reconstruct it), leaving the residue for the later stages to fuse and
    # schedule. It must also precede ``reduction_to_wcr_map``, whose ``PinCarriedTopLevelLoops``
    # can pin a loop this pass would claim, and a pinned loop is refused here (correctly: the pin
    # is a directive, not an obstacle).
    # ``LoopToReduce`` / ``LoopToScan`` are pure matchers -- they touch nothing when they lift
    # nothing -- so the body normalization each one needs is spelled out here: WCR-to-augassign
    # for the reduction matcher, the full ``LiftPreprocess`` (which adds copy-tasklet folding,
    # negative-stride flipping and body-state fusion) for the scan matcher. Order matters: the
    # extra folds must NOT precede ``LoopToReduce``, whose single-tasklet matcher claims the
    # un-folded augassign shape.
    # loop_to_transpose (semantic lift, gated like the other LoopTo* lifts): a hand-written
    # tensor permutation ``B[i, j] = A[j, i]`` -- a perfect nest whose innermost body is a PURE
    # copy at permuted point subscripts -- becomes a ``Transpose`` (2-D, BLAS omatcopy) or
    # ``TensorTranspose`` (N-D, HPTT / cuTENSOR) node.
    #
    # Nothing else in the recipe covers this shape, which is why it gets its own slot:
    # ``LiftEinsum`` refuses a single tensor operand outright ("a unary copy / transpose /
    # reduction", ``lift_einsum.py:120``), and ``AssignmentAndCopyKernelToMemsetAndMemcpy``
    # deliberately REJECTS permutations -- the tasklet body ``_out = _in`` is identical for a
    # copy, a broadcast and a transpose, so it discriminates on subscript order and hands the
    # permuted case back (``assignment_and_copy_kernel_to_memset_and_memcpy.py:99-109``).
    # Left unlifted, a transpose nest stays a strided element-wise copy.
    #
    # It runs FIRST in the block, ahead of ``LoopToEinsum``: both read a perfect nest of point
    # subscripts, and the transpose matcher is the strictly narrower one (pure copy, one
    # operand, bijective permutation), so letting it claim its shape first costs the einsum
    # matcher nothing -- an einsum needs >= 2 operands and refuses this shape anyway.
    # Re-fold: the stages since 'lower' mint fresh bridges (tsvc s254).
    s += _fold_scalar_slices('loop_to_x')
    if semantic_lifting and lift:
        s += [('loop_to_x', LoopToTranspose())]
    s += [('loop_to_x', LoopToEinsum()), ('loop_to_x', PatternMatchAndApplyRepeated([WCRToAugAssign()])),
          ('loop_to_x', LoopToReduce()), ('loop_to_x', LiftPreprocess()),
          ('loop_to_x', LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target)),
          ('loop_to_x', ArgMaxLift()), ('loop_to_x', LoopToConditionalReduce()),
          ('loop_to_x', LoopToStreamCompaction())]

    # cascade_iedges_up (pre-parallelize): re-run after fission / normalize rewrite
    # the CFG; MUST precede LoopToMap. Re-unique the iterators (ssa) so the
    # distributed siblings are independent.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp()), ('ssa', _uniq2)]

    # NOTE: MoveLoopInvariantIfUp is deliberately NOT wired here. It is the dual of
    # the earlier ``MoveIfIntoLoop`` stage, so hoisting guards back out here would
    # undo that work and ping-pong. The terminal ``hoist_guards`` stage runs it
    # once, AFTER fuse, where the fusion it would otherwise undo has happened.

    # Wavefront skewing has a single home: the ``loop_fuse`` block after
    # ``post_l2m`` (see below). It is the final parallelization attempt, applied
    # to the sequential residue ``LoopToMap`` refused -- so it must run after
    # ``LoopToMap``, not before it.

    # loop_to_scan (late, post-fission + post-skew): a second LoopToScan pass
    # catches prefix-scan recurrences that only emerged AFTER ``LoopFission``
    # isolated the recurrence statement (TSVC ``s221``:
    # ``a[i] = a[i] + c[i]*d[i]; b[i] = b[i-1] + a[i] + d[i]`` -> two fissioned
    # loops, the ``b`` loop is a clean scan). The earlier in-``reduce``
    # LoopToScan handles single-statement scan bodies that don't need fission
    # (``s242``, ``s1221``); running it again here also lifts the post-fission
    # ones without harming the already-lifted shapes.
    s += [('loop_to_scan', LiftPreprocess()),
          ('loop_to_scan', LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target))]
    # Close the semantic-lifting band. This is the last of the lifting stages (``lift_inv`` /
    # ``normalize_reduction`` / ``loop_to_symm`` / ``loop_to_scan``), and the next phase reads the
    # graph differently: ``parallelize`` asks dependence questions of every remaining loop. Lifting
    # splices states and rewrites bodies, so the phase boundary is exactly where the tidy belongs --
    # the cleanup in ``reduction_to_wcr_map`` below is the next one, and it sits after LoopToMap has
    # already run, which is too late to be "after lifting".
    s += _structural_cleanup('loop_to_scan')

    # parallelize: the canonical (fissioned / normalized) loops -> parallel maps.
    # ``LoopToMap`` reads the scope-summary memlets as its write set, so rebuild them from the
    # bodies first: the inline stages above expose exact body subsets without re-propagating the
    # enclosing map, which leaves polybench ``covariance``'s map exit claiming ``cov[0:M, 0:M]``
    # while the body writes ``cov[i, i:M]``. See :class:`PropagateMemlets`.
    # A store a LATER iteration overwrites unread is the only carrier some loops have (TSVC
    # ``s244``); dropping it, and peeling the tail iterations whose store does survive, hands
    # LoopToMap a DOALL loop. Must precede it -- afterwards there is no LoopRegion to peel.
    s += [('parallelize', DeadCarriedStoreElimination())]
    s += [('parallelize', PropagateMemlets())]
    s += [('parallelize', PatternMatchAndApplyRepeated([LoopToMap()]))]

    # ``LoopToMap`` is where body NestedSDFGs are MINTED, and it derives their connector set from
    # the loop's read/write sets rather than from what the body still uses -- so a statement split
    # or a fission upstream can leave a connector nothing inside reads. That is not cosmetic: the
    # inliner materialises an access node for it in the parent, held by an ordering edge alone, and
    # the next pass to derive read sets from memlets builds a body SDFG without that descriptor and
    # dies looking the node up. ``PruneConnectors`` removes the connector, its outer memlets and the
    # orphaned descriptor; the earlier 'lower' instance runs long before these nodes exist.
    s += [('parallelize', PatternMatchAndApplyRepeated([PruneConnectors()]))]

    # GPU: perfect MAP nests for the grid collapse, via the map-side PerfLoopNesting
    # (delegates to MapFission -- the safe, data-parallel distribution; map iterations carry no
    # dependences, so unlike the removed loop-side PerfectLoopNesting no grouping analysis can
    # silently split a recurrence). Runs after LoopToMap, once maps exist.
    if target == 'gpu':
        s += [('parallelize', PatternMatchAndApplyRepeated([PerfLoopNesting()]))]

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
    # ``AccumulatorCopyChainToWCR`` + ``LoopToMap`` then lift to a parallel WCR-map.
    s += [('reduction_to_wcr_map', FuseChainedScalarReductions())]
    # The normalization steps ``LoopToReduce(wcr-scalar)`` used to run itself, now explicit so
    # the lifter stays side-effect free on a no-match. Order is load-bearing:
    # ``AccumulatorCopyChainToWCR`` destroys the augassign shape ``LoopToReduce`` claims and
    # creates the WCR shape ``RetargetWCRAccumulator`` claims, so it sits strictly between
    # them and ``LoopToReduce`` must not run again after it.
    s += [('reduction_to_wcr_map', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]
    # Re-use LoopToMap's dependence analysis to pin any top-level loop it refuses because of
    # a carried dependency; leave DOALL-eligible top-level loops untouched. Nesting alone pins
    # nothing: whether a reduction map inside a sequential loop earns its OpenMP region is a CPU
    # question, decided in the ``cpu_specialize`` band by
    # ``SequentializeUnprofitableParallelScopes``.
    s += [('reduction_to_wcr_map', PinCarriedTopLevelLoops())]
    s += [('reduction_to_wcr_map', AccumulatorCopyChainToWCR())]
    s += [('reduction_to_wcr_map', RetargetWCRAccumulator())]
    # Rebuild the scope summaries LoopToMap reads (see the note at the first parallelize stage).
    s += [('reduction_to_wcr_map', PropagateMemlets())]
    s += [('reduction_to_wcr_map', PatternMatchAndApplyRepeated([LoopToMap()]))]
    # ``LoopToMap`` splits the loop body into per-iteration NestedSDFG
    # states whose intermediate transients share names across siblings --
    # scratch arrays as much as scalars. Renaming each scope's transient
    # here keeps the downstream structural cleanup's same-name candidate
    # list short -- defence-in-depth for the StateFusionExtended same-
    # name writer-merge guard.
    s += [('reduction_to_wcr_map', _PrivatizeScalarsStage()), ('reduction_to_wcr_map', _PrivatizeArraysStage())]
    s += _inline_single_state('reduction_to_wcr_map')
    s += _structural_cleanup('reduction_to_wcr_map')
    # LoopToMap above outlines the body, trapping the fresh WCR inside the nsdfg; the
    # normalize_reduction run is one band too early to see it (tsvc s4115). Idempotent.
    s += [('reduction_to_wcr_map', NormalizeWCR())]

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

    # post_l2m: insert assign tasklets at map boundary, then inline the single-state bodies
    # LoopToMap left -- after LoopToMap. State fusion waits for the ``fuse`` cleanup phase.
    #
    # Load-bearing, measured: dropping it leaves tsvc ``va`` (``a[i] = b[i]``) as a plain
    # Map (0L/0M -> 0L/1M over the corpus) instead of the Memcpy library node
    # ``AssignmentAndCopyKernelToMemsetAndMemcpy`` lifts it to -- that recogniser matches the
    # assign tasklet this pass plants on the boundary copy.
    s += [('post_l2m', InsertAssignTaskletsAtMapBoundary())]
    s += _inline_single_state('post_l2m')

    # coalesce: prepare the graph for maximal map fusion now that the DOALL
    # loops have become maps -- see ``_coalesce`` for the per-step rationale.
    s += _coalesce()

    # loop_fuse (post-parallelize recovery): every DOALL loop is now a Map, so the
    # remaining LoopRegions are exactly the sequential residue LoopToMap refused.
    # ``ReconstructWavefrontNest`` runs FIRST (gated, off by default): a stencil body
    # whose slice-vectorized statement stayed a Map beside its sequential scan
    # LoopRegion is an imperfect nest ``WavefrontSkew`` refuses outright, so this
    # rebuilds the single-LoopRegion body it requires, committing only when a trial
    # proves the rebuild unlocks a skew. ``LoopFusion`` fuses consecutive same-range
    # sequential siblings (locality; it cannot touch the already-parallel Maps, which
    # it never matches). ``WavefrontSkew`` then makes its final parallelization attempt
    # on those residual 2-D nests (seidel / nussinov and any nest the two passes above
    # merged), and a second ``LoopToMap`` maps the inner axis it exposes. This is the
    # single home of wavefront skewing: it only ever runs on what LoopToMap could not take.
    if reconstruct_wavefront_nest:
        # Same WCRToAugAssign-before-MapToForLoop ordering the 'lower' stage above establishes:
        # ReconstructWavefrontNest drives MapToForLoop, whose "a surviving WCR output is a genuine
        # reduction" refusal is only true once the conflict-free ones are already reverted. The
        # slice statement's ``A[i, 1:-1] += ...`` is a MULTI-element tasklet write at 'lower' time,
        # which WCRToAugAssign must refuse (a scalar aug-assign tasklet over an array memlet would
        # codegen ``double* + double*``); only after LoopToMap splits the slice into a per-element
        # map body is the write scalar and the spurious WCR revertible (polybench seidel_2d).
        s += [('loop_fuse', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]
        # ...then tidy, because reverting the WCR is what makes the body inlinable at all: while the
        # RMW lives in the WCR, the body NestedSDFG's in/out connector for the destination has NO
        # read AccessNode inside, and ``InlineSDFG`` refuses a connector with no valid matching
        # access node. Inlining is what replaces the whole-array boundary memlet with the body's
        # real ``A[i, j+1]``, which every downstream dependence test needs.
        s += _inline_single_state('loop_fuse')
        s += [('loop_fuse', ReconstructWavefrontNest())]
    # GPU only: a state stranded between two loops blocks FuseLoops outright (it matches a two-node
    # path graph). SinkStateIntoLoop above already recovers the case where the state can be replicated
    # per iteration; ReorderStateForLoopFusion recovers the disjoint case instead, reordering the state to after the
    # second loop. GPU-gated because there fusing is worth more than the tidier block order -- one
    # kernel launch instead of two -- while on CPU the two loops cost about the same either way.
    # Measured over all four corpora at both targets: it fires ZERO times, so un-gating it to CPU
    # was not justified. The 14 loop-state-loop chains that exist all die on a RAW, a NestedSDFG in
    # the second loop, or an interstate assignment -- and tsvc produces no chain at all, because
    # LoopToMap has consumed the loops before this stage runs.
    if target == 'gpu':
        # Wrapped because it declares ``depends_on([AccessSets])``: stages are applied with an empty
        # results dict, so a bare dependency-bearing pass trips ``_assert_self_contained`` -- which
        # is why the original bare wiring made ``canonicalize(target='gpu')`` raise outright.
        s += [('loop_fuse', ppl.Pipeline([ReorderStateForLoopFusion()]))]
    s += [('loop_fuse', LoopFusion())]
    s += [('loop_fuse', WavefrontSkew(target=target))]
    # Rebuild the scope summaries LoopToMap reads (see the note at the first parallelize stage).
    s += [('loop_fuse', PropagateMemlets())]
    s += [('loop_fuse', PatternMatchAndApplyRepeated([LoopToMap()]))]
    s += _inline_single_state('loop_fuse')

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
        s += _inline_single_state('lift_copy')

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
    s += _inline_single_state('interchange')

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
    s += _inline_single_state('fuse')
    s += _structural_cleanup('fuse')
    s += [('fuse',
           PatternMatchAndApplyRepeated([DistributeTaskletIntoMap(),
                                         MapFusionVertical(),
                                         MapFusionHorizontal()]))]

    # A map that only fills a transient for an immediately following reduction is that reduction:
    # ``maxv = max(|a[i]|)`` reaches here as ``map -> _argf_buf[LEN_1D] -> Reduce``, so the
    # canonical form allocates a whole problem-size buffer to hold values the reduction consumes
    # once (s3113: 4.166 GB at XL). Fusing folds the producer in and leaves a WCR map -- the shape
    # that already IS the exposed parallelism -- with no buffer and no library node. Both guard on
    # the intermediate being a transient nothing else reads, which is what makes the producer safe
    # to delete. After MapFusion, so a producer built from several fused maps is matched whole.
    s += [('fuse', PatternMatchAndApplyRepeated([MapReduceFusion(), MapWCRFusion()]))]

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
    s += _inline_single_state('fuse')
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

    # hoist_guards (terminal): hoist any still-invariant guard outward. Run
    # AFTER fuse -- the dual ``MoveIfIntoLoop`` (prep stage) has already pushed
    # guards in to enable sibling fusion, so a terminal hoist of a guard that
    # is STILL invariant w.r.t. the whole remaining loop nest does not undo
    # that fusion; it lifts the surviving config-flag guards (ICON ``istep ==
    # 1``, cloudsc ``IWARMRAIN`` etc.) to the cheapest scope. MoveLoopInvariantIfUp's
    # dead-outside-branch match lifts past per-iteration iedge assignments
    # (``start = jb // 4``), which stay in the now-guarded loop body.
    #
    # Target-gated. On CPU a guard is hoisted as far as it will go: every level
    # it clears removes a re-evaluation from an enclosing iteration, and a guard
    # that stalls partway is still strictly cheaper than one left innermost. On
    # GPU a partial hoist is not a win but a hazard -- a guard stalled between
    # two levels of a map chain splits the nest at that level, so what would
    # have been one kernel becomes a branch around several launches. There the
    # hoist is all-or-nothing (``require_full_hoist``): take it only when the
    # guard clears the complete chain and the whole nest stays one kernel.
    s += [('hoist_guards', MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu')))]

    # By this point fully-parallel guarded nests are collapsed maps, so the
    # surviving invariant guard sits inside a map body (not a loop) where
    # MoveLoopInvariantIfUp cannot reach it. MoveMapInvariantIfUp is its map
    # analogue / the inverse of MoveIfIntoMap: a guard whose condition does not
    # depend on the map parameters is lifted out of the map, one map copy per
    # branch (``map[i, j]: { if c: A else B }`` -> ``if c: { map[i, j]: A } else
    # { map[i, j]: B }``), so each branch is a clean unconditional parallel map.
    # Same target gate as the loop-side hoist above: on GPU a guard that stalls
    # between two levels of a map chain splits the nest there, so take the
    # hoist only when it clears the whole chain and the nest stays one kernel.
    s += [('hoist_guards', MoveMapInvariantIfUp(require_full_hoist=(target == 'gpu')))]

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

    # relax_powers: freeze a provable non-negative integer ``base ** exp`` to the exact integer
    # ``ipow`` on the size / subscript / bound sites WHILE the loop-iterator ranges that prove the
    # exponent non-negative are still live. This must be a canonicalization pass, NOT deferred to
    # codegen: an earlier ``SimplifyPass`` folds ``R**i * R**(K-i-1)`` (both exponents
    # range-nonnegative inside the enclosing loop) into ``R**(K-1)``, and by codegen that size is a
    # persistent state-struct allocation OUTSIDE any loop -- the range that proved ``K-1 >= 0`` is
    # gone, so the codegen-time relax leaves a ``dace::math::pow`` (double) size that is not
    # integral (``new complex128[...]`` -> compile error, stockham_fft).
    # (This does NOT harm ``N**2``-style sizes: freezing ``N**2 -> ipow(N, 2)`` is value-exact;
    # an earlier suspicion that it miscompiled gramschmidt was a misdiagnosis -- that was an
    # uninitialized WCR read whose layout the relax merely perturbed.)
    s += [('relax_powers', RelaxIntegerPowers())]

    # end (reclaim): the two reclaimers, on their own -- NOT a SimplifyPass. These are the only parts of
    # the former terminal simplify the recipe actually relied on; the rest of ``SIMPLIFY_PASSES``
    # (inlining, state fusion, control-flow raising, scalar-to-symbol promotion, constant
    # propagation) is either done by dedicated stages above or actively unwanted this late.
    #
    # ``DeadDataflowElimination`` first: ``SplitStatements`` replicates a statement per independent
    # output, and a replica whose output no later stage consumes (``fission_dep_then_indep``'s
    # ``nested_sdfg_a`` chain -- an uninitialized read feeding a write nothing reads back) is orphaned
    # only AFTER the ``reduce`` simplifies have run, by the fission / parallelize stages that drop its
    # consumer. Nothing else in the recipe removes a dead chain.
    #
    # ``ArrayElimination`` second, on the smaller graph: ``MapFusionVertical`` mints a fresh
    # ``__map_fusion_<x>`` carrier per fused edge, so a value consumed by two fused consumers (the
    # ``fuse_diamond`` shape) ends up with a second carrier that is a plain copy of the first. The
    # post-fuse ``RedundantArray`` below cannot fold that -- it only redirects a transient's WRITERS,
    # which is impossible while the SOURCE still has another reader -- and the mirrored
    # ``RedundantSecondArray`` is unsafe to run bare (it would redirect a copy's READERS onto a WAR
    # carrier, TSVC s212). ``ArrayElimination`` is the pass that carries the ``_is_war_carrier`` /
    # ``_state_has_read_write_sibling_carrier`` guards for exactly that case.
    #
    # Wrapped in a ``Pipeline`` because both declare dependencies (``ControlFlowBlockReachability`` /
    # ``AccessSets``, ``StateReachability`` / ``FindAccessStates``), the same way ``ScalarFission`` /
    # ``ArrayFission`` are above. Placed where the terminal simplify used to sit, so the reclamation
    # point is unchanged and the stages after it see the graph they were written against.
    s += [('end', ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]))]

    # ...then tidy the state machine, with the recipe's own between-phase helper rather than a
    # SimplifyPass. The terminal simplify used to be the last thing that ran ``FuseStates`` /
    # ``DeadStateElimination``, so the scaffolding earlier stages leave behind reached codegen the
    # moment it went. Running it AFTER the reclaimers means it also splices out whatever they just
    # emptied. Led by the inline, like every other cleanup site: an un-inlined map body reports
    # whole-array memlets, so ``StateFusionExtended`` would judge the merge on the bounding box.
    #
    # It does NOT reach ``ChunkAntiDependence``'s ``*_antidep_prologue`` / ``*_antidep_seam*``
    # states: that pass belongs to the ``cpu_specialize`` stage, which runs after this whole
    # pipeline, and those states are its output rather than residue -- each carries a map of the
    # chunked lift.
    #
    # ``PruneEmptyConditionalBranches`` closes the case the state-level cleanup cannot see: an empty
    # conditional ARM is a ControlFlowRegion, not a state, so ``DeadStateElimination`` walks past it.
    # ``ConditionFusion`` merges two adjacent guards into one ConditionalBlock whose branches are
    # their cross product, and the combination that does no work is an empty arm -- the terminal
    # simplify used to drop it, and without that the collapsed nest carries a dead fourth branch
    # (``canonicalize_coexisting_guards``) and the guarded scan split keeps an empty ``else``
    # (``scan_conditional``). It only ever removes a branch with no work in it, so the guarded
    # specializations the recipe leans on -- whose arms all carry a body -- are untouched.
    s += _inline_single_state('end')
    s += [('end', PruneEmptyConditionalBranches())]

    # Final parallelize sweep: the symbolic-stride scan specialization
    # (``LoopToScan._specialize_scan_under_stride_guard``) emits its carry-free
    # delta-build loop INSIDE the ``if stride >= 1`` ConditionalBlock branch, i.e. AFTER
    # the earlier ``parallelize`` LoopToMap stages have run, so it survived as a residual
    # sequential loop even though it is embarrassingly parallel (``scan_strided_sym`` /
    # ``ext_floordiv_offset`` / ``fission_dep_sym_offset`` -- the delta-build
    # ``_scan_in[i-K] = x[i]``). Lift any such residual now. Nothing normalizes the graph
    # between the earlier sweeps and this one, so ``LoopToMap`` matches on un-simplified
    # input; its bound/subset comparisons re-parse through the symbol registry by NAME
    # (:func:`~dace.transformation.interstate.loop_to_map._same_injective_index`,
    # ``symbolic.equalize_symbols``), which is what keeps that robust.
    # LoopToMap only fires on genuinely parallel loops and
    # no-ops otherwise, so this cannot mis-parallelize a real carry. BEFORE
    # AssumeSymbolConstraints, which must stay the terminal stage.
    # Lift loop-carried in-place array reductions (contour_integral's
    # ``for idx: P[i, j] += X[i, j]``) to WCR writes so the terminal LoopToMap
    # parallelizes the enclosing loop. Runs post-canon (loops in fissioned /
    # normalized form) right before the terminal parallelize sweep.
    s += [('end', LiftLoopCarriedReduction())]
    # Rebuild the scope summaries LoopToMap reads (see the note at the first parallelize stage).
    s += [('end', PropagateMemlets())]
    s += [('end', PatternMatchAndApplyRepeated([LoopToMap()]))]
    s += _inline_single_state('end')

    # Terminal fuse: the main ``fuse`` stage runs BEFORE ``normalize_wcr`` and the
    # terminal ``LoopToMap`` above. Two maps that were not yet fuseable at that point
    # can become fuseable only afterwards: ``NormalizeWCRSource`` reshapes a reduction
    # consumer's WCR from the seeded privatized-accumulator form (IN-wcr / plain
    # copy-out) into a plain map-exit WCR that IS fuseable, and the terminal
    # LoopToMap lifts residual loops into fresh maps adjacent to existing ones. With no
    # fuse after those stages, such producer->consumer pairs stay split (polybench
    # ``syrk``: the ``alpha*A*A`` product map + the ``C += ...`` k-reduction map, both
    # over the same ``0:i+1`` slice, stayed as two maps == two fork/joins per k step).
    # Re-run vertical+horizontal fusion in final map form so every fuseable pair is
    # fused; the dependency guards still refuse the unsafe ones. The
    # following SymbolDedup cleans up the duplicate index symbols fusion introduces.
    s += [('end', PatternMatchAndApplyRepeated([DistributeTaskletIntoMap(),
                                                MapFusionVertical(),
                                                MapFusionHorizontal()]))]

    # redundant_array (post-fuse cleanup): drop a transient that only ever gets copied wholesale into
    # its destination, so the producing map writes the destination directly. No ``SimplifyPass`` runs
    # after the ``reduce`` stage, so from well before the terminal LoopToMap and the terminal fuse
    # above nothing reclaims arrays at all -- and ``ArrayElimination`` (Simplify's array reclaimer)
    # refuses this shape anyway: its ``_is_war_carrier`` guard skips the candidate whenever the DESTINATION is read
    # and written in the same state, which is every in-place stencil sweep (heat3d's
    # ``A_slice -> A[1:-1, 1:-1, 1:-1]``, an (N-2)^3 buffer). ``RedundantArray`` only ever redirects a
    # transient's WRITERS into the destination, so it cannot expose a read to a later in-place write;
    # the mirrored ``RedundantSecondArray`` fold (redirecting a copy's READERS onto a WAR carrier, TSVC
    # s212) is what that guard exists for and is deliberately NOT run here -- on the corpus it never
    # matched anyway, and refusing costs a warning. BEFORE the remat stage: deleting the buffer first
    # shortens the chains remat then walks.
    s += [('end', PatternMatchAndApplyRepeated([RedundantArray()]))]

    # remat: vertical fusion pulls a consumer sub-expression UP into the producer, because the value it
    # is built from is a register there; when a THIRD map still consumes the result, fusion cannot delete
    # it and it crosses the map boundary as a full transient array. Recompute it in the consumer instead.
    # Legal only when every input of the recomputed chain is already on an existing consumer read, so the
    # rewrite adds no memory traffic and deletes an array outright. Runs AFTER the terminal fuse -- that
    # is the last stage that can create the shape -- and cleans up after itself, so no simplify is needed.
    s += [('end', RematerializeDerivedTemporaries())]

    # Post-optimization structural cleanup. Every other occurrence sits BETWEEN canonicalization
    # phases, so the optimization tail above -- terminal LoopToMap, terminal fuse, redundant-array,
    # remat -- is the one band whose output nothing tidies: each of those merges or deletes nodes
    # and leaves states that can now fuse and ordering edges the fused dataflow already implies.
    # Before the terminal bookkeeping (symbol pruning, OptionalArrayInference,
    # PruneUnreferencedTransients) rather than after it, so those still observe the final graph.
    s += _structural_cleanup('end')

    # Terminal symbol cleanup: after fusion, a fused gather-map body carries
    # duplicate index symbols that map fusion introduced -- ``idx_index`` and
    # ``idx_index_0`` both ``idx[i]`` (``idx[i]`` computed twice). ``SymbolDedup``
    # merges provably-equal interstate-edge symbols; the following
    # ``SymbolPropagation`` + ``ConstantPropagation`` then re-fold the survivors
    # (the merge can expose fresh constant/symbol chains). BEFORE
    # AssumeSymbolConstraints, which must stay the terminal stage.
    s += [('end', SymbolDedup()), ('end', SymbolPropagation()), ('end', ConstantPropagation())]
    # SymbolPropagation above folds symbols out of interstate edges but leaves the
    # now-unreferenced entries in sdfg.symbols. No SimplifyPass runs past the ``reduce``
    # stage, so this is the ONLY thing that prunes them -- it is load-bearing, not a top-up.
    s += [('end', RemoveUnusedSymbols())]

    # OptionalArrayInference: ``optional`` is a DERIVED annotation on every array descriptor, and the
    # terminal simplify was what last recomputed it. Without it canonicalize emits a graph whose
    # descriptors carry no ``optional``, and re-canonicalizing that output annotates it at the
    # leading ``clean`` simplify -- so the pipeline is not idempotent (tsvc s000 / s111 / s1112 /
    # s1113 diverge on ``_arrays.*.attributes.optional`` ABSENT -> PRESENT). Recompute it here,
    # after the last stage that changes the graph, so the output is already the fixed point.
    s += [('end', OptionalArrayInference())]

    # ConvertLengthOneArraysToScalars a SECOND time (it leads ``prep``): the canonical spelling of
    # a single-value transient is a Scalar, and the stages between the two -- map fusion's
    # ``__map_fusion_*`` carriers above all -- MINT length-1 Array transients that the ``prep``
    # occurrence ran too early to see. Left as Arrays they are the descriptor a re-canonicalize
    # then converts at ``prep``, i.e. the output is one pass short of its own canonical form.
    s += [('end', ConvertLengthOneArraysToScalars())]

    # revert_nonreduction_wcr (terminal): the terminal ``LoopToMap`` + fusion above form fresh
    # ``map_exit -> output`` WCR edges that the earlier ``revert_nonreduction_wcr`` (which ran
    # before those maps existed) never saw -- an injective slice aug-assign such as seidel's
    # ``A[i, 1:-1] += <neighbours>`` whose ``j``-loop only became a Map here. Its per-iteration
    # write is a single conflict-free element and the tasklet already reads the destination back,
    # so the map-exit WCR is a spurious atomic over a conflict-free store: WCRToAugAssign (expr 4)
    # drops it to a plain indexed write. The injectivity gate still keeps genuine reductions (a
    # real ``w[i] += ...`` k-reduction whose write does NOT vary with the map lane).
    s += [('end', PatternMatchAndApplyRepeated([WCRToAugAssign()]))]

    # cleanup (terminal): drop transients nothing names any more. The stages above delete a
    # temporary's last reader without deleting its descriptor, and ``ArrayElimination`` -- the only
    # reclaimer that erases a descriptor -- runs well before them and skips ``Scalar`` outright, so
    # the frontend's per-expression scalars (``b_index``, ``a_slice_times_b_slice``) survive a full
    # canonicalize with no node left referring to them. Codegen ignores them; a re-run does not, and
    # neither does anything that reads the serialized SDFG.
    s += [('end', PruneUnreferencedTransients())]

    # cleanup (terminal): inline the plain control-flow regions the middle stages leave standing.
    # ``rotate`` splits a block and no ``clean`` stage runs after it, so the recipe can finish
    # holding regions that carry a single body each (tsvc s255 ends with two). They are not a state
    # fusion -- ``StateFusionExtended`` matches ``SDFGState`` and has nothing to say about a region
    # -- so only an inline reclaims them. Defaults skip LoopRegion / ConditionalBlock / named /
    # function-call regions, i.e. every region whose structure carries meaning; a bare
    # ``ControlFlowRegion`` carries none, which is why it is the one safe to flatten here.
    s += [('end', InlineControlFlowRegions())]

    # NOTE: fresh WCR accumulators are identity-seeded by ``NormalizeWCRSource`` (the
    # ``normalize_wcr`` stage above), not a separate pass -- codegen never seeds a WCR
    # accumulator, so a reduction into genuinely-uninitialized scratch reads garbage. That pass
    # seeds only a provably-fresh, write-once accumulator: a transient (or an out-only nested
    # connector whose every caller binding is a transient -- never an aliased live array such as
    # gramschmidt's in-place ``__tmp_78 -> &A[j]``), with no plain initializer, whose WCR writes
    # a Map-parameter-indexed slot (so a same-slot fold that continues a live prior -- nussinov's
    # ``_priv_table`` -- is left alone). It does not attempt full cross-nested-SDFG liveness, so a
    # fresh accumulator whose WCR is already AccessNode-sourced before that pass is not covered.

    # The CPU specialization band used to run here, as the pipeline's terminal stage. It is now a
    # SEPARATE STAGE: :func:`~dace.transformation.passes.cpu_specialization.pipeline.cpu_specialize`,
    # run after this whole pipeline returns (``finalize_for_target`` calls it for you). Wherever the
    # choice between parallel and sequential is open, canonicalization takes PARALLEL and stops
    # there; giving parallelism back is a target decision, and a target decision inside the
    # device-neutral pipeline is one a GPU or a vectorizer then has to undo. The order is unchanged
    # -- specialization still runs after cleanup and parallelization, at the very end -- only the
    # stage boundary is.

    # assume_constraints (LAST): make the assumptions the pipeline relied on
    # explicit and runtime-checked, by prepending a side-effecting
    # ``std::abort`` start state that aborts when one is violated -- a
    # negative signed-integer free symbol (the offset-sign nonnegativity
    # contract) or a false tracked relation (e.g. the ``K < N`` a modular-wrap
    # split leaned on). Runs AFTER every structural pass:
    # a guard prepended earlier is orphaned by any pass that builds its own entry
    # state (LoopToScan's scan-init block, reduction init, ...), which resets the
    # top-level start block and leaves the guard a disconnected source that
    # dominator analyses then KeyError on. Emitting it last -- nothing runs after
    # -- avoids that entirely while still yielding a first-state guard at codegen.
    # The external free-symbol set is unchanged by canonicalization (only loop
    # iterators are renamed, and those are bound).
    s += [('end', AssumeSymbolConstraints())]

    # Last: any pass above may have built an index with python's `//` on a sympy expression,
    # which is sympy floor() -- distributed by sympy and then printed WITHOUT the floor, so the
    # index truncates term by term. Normalizing here means nothing reaches codegen holding one.
    s += [('end', NormalizeFloorDivision())]

    # Pipeline does not propagate `progress` to subpasses, so sweep once here instead of at each call site.
    for _, unit in s:
        if isinstance(unit, PatternMatchAndApplyRepeated):
            unit.progress = False
    return s


#: A stage factory returns that stage's fresh passes, in order.
StageFactory = Callable[[], List[ppl.Pass]]


def _stage_runs() -> List[Tuple[str, int, int]]:
    """Split the flat recipe into contiguous runs of one stage label.

    A label may appear in several places in the recipe (``clean``,
    ``cascade_iedges_up`` and ``peel`` all do). Grouping by label alone would
    gather those separated occurrences at the position of the first one and so
    reorder the recipe -- which silently breaks documented constraints, e.g.
    ``RemoveViews`` must run before the ``loop_to_symm`` / ``lift_inv`` lifts read
    the raw frontend shape while the rest of the ``clean`` block runs after them,
    and both halves are labelled ``clean``. Runs preserve the real order.

    :returns: ``(label, start, stop)`` index ranges into the flat recipe.
    """
    runs: List[List] = []
    for i, (lbl, _p) in enumerate(_build_stages()):
        if runs and runs[-1][0] == lbl and runs[-1][2] == i:
            runs[-1][2] = i + 1
        else:
            runs.append([lbl, i, i + 1])
    return [(lbl, a, b) for lbl, a, b in runs]


def _stage_factory(start: int, stop: int) -> StageFactory:
    """Return a factory yielding fresh passes for one run of the recipe.

    :param start: Index of the run's first pass in the flat recipe.
    :param stop: Index one past the run's last pass.
    :returns: A factory that builds that run's passes in order.
    """
    return lambda: [p for _lbl, p in _build_stages()[start:stop]]


#: Grouped view of :func:`_build_stages`: ``(label,
#: factory)`` per stage, in order, where ``factory()`` builds that stage's
#: fresh passes. ``_build_stages`` (flat ``(label, pass)``) is the source of
#: truth used by the pipeline; this view exists for callers that iterate
#: stage-by-stage (``for name, factory in CANONICALIZE_STAGES:
#: for unit in factory(): ...``).
CANONICALIZE_STAGES: List[Tuple[str, StageFactory]] = [(label, _stage_factory(start, stop))
                                                       for label, start, stop in _stage_runs()]


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
    :param perfect_loop_nesting: Run ``PerfectLoopNesting`` at the fission stage. ON by
                                 default -- see the ruling at the fission stage below.
    :param target: ``'cpu'`` (default) or ``'gpu'``. Picks the per-target knob
                   preset (see ``_CPU_DEFAULTS`` / ``_GPU_DEFAULTS``). Any
                   explicit knob argument (e.g. ``interchange_carry_with_map=...``)
                   overrides the preset for that knob.
    :param interchange_carry_with_map: ``LoopToScan`` knob: relocate the carry
                                       ``LoopRegion`` INTO the per-column Map so
                                       the scan runs sequential-per-thread.
                                       ``None`` (default) -> per-target preset.
    :param reconstruct_wavefront_nest: Run ``ReconstructWavefrontNest`` right before
                                       ``WavefrontSkew`` in the ``loop_fuse`` stage.
                                       ``None`` (default) -> per-target preset (off on
                                       both targets; see ``_CPU_DEFAULTS``).
    :param normalize_loop_and_map_origin: Run ``NormalizeLoopAndMapOrigin`` right before the
                                          ``loop_to_x`` stage. ``None`` (default) -> per-target
                                          preset (off on both targets; see ``_CPU_DEFAULTS``).
    :param specialize_constants: Optional ``{symbol: value}`` map (e.g. CloudSC's
                             ``{'nclv': 5}``, or a kernel's shape symbols like
                             ``{'Norb': 3}``) baked into the SDFG via
                             ``specialize_symbols`` (recursively, dropping the symbol)
                             BEFORE canonicalization -- the same specialization the
                             cloudsc parallelization pipeline does. Symbolic trip
                             counts that become concrete then unroll under
                             ``ShortLoopUnroll``, and concrete matmul extents let
                             ``canonicalize_set_fast_implementations`` pick the inlined
                             ``'pure'`` GEMM for known-small dims. ``None`` leaves every
                             symbol symbolic.
    """

    CATEGORY: str = 'Canonicalization'

    validate = properties.Property(dtype=bool, default=True, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(
        dtype=bool,
        default=True,
        desc='Validate the SDFG after EVERY stage that reported a modification, which pinpoints the '
        'stage that produced an invalid SDFG instead of only catching it at the end. Affordable by '
        'default because it is scoped: a stage returning None is skipped, and a Pipeline validates '
        'its own sub-passes rather than re-walking the whole SDFG here. Set False to skip it.')
    dump_dir = properties.Property(
        dtype=str,
        default=None,
        allow_none=True,
        desc='Directory to save the SDFG into after every stage, for bisecting which stage broke it. '
        'None (default) disables dumping entirely.')
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
    reconstruct_wavefront_nest = properties.Property(
        dtype=bool,
        default=False,
        desc='Run ReconstructWavefrontNest right before WavefrontSkew in the loop_fuse stage, rebuilding '
        'an imperfect Map-plus-LoopRegion stencil body into the single loop WavefrontSkew requires. '
        'Off by default on both targets (unproven corpus benefit; see _CPU_DEFAULTS).')
    normalize_loop_and_map_origin = properties.Property(
        dtype=bool,
        default=False,
        desc='Run NormalizeLoopAndMapOrigin right before the loop_to_x stage, rebasing every Map range / '
        'LoopRegion counter to a 0-based begin while keeping the stride. Off by default on both targets '
        '(see _CPU_DEFAULTS).')
    perfect_loop_nesting = properties.Property(
        dtype=bool,
        default=True,
        desc='Distribute a loop over its data-independent statement groups (PerfectLoopNesting) so '
        'each statement parallelizes on its own axes. On by default since the pass stopped grouping '
        'through LoopFission -- see the fission-stage comment for what that changed and for the '
        'CloudSC gate that has not yet been re-run.')

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
        desc='Lift contiguous copy/zero-init maps to Copy/Fill library nodes (False keeps them as maps).')
    semantic_lifting = properties.Property(
        dtype=bool,
        default=True,
        desc='Master gate for the post-LoopToMap map->library-node lifts (Einsum + Copy/Fill). '
        'False (set by the vectorizer) keeps the residual as raw maps it can lower.')

    def __init__(self,
                 validate: bool = True,
                 validate_all: bool = True,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 peel_limit: Optional[int] = None,
                 break_anti_dependence: Optional[bool] = None,
                 target: str = 'cpu',
                 interchange_carry_with_map: Optional[bool] = None,
                 scatter_to_guarded_maps: Optional[bool] = None,
                 privatize_scatter_reductions: Optional[bool] = None,
                 reconstruct_wavefront_nest: Optional[bool] = None,
                 normalize_loop_and_map_origin: Optional[bool] = None,
                 assume_parallel_guards: bool = False,
                 perfect_loop_nesting: bool = True,
                 specialize_constants: Optional[Dict[str, int]] = None,
                 lift: bool = True,
                 lift_copy: bool = True,
                 semantic_lifting: bool = True,
                 dump_dir: Optional[str] = None):
        if target not in _TARGET_DEFAULTS:
            raise ValueError(f"target must be one of {sorted(_TARGET_DEFAULTS)}; got {target!r}")
        self.validate = validate
        self.validate_all = validate_all
        self.dump_dir = dump_dir
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
        self.reconstruct_wavefront_nest = _resolve_target_default(target,
                                                                  'reconstruct_wavefront_nest',
                                                                  reconstruct_wavefront_nest,
                                                                  fallback=False)
        self.normalize_loop_and_map_origin = _resolve_target_default(target,
                                                                     'normalize_loop_and_map_origin',
                                                                     normalize_loop_and_map_origin,
                                                                     fallback=False)
        self.assume_parallel_guards = assume_parallel_guards
        self.perfect_loop_nesting = perfect_loop_nesting
        self.lift = lift
        self.lift_copy = lift_copy
        self.semantic_lifting = semantic_lifting
        self._specialize_constants = specialize_constants or {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Canonicalize ``sdfg`` in place.

        :param sdfg: The SDFG to canonicalize.
        :returns: The number of passes applied.
        """
        disable_openmp_sections(sdfg)
        # Specialize chosen symbols to constants first (e.g. ``nclv = 5``), so the
        # otherwise-symbolic species-loop trip counts become concrete and unroll.
        # ``specialize_symbols`` descends into nested SDFGs (and strips their
        # ``symbol_mapping``); a plain ``replace_dict`` would leave nested-SDFG
        # bodies -- the bulk of a real cloudsc build -- unspecialized. Baking the whole
        # set in one call costs one graph walk per SDFG instead of one per symbol.
        if self._specialize_constants:
            from dace.sdfg.utils import specialize_symbols
            specialize_symbols(sdfg, self._specialize_constants)
        stages = _build_stages(unroll_limit=self.unroll_limit,
                               peel_limit=self.peel_limit,
                               break_anti_dependence=self.break_anti_dependence,
                               interchange_carry_with_map=self.interchange_carry_with_map,
                               scatter_to_guarded_maps=self.scatter_to_guarded_maps,
                               privatize_scatter_reductions=self.privatize_scatter_reductions,
                               reconstruct_wavefront_nest=self.reconstruct_wavefront_nest,
                               normalize_loop_and_map_origin=self.normalize_loop_and_map_origin,
                               assume_parallel_guards=self.assume_parallel_guards,
                               perfect_loop_nesting=self.perfect_loop_nesting,
                               target=self.target,
                               lift=self.lift,
                               lift_copy=self.lift_copy,
                               semantic_lifting=self.semantic_lifting)
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            sdfg.save(os.path.join(self.dump_dir, '000_input.sdfgz'), compress=True)
        for index, (_label, unit) in enumerate(stages, start=1):
            _assert_self_contained(unit)
            # A Pipeline validates its own members when asked, so scope validation to the
            # sub-pass that actually changed something instead of re-walking the whole SDFG here.
            is_pipeline = isinstance(unit, ppl.Pipeline)
            if self.validate_all and is_pipeline:
                unit.validate_subpasses = True
            result = unit.apply_pass(sdfg, {})
            # apply_pass returns non-None iff it modified the SDFG, so an unchanged SDFG needs no
            # re-validation -- that is what makes validate_all affordable by default.
            if self.validate_all and result is not None and not is_pipeline:
                sdfg.validate()
            # Dump AFTER every stage, including no-ops: a bisect needs a dense index, and a stage
            # that reports no change can still have rewritten the graph (5 units do exactly that).
            if self.dump_dir:
                sdfg.save(os.path.join(self.dump_dir, f'{index:03d}_{_label}.sdfgz'), compress=True)
        disable_openmp_sections(sdfg)
        if self.validate:
            sdfg.validate()
        return len(stages)


def canonicalize(sdfg: SDFG,
                 validate: bool = True,
                 validate_all: bool = True,
                 unroll_limit: int = DEFAULT_UNROLL_LIMIT,
                 peel_limit: Optional[int] = None,
                 break_anti_dependence: Optional[bool] = None,
                 target: str = 'cpu',
                 interchange_carry_with_map: Optional[bool] = None,
                 scatter_to_guarded_maps: Optional[bool] = None,
                 privatize_scatter_reductions: Optional[bool] = None,
                 reconstruct_wavefront_nest: Optional[bool] = None,
                 normalize_loop_and_map_origin: Optional[bool] = None,
                 assume_parallel_guards: bool = False,
                 perfect_loop_nesting: bool = True,
                 specialize_constants: Optional[Dict[str, int]] = None,
                 lift: bool = True,
                 lift_copy: bool = True,
                 semantic_lifting: bool = True,
                 dump_dir: Optional[str] = None) -> SDFG:
    """Canonicalize ``sdfg`` in place and return it.

    One-call recipe analogous to ``auto_optimize``.

    Canonicalization is the FIRST of two stages, and it stops where the target begins: wherever the
    choice between a parallel and a sequential form is open it takes parallel, because that is the
    form a GPU, a vectorizer and a CPU can each still specialize from. Nothing here decides whether
    a scope earns an OpenMP region. That is
    :func:`~dace.transformation.passes.cpu_specialization.pipeline.cpu_specialize`, run afterwards
    (:func:`~dace.transformation.passes.canonicalize.finalize.finalize_for_target` calls it), and it
    is where a map is made sequential again for its size or its nesting.

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
    :param reconstruct_wavefront_nest: Rebuild an imperfect Map-plus-LoopRegion stencil
                                   body (``ReconstructWavefrontNest``) into the single
                                   loop ``WavefrontSkew`` requires, right before it in the
                                   ``loop_fuse`` stage; commits only on a proven skew.
                                   ``None`` (default) -> per-target preset (off on both
                                   targets -- unproven corpus benefit).
    :param normalize_loop_and_map_origin: Rebase every Map range / ``LoopRegion`` counter to a
                                   0-based begin, keeping the stride (``NormalizeLoopAndMapOrigin``),
                                   right before the ``loop_to_x`` stage. ``None`` (default) ->
                                   per-target preset (off on both targets).
    :param assume_parallel_guards: Assume every parallel-guard condition holds --
                                   ``ParallelizeUnderConstraint`` and
                                   ``ScatterToGuardedMaps`` emit only the parallel
                                   Map (no ``if cond: par else: seq`` fallback, no
                                   scatter sort/trap). Unsound if a condition is
                                   violated at runtime; ``False`` (default) keeps
                                   the sound guards.
    :param perfect_loop_nesting: Distribute a loop over its data-independent statement groups
                                 (``PerfectLoopNesting``) so each statement gets its own complete
                                 nest and parallelizes on its own axes. ``True`` (default) since
                                 the pass stopped grouping through ``LoopFission``; the
                                 fission-stage comment records what that changed and which gate
                                 is still outstanding. Pass ``False`` to keep bodies undistributed.
    :param specialize_constants: Optional ``{symbol: value}`` baked in via
                             ``specialize_symbols`` (cloudsc-style, recursive into nested
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
                             lifts (Einsum + Copy/Fill). Default ``True``; the
                             vectorizer sets ``False`` to keep the residual as raw
                             maps (a library node is not vectorizable).
    :returns: The same ``sdfg`` instance, canonicalized.
    """
    # Every stage below recovers loop bounds from STRING-backed properties, which means re-parsing
    # them -- and an unscoped parse mints ``DEFAULT_SYMBOL_TYPE``. Declaring the SDFG's own symbol
    # table as the authority for the whole run is what keeps one name spelled as ONE symbol, so a
    # bound and a descriptor shape still cancel (see ``sympy_to_dace``). Nested SDFGs are covered
    # by name: a nested scope re-declares the same names, and only names this table does not carry
    # keep the default.
    authority = {name: dtype for nested in sdfg.all_sdfgs_recursive() for name, dtype in nested.symbols.items()}
    with symbolic.serialization_symbol_dtypes(authority):
        return canonicalize_under_authority(sdfg, validate, validate_all, unroll_limit, peel_limit,
                                            break_anti_dependence, target, interchange_carry_with_map,
                                            scatter_to_guarded_maps, privatize_scatter_reductions,
                                            reconstruct_wavefront_nest, normalize_loop_and_map_origin,
                                            assume_parallel_guards, perfect_loop_nesting, specialize_constants, lift,
                                            lift_copy, semantic_lifting, dump_dir)


def canonicalize_under_authority(sdfg: SDFG, validate, validate_all, unroll_limit, peel_limit, break_anti_dependence,
                                 target, interchange_carry_with_map, scatter_to_guarded_maps,
                                 privatize_scatter_reductions, reconstruct_wavefront_nest,
                                 normalize_loop_and_map_origin, assume_parallel_guards, perfect_loop_nesting,
                                 specialize_constants, lift, lift_copy, semantic_lifting, dump_dir) -> SDFG:
    """The body of :func:`canonicalize`, run with the SDFG's symbol dtypes already in scope."""
    CanonicalizationPipeline(validate=validate,
                             validate_all=validate_all,
                             unroll_limit=unroll_limit,
                             peel_limit=peel_limit,
                             break_anti_dependence=break_anti_dependence,
                             target=target,
                             interchange_carry_with_map=interchange_carry_with_map,
                             scatter_to_guarded_maps=scatter_to_guarded_maps,
                             privatize_scatter_reductions=privatize_scatter_reductions,
                             reconstruct_wavefront_nest=reconstruct_wavefront_nest,
                             normalize_loop_and_map_origin=normalize_loop_and_map_origin,
                             assume_parallel_guards=assume_parallel_guards,
                             perfect_loop_nesting=perfect_loop_nesting,
                             specialize_constants=specialize_constants,
                             lift=lift,
                             lift_copy=lift_copy,
                             semantic_lifting=semantic_lifting,
                             dump_dir=dump_dir).apply_pass(sdfg, {})
    # The guard stage runs last, so nothing cleans up after it: on kernels whose old entry was
    # empty it leaves a redundant empty state between guard and body, which a second canonicalize
    # then removes -- a difference that is only in run 1.
    EmptyStateElimination().apply_pass(sdfg, {})
    DeadStateElimination().apply_pass(sdfg, {})
    # A pin is redundant once a region has a single source, and passes set it inconsistently, so
    # the same SDFG can serialize two different ``start_block`` values. Leave the entry implicit.
    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, ControlFlowRegion) and len(region.source_nodes()) == 1:
            region._start_block = None
            region._cached_start_block = None
    # Canonicalized output opts in to OpenMP array-section reduction codegen (whole-buffer
    # WCR accumulators of a parallel map -> ``reduction(op:A[0:n])`` instead of per-element
    # atomics; complex via ``declare reduction``). Off by default elsewhere; only provably
    # contiguous cases take the clause, everything else still falls back to atomics.
    for nested in sdfg.all_sdfgs_recursive():
        nested.openmp_array_reductions = True
    # Enforced, not hoped for: the pipeline opts out of omp sections at entry and exit, and a
    # caller who flipped ``compiler.cpu.openmp_sections`` globally must not be able to undo it.
    assert not any(nested.openmp_sections for nested in sdfg.all_sdfgs_recursive())
    return sdfg
