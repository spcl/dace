# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline.

Rewrites an SDFG into a deterministic canonical form so later passes (fusion,
vectorization, scheduling, equivalence checks) observe one shape per
computation.
"""
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dace import SDFG, data, symbolic, properties
from dace.ordered import OrderedSet
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import helpers as xfh
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds
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
from dace.transformation.passes.canonicalize.require_structured_control_flow import RequireStructuredControlFlow
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
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
from dace.transformation.passes.pattern_matching import (PatternApplyOnceEverywhere, PatternMatchAndApply,
                                                         PatternMatchAndApplyRepeated)
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import (PruneUnreferencedTransients)
from dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination import (
    RedundantOrderingEdgeElimination)
from dace.transformation.passes.fusion_inline import (FuseStates, InlineControlFlowRegions, InlineSDFGs)
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.symbol_ssa import SymbolSSA
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
from dace.transformation.dataflow.redundant_array import RedundantArray
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import EliminateTrivialTasklets
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import RevertNonReductionWCR
from dace.transformation.passes.canonicalize.prune_and_inline_nested_sdfgs import PruneAndInlineNestedSDFGs
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
from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops
from dace.transformation.passes.canonicalize.reconstruct_wavefront_nest import ReconstructWavefrontNest
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.loop_to_rank_k_update import LoopToRankKUpdate
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination

from dace.transformation.passes.parallelize_loops import ParallelizeLoops
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
from dace.transformation.passes.canonicalize.fuse_conditions import FuseConditions
from dace.transformation.dataflow.prune_connectors import PruneConnectors


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

    ``SymbolSSA`` is deliberately NOT here. State fusion does union the interstate assignments of
    the states it merges, so the phase can mint a chain assigning one symbol several times over --
    but versioning those at every boundary buys nothing the runs after ``ShortLoopUnroll`` and at
    ``ssa`` have not already bought, and this phase runs at nine of them.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, StructuralCleanup())]


@properties.make_properties
@transformation.explicit_cf_compatible
class PropagateAndPrune(ppl.Pass):
    """Fold symbols and constants, then drop the dataflow that folding made dead -- TWICE.

    Two rounds, not a fixpoint. The prune is what exposes the second round's propagation, and a
    fixpoint pays a third round that only confirms convergence: on the corpus the third round has
    nothing to fold and nothing to drop, so it is three whole-SDFG walks for no rewrite.
    """

    CATEGORY: str = 'Canonicalization'

    #: Rounds to run. Two is the measured requirement, not a guess.
    ROUNDS: int = 2

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return True

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        changed = 0
        for round_index in range(self.ROUNDS):
            # A Pipeline per round, not bare passes: DeadDataflowElimination declares a
            # ControlFlowBlockReachability dependency that only a Pipeline resolves.
            members = [SymbolPropagation(), ConstantPropagation(), DeadDataflowElimination()]
            result = ppl.Pipeline(members).apply_pass(sdfg, {})
            if result:
                changed += 1
            # The results dict also holds the resolved analyses, so only the members' own keys say the round
            # modified the graph. A round that did not hands every later round the identical graph, which it would
            # repeat verbatim with the same report: count those rounds, skip the work (CloudSC: 2.0s of 5.3s).
            if result and any(type(member).__name__ in result for member in members):
                continue
            if result:
                changed += self.ROUNDS - 1 - round_index
            break
        return changed or None


@properties.make_properties
@transformation.explicit_cf_compatible
class StructuralCleanup(ppl.Pass):
    """The between-phase structural cleanup, as ONE unit.

    One unit rather than nine so the pipeline can skip the whole block when nothing has touched the
    graph since the last one -- the block is spliced in at 8 stage boundaries and on a settled graph
    every member reports no change.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return True

    def depends_on(self):
        return {}

    def units(self) -> List[ppl.Pass]:
        """The block's members, in order. Symbols are folded before the state machine is rewritten,
        then ``FuseStates`` walks the region's edges to fuse what it can.

        ``FuseStates`` now drives the same ``StateFusionExtended.can_be_applied`` a ``PatternApplyOnceEverywhere``
        matcher would, plus ``BlockFusion`` on non-state blocks -- the removed matcher only re-derived
        the same matches through VF2 for a fixpoint the walk reaches alone. Fusion is not confluent, so
        results can still differ (channel_flow: 11 states via the matcher, 12 via the walk).

        ``PruneUnreferencedTransients`` is LAST, after the state deletions above have taken the
        readers with them, so it decides against the block's own final graph. It pays for itself:
        on CloudSC the terminal band alone collects 3857 dead descriptors in 0.53 s, and every
        stage between two cleanups was walking them."""
        walk_fuse = FuseStates()
        walk_fuse.progress = False
        return [
            SymbolDedup(),
            SymbolPropagation(),
            ConstantPropagation(),
            RemoveUnusedSymbols(),
            SymbolDedup(),
            walk_fuse,
            EmptyStateElimination(),
            DeadStateElimination(),
            RedundantOrderingEdgeElimination(),
            PruneUnreferencedTransients(),
        ]

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        changed = 0
        for unit in self.units():
            if unit.apply_pass(sdfg, {}) is not None:
                changed += 1
        return changed or None


def run_structural_cleanup(sdfg: SDFG) -> None:
    """Apply the between-phase structural cleanup once, to a finished SDFG.

    The recipe's own helper, for callers that run their own stage after the pipeline returns
    (:func:`~dace.transformation.passes.canonicalize.finalize.offload_to_gpu` does) and need the
    same tidy-up on the graph they were handed. One source of truth for what "structural cleanup"
    means, rather than a second copy of the list.

    :param sdfg: The SDFG to clean up in place.
    """
    for label, unit in _structural_cleanup('structural_cleanup'):
        unit.apply_pass(sdfg, {})


def _inline_single_state(label: str) -> List[Tuple[str, ppl.Pass]]:
    """Flatten single-state NestedSDFG bodies; un-inlined, they report whole-array memlets and
    every dependence test refuses on the box (seidel_2d). ``PruneConnectors`` shares the fixpoint
    because a dead connector is a hard ``InlineSDFG`` refusal.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, PruneAndInlineNestedSDFGs())]


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
                                     ('coalesce', PatternApplyOnceEverywhere([TrivialMapElimination()])),
                                     ('coalesce', EmptyLoopElimination()),
                                     ('coalesce', PatternApplyOnceEverywhere([MoveIfIntoMap()]))]
    s += _inline_single_state('coalesce')
    s += _structural_cleanup('coalesce')
    # Direction before order: a reversed source loop reaches here as an ascending parameter over
    # DESCENDING addresses, and the permuter scores unit coefficients -- so orient first and it
    # scores the accesses the emitted code will actually make.
    s += [('coalesce', ReverseMapTraversal())]
    s += [('coalesce', MinimizeStridePermutation())]
    s += [('coalesce', PatternApplyOnceEverywhere([MapCollapse()]))]
    s += [('coalesce', PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]))]
    s += [('coalesce', ppl.Pipeline([FuseMaps()]))]
    s += [('coalesce', PatternApplyOnceEverywhere([MapCollapse()]))]
    # 10. structural cleanup AGAIN -- map fusion rebuilds map bodies as fresh single-state
    #     NestedSDFGs, and an un-inlined body hides its precise per-element memlets behind a
    #     whole-array boundary memlet. Every downstream dependence test then reads the bounding
    #     box instead of the real subset and refuses (polybench seidel_2d: LoopFusion saw
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
IV_SPLIT_MAX_ROUNDS = 2

#: Rounds of the terminal ``end`` symbol cleanup (``SymbolDedup`` -> ``SymbolPropagation`` ->
#: ``ConstantPropagation``). Two, so a merge that only the folded spelling exposes is still caught
#: -- ``end`` has no later boundary to catch it. The second round is a guard, NOT a measured
#: reduction: on the two graphs probed for it (canonicalize's own CloudSC output, and the
#: ``scatter_accum_dup`` canary where round 1 merges 2 symbols) round 2 found nothing left to do.
#: It costs three no-op whole-graph walks per compile. Raise the pin only against a case that
#: shows a second round doing work.
TERMINAL_SYMBOL_ROUNDS = 2


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
                                     default=IV_SPLIT_MAX_ROUNDS,
                                     desc='Cap on alternation rounds; the loop breaks earlier when '
                                     'a round changes nothing.')

    def __init__(self, max_rounds: int = IV_SPLIT_MAX_ROUNDS) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
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
                if isinstance(unit, SimplifyPass):
                    unit_changed = self.simplify_until_settled(unit, sdfg)
                else:
                    unit_changed = bool(unit.apply_pass(sdfg, {}))
                changed = changed or unit_changed
            if not changed:
                break
            rounds += 1
        return rounds or None

    @staticmethod
    def simplify_until_settled(simplify: SimplifyPass, sdfg: SDFG) -> bool:
        """``simplify.apply_pass(sdfg, {})`` without the tail of its confirming sweep.

        ``FixedPointPipeline`` repeats whole sweeps until one changes nothing. A pass that reported no
        change on the current graph reports none again, so once every simplification pass has done so
        since the last change the rest of that sweep is skipped. On CloudSC that tail is most of a sweep.

        :param simplify: The round's ``SimplifyPass``; its own ``apply_subpass`` runs every pass.
        :param sdfg: The SDFG to simplify.
        :returns: Whether any simplification pass changed ``sdfg``.
        """
        names = simplify._pass_names
        state: Dict[str, Any] = {}
        settled: Dict[str, None] = {}
        changed = False
        sweep_changed = True
        while sweep_changed and len(settled) < len(names):
            sweep_changed = False
            simplify._modified = ppl.Modifies.Nothing
            for p in simplify.iterate_over_passes(sdfg):
                result = simplify.apply_subpass(sdfg, p, state)
                name = type(p).__name__
                if result is not None:
                    state[name] = result
                    simplify._modified = p.modifies()
                if name not in names:
                    continue
                if result is not None:
                    settled.clear()
                    changed = sweep_changed = True
                    continue
                settled[name] = None
                if len(settled) == len(names):
                    break
        if changed:
            # What SimplifyPass.apply_pass does after a modifying fixpoint.
            xfh.split_interstate_edges(sdfg)
            if simplify.validate and not simplify.validate_all:
                sdfg.validate()
        return changed


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
        privatizer = self.privatizer()
        result = privatizer.apply_pass(sdfg, {})
        if not result:
            return None
        # Report only the privatizer's own rewrites: its resolved analyses would read as a change.
        return {k: v for k, v in result.items() if k in privatizer._pass_names} or None


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


# Per-target knob presets; explicit knob args to ``canonicalize(..., target=...)`` override them.
# Per-target asymmetries cite their A/B test in ``tests/ab_perf/``.
#
# ``interchange_carry_with_map``: ``Loop[jk]{Map[jl]}`` -> ``Map[jl]{Loop[jk]}``, carry per thread.
#   CPU 4.6-5.2x, GPU 0.73-0.80x (BW-bound; short coalesced kernels win) -- test_for_1133_ab.py.
# ``peel_limit``: 4 lifts every peelable TSVC boundary-conflict kernel; higher adds cost only.
# ``break_anti_dependence``: snapshot-rename read-ahead anti-deps. On ``A[i] = A[i+1] + B[i]``
#   CPU off wins 1.16x, GPU on wins 82.6x (test_canon_knobs_ab.py). ON for both: the CPU loss is
#   the trivial-kernel worst case.
# ``scatter_to_guarded_maps``: sort + duplicate guard, ~1.04x CPU / ~1.03x GPU (test_scatter_ab.py).
# ``privatize_scatter_reductions``: whole-buffer map WCR so CPU emits an OpenMP array-section
#   reduction instead of atomics (azimint_hist ~200x -> ~1x vs numpy). CPU only.
# ``reconstruct_wavefront_nest``: rebuild seidel_2d's Map+Loop body for ``WavefrontSkew``. Off:
#   safe but never fires on the corpus yet (range offsets do not line up).
# ``normalize_loop_and_map_origin``: rebase ranges to 0 keeping the stride (``NormalizeLoopsAndMaps``
#   folded it and lost TSVC s172's map); see the AB note at its wiring below.
CPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': True,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': True,
    'reconstruct_wavefront_nest': False,
    'normalize_loop_and_map_origin': False,
}
GPU_DEFAULTS: Dict[str, Any] = {
    'interchange_carry_with_map': False,
    'peel_limit': 4,
    'break_anti_dependence': True,
    'scatter_to_guarded_maps': True,
    'privatize_scatter_reductions': False,
    'reconstruct_wavefront_nest': False,
    'normalize_loop_and_map_origin': False,
}
TARGET_DEFAULTS: Dict[str, Dict[str, Any]] = {'cpu': CPU_DEFAULTS, 'gpu': GPU_DEFAULTS}


def _resolve_target_default(target: str, knob: str, explicit: Optional[Any], fallback: Any) -> Any:
    """Pick ``explicit`` if not ``None``, else the per-target preset, else
    ``fallback``. Used to resolve every per-target knob in one place."""
    if explicit is not None:
        return explicit
    return TARGET_DEFAULTS.get(target, {}).get(knob, fallback)


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
                  iv_split_rounds: int = IV_SPLIT_MAX_ROUNDS,
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
                                       ``CPU_DEFAULTS`` / ``GPU_DEFAULTS``
                                       above): relocate the carry LoopRegion
                                       INTO the per-column Map so the scan runs
                                       sequential-per-thread. On for CPU, off
                                       for GPU.
    :param reconstruct_wavefront_nest: Rebuild an imperfect Map-plus-LoopRegion stencil
                                       body into the single loop ``WavefrontSkew`` requires
                                       (``ReconstructWavefrontNest``), right before it in the
                                       ``loop_fuse`` stage; commits only on a proven skew.
                                       Off by default on both targets (see ``CPU_DEFAULTS``).
    :param normalize_loop_and_map_origin: Rebase every Map range / ``LoopRegion`` counter to a
                                          0-based begin, keeping the stride
                                          (``NormalizeLoopAndMapOrigin``), right before the
                                          ``loop_to_x`` stage -- BEFORE every ``LoopTo*`` lift so
                                          they see the normalized shape. Off by default on both
                                          targets (see ``CPU_DEFAULTS``).

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
    # exists to remove. The rename is scoped to the loop, so a read of the
    # original name AFTER the loop (``C[0] = i``) still gets the epilogue.
    unique_loop_iterators = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    unique_loop_iterators_ssa = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    unique_loop_iterators_fission = UniqueLoopIterators(assign_loop_iterator_post_value=False)
    unique_loop_iterators_unroll = UniqueLoopIterators(assign_loop_iterator_post_value=False)

    # clean: raise structured control flow (required by every stage), then RemoveViews up front so
    # no matcher reasons through a view (library-node operand views are kept for BLAS expansions).
    s += [('clean', ControlFlowRaising()), ('clean', RequireStructuredControlFlow()), ('clean', RemoveViews())]

    # loop_to_symm: lift the polybench symm MAP form (triangular self-scatter ``C[0:i, j]`` plus
    # point write) to ``Symm``. Must precede normalize_reduction, which rewrites that boundary WCR.
    if semantic_lifting and lift:
        s += [('loop_to_symm', LoopToSymm())]

    # lift_inv: ``solve(A, eye(N))`` -> ``Inv``. Runs on the raw frontend shape, before MapToForLoop
    # and ITE lowering rewrite the identity-construction map.
    if semantic_lifting and lift:
        s += [('lift_inv', LiftInv())]

    # privatize_scatter: surface ``hist[bin[i]] (+)= w[i]`` as a whole-buffer map-exit WCR so CPU
    # emits an OpenMP array-section reduction (~200x on azimint_hist). Must precede NormalizeWCR,
    # whose drop-WCR shortcut would otherwise produce a partial plain write. CPU only.
    if privatize_scatter_reductions:
        s += [('privatize_scatter', PrivatizeScatterReduction())]

    # normalize_reduction: an in-nsdfg WCR into a write-only connector -> the seeded-local +
    # map-exit-WCR shape, so downstream treats it as any map-exit reduction. Idempotent.
    s += [('normalize_reduction', NormalizeWCR())]

    # CollapseNoOpCast first: ``__out = dace.float64(__inp)`` with matching dtypes hides a plain copy
    # from every textual matcher, TrivialTaskletElimination included. RewriteModuloToPyMod makes every
    # floored modulo ``py_mod``. NormalizeNegativeStride gives downstream positive strides only.
    # ContinueToCondition turns ``continue`` into a guard before SplitStatements / IVS (``break``
    # loops stay sequential). SimplifyPass's FuseStates collapses multi-state bodies for fission.
    s += [('clean', CollapseNoOpCast()), ('clean', RewriteModuloToPyMod()), ('clean', NormalizeNegativeStride()),
          ('clean', unique_loop_iterators), ('clean', ContinueToCondition()), ('clean', SimplifyPass())]

    # loop_to_rank_k_update: syrk / syr2k nests -> ``Syrk`` / ``Syr2k``. Matches the dataflow of one
    # fused body state, so it runs after the clean block's state fusion and before prep / lower.
    # LoopToSymm runs again for the same reason: the npbench slice form only becomes matchable after
    # fusion. Each form is a no-op for the other.
    if semantic_lifting and lift:
        s += [('loop_to_symm', LoopToSymm()), ('loop_to_rank_k_update', LoopToRankKUpdate())]

    # prep (still maps): SupplyNumThreads defines the thread-count symbol in the IR. MoveIfIntoMap,
    # then ConvertLengthOneArraysToScalars (transients only; the ABI keeps its arrays) so the split
    # sees one spelling. ForwardStoreToLoad right before SplitStatements: feeding an in-iteration
    # reread from the store leaves one crossing array the split can order (TSVC s323).
    s += [('prep', SupplyNumThreads())]
    s += [('prep', PatternApplyOnceEverywhere([MoveIfIntoMap()])), ('prep', ConvertLengthOneArraysToScalars()),
          ('prep', ForwardStoreToLoad()), ('prep', SplitStatements())]
    # Distribute first: the split removes the anti-dependence where reader and writer separate.
    if break_anti_dependence:
        s += [('prep', BreakAntiDependence(forward_reads=True))]

    # Revert conflict-free WCRs while maps are still maps; the remaining WCRs are true reductions
    # MapToForLoop keeps parallel.
    s += [('lower', RevertNonReductionWCR())]
    # lower: every map -> LoopRegion. State-local matching avoids a quadratic re-walk
    # (warpx_field_gather lowers 3300 maps).
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True  # canon preference, off in the transformation's default contract
    s += [('lower', PatternApplyOnceEverywhere([lower_maps], state_local=True))]
    # The pipeline's only ``InlineMultistateSDFG``: lowering mints the nestings here.
    s += [('lower', PatternApplyOnceEverywhere([PruneConnectors()]))]
    s += [('lower', InlineSDFGs())]
    s += _fold_scalar_slices('lower')
    # The cleanup's EmptyStateElimination splices out MapToForLoop's empty boundary states, which
    # would send MoveIfIntoLoop down its imperfect path.
    s += _structural_cleanup('lower')

    # Again for loops that were maps at 'clean'.
    s += [('lower', NormalizeNegativeStride())]

    # reroll: a step-``S`` loop of ``m`` equally spaced lanes -> a step-``g`` loop, while the loop
    # is still in step-``S`` form.
    s += [('reroll', RerollUnrolledLoops())]

    # reduce: drop copy tasklets, revert WCRs to augassign, privatize + propagate symbols and
    # constants; then (below) untile, unroll short loops, IVS before LICM, SimplifyPass, and
    # loop_to_reduce. AugAssignToWCR is deliberately absent: reductions become Reduce nodes.
    s += [('reduce', EliminateTrivialTasklets()), ('reduce', RevertNonReductionWCR()),
          ('reduce', _PrivatizeScalarsStage()), ('reduce', _PrivatizeArraysStage()), ('reduce', SymbolPropagation()),
          ('reduce', ConstantPropagation())]
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
        s += [('reduce', ShortLoopUnroll(unroll_limit)), ('reduce', unique_loop_iterators_unroll)]
        # Version the index symbols the unroll just multiplied: every replay reassigns the same
        # frontend-materialized ``idx = arr[k]`` on the edge feeding its copy, so one name carries
        # one value per replay. That false dependence pins the chain in place -- it is what stops
        # ``MoveIfIntoLoop`` distributing a guard over an unrolled imperfect nest.
        s += [('reduce', SymbolSSA())]
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
    # Privatize false carried deps that block LoopToMap: constant-index slots of a shared array
    # (PCIA) and fully rewritten scratch buffers (BufferExpansion). After ShortLoopUnroll, which can
    # make the pattern concrete; each probes and no-ops unless it unblocks a refusal.
    s += [('reduce', PromoteConstantIndexAccess()), ('reduce', BufferExpansion())]
    # IV fixpoint: PromoteConstInputs, then HoistInductionVariableUpdates fissions IV updates out of
    # compound bodies so single-tasklet IVS closes them (O(N) -> O(1)).
    s += [('reduce', IvSubstitutionFissionFixpoint(max_rounds=iv_split_rounds))]
    # MaterializeLoopExitSymbols after the rounds: it wants the settled set of surviving counters.
    s += [('reduce', MaterializeLoopExitSymbols()), ('reduce', LoopInvariantCodeMotion()), ('reduce', SimplifyPass())]

    # index_subsets: the frontend hides computed indices (``i * inc``) behind interstate symbols that
    # SymbolPropagation does not substitute back, so every affine matcher sees the loop variable as
    # absent. Recover the arithmetic (data-dependent gathers stay), then drop the dead symbols.
    s += [('index_subsets', PropagateIndexSubsets()), ('index_subsets', RemoveUnusedSymbols())]
    # Fold the exposed arithmetic; otherwise reduction_to_wcr_map can leak free ``__tmp_*`` symbols
    # (split_tasklets_test::test_add_missing_symbols_honors_integer_cast).
    s += [('index_subsets', PropagateAndPrune())]

    # cascade_iedges_up (post-reduce): lift invariant iedge assignments (``kfdia_plus_1 = kfdia + 1``)
    # past every enclosing loop for the later body-assigns-range-symbol check.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # distribute: split loops across a forward producer->consumer dependence (atax, covariance) so
    # LoopToEinsum / LoopToSymmetrize see one contraction / pure copy each.
    s += [('distribute', DistributeProducerConsumerLoop())]

    # loop_to_symmetrize: before break_antidep, which would snapshot-rename the in-place copy. The
    # Symmetrize node stays unexpanded for codegen.
    s += [('loop_to_symmetrize', LoopToSymmetrize())]

    # peel / break_antidep: last-resort unblocking before move_if / fission. Both only target loops
    # LoopToMap refuses and only probe ``can_be_applied``. Loop reversal is not a pass: it flips a
    # dependence's direction, never removes one.
    if peel_limit > 0:
        # rotate: a delay line (``x = b[i]`` read one iteration later) -> shifted read, first
        # iteration peeled; shares ``peel_limit``. Before peel so the search sees the DOALL loop.
        s += [('rotate', LoopCarriedRotationSubstitution(peel_limit))]
        s += [('peel', BestEffortLoopPeeling(peel_limit))]
    if break_anti_dependence:
        s += [('break_antidep', BreakAntiDependence())]
    # Re-run the reduce-stage prep on the freshly unblocked loops.
    if peel_limit > 0 or break_anti_dependence:
        s += [('peel', _PrivatizeScalarsStage()), ('peel', _PrivatizeArraysStage()), ('peel', SymbolPropagation()),
              ('peel', ConstantPropagation())]

    # move_if_into_loop: push guards into loop bodies; bare siblings of an inner loop get a trivial
    # loop wrapper, removed at 'untrivialize'. Removal probed but not proven safe (the witnessing
    # suites already had reds at d2be1fde4); PerfectLoopNesting does not subsume it.
    s += [('move_if_into_loop', MoveIfIntoLoop())]

    # Lift assignments MoveIfIntoLoop buried. Collapsing the cascade runs to one was probed green but
    # kept: cloudsc shapes no local gate covers depend on them.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # fission: a second whole-array WAR rename on the fused body (s1244's mixed shape is handled by
    # SplitStatements), PerfectLoopNesting, then unique iterators for the cloned siblings.
    if break_anti_dependence:
        s += [('fission', BreakAntiDependence())]
    # PerfectLoopNesting is on by default (ruling 2026-09-01). The old LoopFission-based grouping
    # miscompiled CloudSC (tendency_loc_a rel=0.13); it now groups Allen-Kennedy at block level and
    # refuses what it cannot prove. OPEN: the CloudSC numerics gate has not been re-run since.
    # ``canonicalize_mixed_parallelism_test`` covers the collapsed-2D-map contract.
    if perfect_loop_nesting:
        s += [('fission', PerfectLoopNesting(target=target))]
    s += [('fission', unique_loop_iterators_fission)]

    # untrivialize: remove MoveIfIntoLoop's trivial-loop wrappers right after fission (which needs
    # them); every matcher up to LoopToMap expects a single-state body and refuses a LoopRegion.
    s += [('untrivialize', PatternApplyOnceEverywhere([TrivialLoopElimination()]))]

    # NormalizeLoopsAndMaps is not wired in: folding the stride into the index blocks LoopToMap on
    # every stride loop (TSVC: 0 gains, -1 map on s172).
    # loop_stride_permutation: interchange a perfect nest so a unit-stride DOALL loop is innermost,
    # turning recurrences into ``for j(seq): for i(parallel)`` without a Scan. Moves DOALL loops only.
    s += [('loop_stride_permutation', LoopStridePermutation())]

    # fuse_consecutive_loops: re-join a hand-tiled main loop and its remainder (``[A, B)`` then
    # ``[B, C)``) so one Reduce covers the range; split, the remainder drops the main partial sum.
    s += [('fuse_consecutive_loops', FuseConsecutiveLoops())]

    # lift_copy_loops: plain copy / zero loops -> Copy / Fill nodes before reduction detection can
    # misread them. Skipped by the vectorizer (``semantic_lifting=False``).
    if semantic_lifting and lift_copy:
        s += [('lift_copy_loops', AssignmentAndCopyKernelToMemsetAndMemcpy())]

    # normalize_origin (knob, off): rebase to 0-based begins right before every LoopTo* lift. Only
    # LoopRegions exist here; the Map half matters for standalone callers.
    if normalize_loop_and_map_origin:
        s += [('normalize_origin', NormalizeLoopAndMapOrigin())]

    # loop_to_x: lift the accumulator / scan / argmax / find-index / conditional-reduce shapes left
    # after LoopFission and LoopStridePermutation, before LoopToMap. Order:
    # - LoopToTranspose first: pure-copy permutations; nothing else lifts them (LiftEinsum refuses
    #   one operand, memcpy lifting rejects permutations). Strictly narrower than LoopToEinsum.
    # - LoopToEinsum before LoopToReduce: a contraction must become one Einsum before its axis
    #   becomes a Reduce. Probes on a copy; no-op otherwise.
    # - RevertNonReductionWCR + LoopToReduce, then LiftPreprocess + LoopToScan: both matchers are
    #   pure, so their prep is explicit here; the extra folds must not precede LoopToReduce.
    # - LoopToConditionalReduce: ``if c: acc OP= x`` -> unconditional masked reduction, so
    #   ``reduction_to_wcr_map`` emits an OpenMP / tree reduction instead of guarded atomics.
    # - LoopToStreamCompaction last: the carried scalar is a write INDEX. Precedes ``parallelize``
    #   (it owns the cursor disjointness proof) and ``reduction_to_wcr_map`` (whose pins it honors).
    # Re-fold: the stages since 'lower' mint fresh bridges (tsvc s254).
    s += _fold_scalar_slices('loop_to_x')
    if semantic_lifting and lift:
        s += [('loop_to_x', LoopToTranspose())]
    s += [('loop_to_x', LoopToEinsum()), ('loop_to_x', RevertNonReductionWCR()), ('loop_to_x', LoopToReduce()),
          ('loop_to_x', LiftPreprocess()),
          ('loop_to_x', LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target)),
          ('loop_to_x', ArgMaxLift()), ('loop_to_x', LoopToConditionalReduce()),
          ('loop_to_x', LoopToStreamCompaction())]

    # cascade_iedges_up (pre-parallelize): re-run after fission / normalize rewrite
    # the CFG; MUST precede LoopToMap. Re-unique the iterators (ssa) so the
    # distributed siblings are independent.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp()), ('ssa', unique_loop_iterators_ssa)]
    # Symbol webs get one name each before any dependence question is asked: peeling and fission
    # copy bodies that reassign their own ``idx = arr[k]``, so a loop and its peeled copy
    # share names and the loop appears to export them (CloudSC's ``llfall_index_*`` blocked
    # MoveLoopIntoMap). Every later canon pass may assume one defining web per interstate symbol.
    s += [('ssa', SymbolSSA())]

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
    s += [('parallelize', ParallelizeLoops(propagate=False))]

    # ``LoopToMap`` is where body NestedSDFGs are MINTED, and it derives their connector set from
    # the loop's read/write sets rather than from what the body still uses -- so a statement split
    # or a fission upstream can leave a connector nothing inside reads. That is not cosmetic: the
    # inliner materialises an access node for it in the parent, held by an ordering edge alone, and
    # the next pass to derive read sets from memlets builds a body SDFG without that descriptor and
    # dies looking the node up. ``PruneConnectors`` removes the connector, its outer memlets and the
    # orphaned descriptor; the earlier 'lower' instance runs long before these nodes exist.
    s += [('parallelize', PatternApplyOnceEverywhere([PruneConnectors()]))]

    # GPU: perfect MAP nests for the grid collapse, via the map-side PerfLoopNesting
    # (delegates to MapFission -- the safe, data-parallel distribution; map iterations carry no
    # dependences, so unlike the removed loop-side PerfectLoopNesting no grouping analysis can
    # silently split a recurrence). Runs after LoopToMap, once maps exist.
    if target == 'gpu':
        s += [('parallelize', PatternApplyOnceEverywhere([PerfLoopNesting()]))]

    # parallelize_guarded: a loop LoopToMap refuses only on an algebraic side condition (s171's
    # ``a[i*inc]``, injective iff ``inc != 0``) gets a runtime guard: Map if it holds, loop otherwise.
    # Before reduction_to_wcr_map would claim it. ``assume_parallel_guards`` drops the check.
    s += [('parallelize_guarded', ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards))]

    # reduction_to_wcr_map: scalar accumulator loops (s313, s4115) -> parallel WCR maps with a
    # privatized Scalar accumulator, lowered to an OpenMP ``reduction`` clause. First re-associate
    # chained accumulations into one (s319 ``sum += a[i]; sum += b[i]``).
    s += [('reduction_to_wcr_map', FuseChainedScalarReductions())]
    # Order: AccumulatorCopyChainToWCR destroys the shape LoopToReduce claims and creates the one
    # RetargetWCRAccumulator claims, so it sits between them.
    s += [('reduction_to_wcr_map', RevertNonReductionWCR())]
    # Pin top-level loops LoopToMap refuses on a carried dependence. Nesting alone pins nothing;
    # that CPU question belongs to ``SequentializeUnprofitableParallelScopes``.
    s += [('reduction_to_wcr_map', PinCarriedTopLevelLoops())]
    s += [('reduction_to_wcr_map', AccumulatorCopyChainToWCR())]
    s += [('reduction_to_wcr_map', RetargetWCRAccumulator())]
    # Rebuild the scope summaries LoopToMap reads (see the note at the first parallelize stage).
    s += [('reduction_to_wcr_map', PropagateMemlets())]
    s += [('reduction_to_wcr_map', ParallelizeLoops(propagate=False))]
    # Rename per-scope transients LoopToMap cloned, keeping the fusion same-name guard's list short.
    s += [('reduction_to_wcr_map', _PrivatizeScalarsStage()), ('reduction_to_wcr_map', _PrivatizeArraysStage())]
    s += _inline_single_state('reduction_to_wcr_map')
    s += _structural_cleanup('reduction_to_wcr_map')
    # LoopToMap above outlines the body, trapping the fresh WCR inside the nsdfg; the
    # normalize_reduction run is one band too early to see it (tsvc s4115). Idempotent.
    s += [('reduction_to_wcr_map', NormalizeWCR())]

    # scatter: a runtime sort + collision-count guard per scatter ``idx`` array, then a permissive
    # lift (s491, vas, s4113; +27 maps on the 151-kernel corpus). A collision takes the sequential
    # clone. ``assume_parallel_guards`` skips the guard.
    if scatter_to_guarded_maps:
        s += [('scatter',
               ScatterToGuardedMaps(emit_unparallelized_else_branch=True, assume_no_conflicts=assume_parallel_guards))]

    # post_l2m: plant assign tasklets on map-boundary copies (without them tsvc ``va`` stays a Map
    # instead of a Memcpy), then inline the single-state bodies.
    s += [('post_l2m', InsertAssignTaskletsAtMapBoundary())]
    s += _inline_single_state('post_l2m')
    # Rebuild scope summaries: a stale whole-array box on the map exit makes codegen emit atomics.
    s += [('post_l2m', PropagateMemlets())]

    # coalesce: prepare the graph for maximal map fusion now that the DOALL
    # loops have become maps -- see ``_coalesce`` for the per-step rationale.
    s += _coalesce()

    # loop_fuse: only the sequential residue is still a LoopRegion. ReconstructWavefrontNest (knob)
    # rebuilds a Map-beside-scan body for WavefrontSkew; FuseLoops fuses same-range siblings;
    # WavefrontSkew (its single home) skews residual 2-D nests; LoopToMap maps the exposed axis.
    if reconstruct_wavefront_nest:
        # Revert WCRs before ReconstructWavefrontNest drives MapToForLoop, as in 'lower'. Only now is
        # seidel_2d's slice write per-element and revertible.
        s += [('loop_fuse', RevertNonReductionWCR())]
        # Reverting makes the body inlinable; inlining replaces the whole-array boundary memlet.
        s += _inline_single_state('loop_fuse')
        s += [('loop_fuse', ReconstructWavefrontNest())]
    # GPU only: reorder a disjoint state stranded between two loops so they fuse (one launch).
    # Fires zero times on all four corpora, so it stays GPU-gated.
    if target == 'gpu':
        # Wrapped: it depends on AccessSets and stages run with an empty results dict.
        s += [('loop_fuse', ppl.Pipeline([ReorderStateForLoopFusion()]))]
    s += [('loop_fuse', FuseLoops())]
    s += [('loop_fuse', WavefrontSkew(target=target))]
    # Rebuild the scope summaries LoopToMap reads (see the note at the first parallelize stage).
    s += [('loop_fuse', PropagateMemlets())]
    s += [('loop_fuse', ParallelizeLoops(propagate=False))]
    s += _inline_single_state('loop_fuse')

    # lift_copy: contiguous copy / zero maps -> Copy / Memset (fissioning mixed maps first). Home of
    # unary copies LiftEinsum skips (durbin ``y[i] = z[i]``); matches MapEntry nodes only.
    if semantic_lifting and lift_copy:
        s += [('lift_copy', AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += _inline_single_state('lift_copy')

    # interchange: ``for t { map[i] }`` -> ``map[i] { for t }``, always on GPU (one kernel), on CPU
    # only when it lowers the innermost stride. The only loop<->map stride minimizer.
    s += [('interchange', MoveLoopIntoMapGated(target=target))]
    s += _inline_single_state('interchange')

    # TODO(GPU): sift pre/post statements of an imperfect nest into the inner loop under boundary
    # guards (``if j == 0: pre(i)``), giving a perfect nest to interchange / collapse. Only for loops
    # without cross-iteration dependences. Prior art: PerfectLoopNesting.
    # TODO: wire PrivatizeReductionAccumulator (+ a structural cleanup). Correct standalone on s313,
    # but the trailing cleanup re-fuses its init/writeback states and drops the map.
    # reorder: permute parallel map nests for unit stride; undeducible strides -> no permutation.
    s += [('reorder', MinimizeStridePermutation())]

    # collapse: ``map i: { map j }`` -> ``map[i, j]``, the canonical fully parallel form. Being N-D it
    # no longer fuses horizontally with a 1-D sibling, so mixed-parallelism nests stay apart.
    s += [('collapse', PatternApplyOnceEverywhere([MapCollapse()]))]

    # fuse: ConditionFusion recombines identical guards split by fission, LiftTrivialIf drops the
    # unsatisfiable combinations, cleanup co-locates the maps, then vertical + horizontal fusion.
    s += [('fuse', FuseConditions(matcher_order=True))]
    s += [('fuse', LiftTrivialIf())]
    s += _inline_single_state('fuse')
    s += _structural_cleanup('fuse')
    s += [('fuse', PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]))]
    s += [('fuse', ppl.Pipeline([FuseMaps()]))]

    # A map that only fills a transient for a following reduction is that reduction; fusing drops the
    # buffer (s3113: 4.166 GB at XL). After MapFusion so multi-map producers match whole.
    s += [('fuse', PatternApplyOnceEverywhere([MapReduceFusion(), MapWCRFusion()]))]

    # normalize_map_body: merge sibling NestedSDFGs in one map body so their same-condition guards
    # become adjacent, fuse, and later hoist out of the map at hoist_guards.
    s += [('fuse', NormalizeMapBody())]
    s += [('fuse', FuseConditions(matcher_order=True))]
    s += _inline_single_state('fuse')
    s += _structural_cleanup('fuse')

    # lift: contraction maps (2mm/3mm/gemm) -> ``Einsum`` (one BLAS GEMM each). After fuse (final
    # shape; LoopToReduce leaves 3-input WCR contractions for this), before normalize_wcr (the WCR
    # folds into beta). ``alpha`` becomes an explicit connector. ``lift=False`` keeps the WCR nest;
    # ``semantic_lifting=False`` (vectorizer) skips both map->library lifts.
    if semantic_lifting and lift:
        s += [('lift', PatternApplyOnceEverywhere([LiftEinsum()]))]

    # licm: hoist loop-invariant code (after LoopToMap, on maps).
    s += [('licm', LoopInvariantCodeMotion())]

    # hoist_guards: after fuse, hoist still-invariant config guards (ICON ``istep == 1``, cloudsc
    # ``IWARMRAIN``) outward; this does not undo MoveIfIntoLoop's fusion. CPU hoists as far as it
    # goes; GPU requires a full hoist, since a guard stalled mid-chain splits the kernel.
    s += [('hoist_guards', MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu')))]

    # The map analog (inverse of MoveIfIntoMap): one map copy per branch. Same GPU gate.
    s += [('hoist_guards', MoveMapInvariantIfUp(require_full_hoist=(target == 'gpu')))]

    # normalize_wcr: source every WCR edge from an AccessNode; codegen's WCR branch only fires for
    # scalar CodeNode outputs, so pointer-typed nsdfg outputs would race.
    s += [('normalize_wcr', NormalizeWCRSource())]

    # Revert WCRs that never became genuine reductions; the injectivity gate keeps real ones.
    s += [('revert_nonreduction_wcr', RevertNonReductionWCR())]

    # relax_powers: provable nonnegative integer ``base ** exp`` -> ``ipow`` while the loop ranges
    # that prove it are live. By codegen a folded size (``R**(K-1)``) sits outside any loop and would
    # stay a double ``pow`` (stockham_fft compile error). Value-exact for ``N**2`` sizes.
    s += [('relax_powers', RelaxIntegerPowers())]

    # end: the two reclaimers the former terminal SimplifyPass was needed for, nothing else.
    # DeadDataflowElimination drops SplitStatements replicas orphaned by fission / parallelize.
    # ArrayElimination then folds duplicate ``__map_fusion_<x>`` carriers (fuse_diamond) with the WAR
    # guards bare RedundantSecondArray lacks (TSVC s212). Wrapped: both declare dependencies.
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
    s += [('end', ParallelizeLoops(propagate=False))]
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
    s += [('end', PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]))]
    s += [('end', ppl.Pipeline([FuseMaps()]))]

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
    s += [('end', PatternApplyOnceEverywhere([RedundantArray()]))]

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

    # Terminal symbol cleanup, and NOT a repeat of the cleanup block above. The block runs its
    # symbol phase FIRST and its state machine second, so its own tail -- the two fusions, then
    # empty- and dead-state elimination -- all run after the last SymbolDedup it does. Fusion
    # unions the interstate assignments of the states it merges, which is exactly what mints a
    # duplicate (a fused gather-map body carries ``idx_index`` and ``idx_index_0``, both
    # ``idx[i]``), and elsewhere such a duplicate is cleaned by the symbol phase of the NEXT
    # boundary. ``end`` has no next boundary, so this IS that phase.
    #
    # It has to be here rather than nowhere because a syntactic-comparison consumer still runs
    # behind it: ``revert_nonreduction_wcr`` below asks WCRToAugAssign whether two subsets name the
    # same slot, and two names for one address answer "different slots" -- the failure that turned
    # an indirect accumulate into a guarded scatter and ``std::abort()``ed at run time
    # (``scatter_accum_dup``). Silent and severe, against three passes on a settled graph.
    #
    # TWO rounds, because the round can feed itself: ``SymbolDedup`` merges only definitions that
    # are already syntactically equal, and the folding behind it rewrites those definitions into
    # the spelling that can make the NEXT pair equal. The cleanup block settles its own symbol
    # phase the same way (its ``SymbolDedup`` runs twice), and ``PropagateAndPrune`` is the same
    # two-round shape for dataflow.
    #
    # ``RemoveUnusedSymbols`` last for the same tail reason: fusion and dead-state elimination
    # delete the interstate edges carrying a symbol's last reference, and propagation substitutes a
    # value while leaving its defining name behind. No SimplifyPass runs past the ``reduce`` stage,
    # so this is the ONLY thing that prunes those. BEFORE AssumeSymbolConstraints, which must stay
    # the terminal stage.
    for round_index in range(TERMINAL_SYMBOL_ROUNDS):
        s += [('end', SymbolDedup()), ('end', SymbolPropagation()), ('end', ConstantPropagation())]
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
    s += [('end', RevertNonReductionWCR())]

    # cleanup (terminal): drop transients nothing names any more. The stages above delete a
    # temporary's last reader without deleting its descriptor, and ``ArrayElimination`` -- the only
    # reclaimer that erases a descriptor -- runs well before them and skips ``Scalar`` outright, so
    # the frontend's per-expression scalars (``b_index``, ``a_slice_times_b_slice``) survive a full
    # canonicalize with no node left referring to them. Codegen ignores them; a re-run does not, and
    # neither does anything that reads the serialized SDFG. The cleanup block runs the same pass at
    # every splice point; this occurrence is what covers the stages that follow the last one.
    s += [('end', PruneUnreferencedTransients())]

    # interchange (terminal re-run): the mid-pipeline ``interchange`` stage sees the graph as it
    # stands then, and the stages after it -- fission, the loop lifts, the inlines -- go on
    # producing fresh ``for(seq) { map }`` nests that it has already walked past. MEASURED on TSVC
    # s2233 at LEN_2D=12354: after the whole recipe the graph still held one, and re-running this
    # very pass by hand interchanged it, so the opportunity was there and only the ordering hid it.
    # On GPU each such nest costs one kernel launch PER outer iteration -- 12,346 of them for that
    # kernel -- which is why the re-run earns its place rather than being left to the next recipe.
    # Same pass, same target gate: on CPU it still declines unless the interchange lowers the
    # innermost stride, so nothing moves on a host graph that the first run already settled.
    s += [('end', MoveLoopIntoMapGated(target=target))]
    s += _inline_single_state('end')

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

    # Terminal, and after everything structural for the same reason the two above are: the hints
    # name the loops the RENDERING will show, so they go on the graph the recipe hands back and
    # not on an intermediate one a later pass reshapes. Comments only -- no pass reads them, and
    # the standalone rendering is the one place they are emitted.
    s += [('end', AnnotateLoopKinds())]

    # cleanup (terminal, and LAST): fold the views the inlines above minted. ``InlineSDFG`` gives a
    # sliced connector its own View descriptor, and the reclaim pipeline runs BEFORE
    # ``_inline_single_state`` / ``InlineControlFlowRegions``, so nothing folds what they leave:
    # CloudSC finished canonicalization holding 18 views of ``zpfplsx``, every one a FULL-array
    # alias carrying the base's own shape and strides, though the source never slices it. A view
    # reaches the vectorizer as an alias to reason about instead of the array itself.
    #
    # LAST, not merely after the inlines: placed straight after ``InlineControlFlowRegions`` the
    # pass folds NOTHING -- ``RemoveSliceView.can_be_applied`` refuses all 18 there, and the same
    # pass on the finished graph accepts 13. What it needs is the symbol constraints and the
    # normalized floor divisions the three stages above register, which is what lets
    # ``map_view_to_array`` prove the view maps onto its array. Same pass as the earlier reclaim,
    # so ``_view_fold_breaks_anti_dependence`` keeps a WAR-carrier view standing here unchanged.
    s += [('end', ppl.Pipeline([ArrayElimination()]))]

    # Pipeline does not propagate `progress` to subpasses, so sweep once here instead of at each call site.
    for _, unit in s:
        disable_unit_validation(unit)
        if isinstance(unit, PatternMatchAndApplyRepeated):
            unit.progress = False
    return s


def disable_unit_validation(unit: ppl.Pass) -> None:
    """Turn off a stage unit's own validation, nested pipeline members included.

    This pipeline's ``validate`` / ``validate_all`` own validation: a unit validating the whole SDFG itself
    repeats that walk, and validation is not read-only (``Fill.validate`` drops an unconnected input).
    """
    if isinstance(unit, (PatternMatchAndApply, FuseMaps)):
        unit.validate = False
        unit.validate_all = False
    if isinstance(unit, ppl.Pipeline):
        for member in unit.passes:
            disable_unit_validation(member)


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
    for i, (lbl, unit) in enumerate(_build_stages()):
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
    return lambda: [p for lbl, p in _build_stages()[start:stop]]


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
    or the ``Pipeline`` wrapping ``FuseMaps``). A bare dependency-bearing
    pass placed directly in a stage would silently lose its inputs.

    :param unit: The pass about to be applied.
    :raises AssertionError: If ``unit`` has unresolved dependencies.
    """
    assert not unit.depends_on() or isinstance(
        unit, ppl.Pipeline), (f"{type(unit).__name__} has dependencies but is not a self-resolving "
                              f"Pipeline; wrap it so its depends_on() is satisfied.")


def changed_the_graph(unit: ppl.Pass, result: Any) -> bool:
    """Whether ``result`` reports a rewrite: plain ``Pipeline`` returns its whole dict even when
    unchanged (unlike ``FixedPointPipeline``), which would pin the caller's dirty flag forever."""
    if result is None:
        return False
    if type(unit) is ppl.Pipeline:
        return any(name in unit._pass_names for name in result)
    return True


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

    :param validate: Validate the SDFG once at the end. OFF by default -- validation is a whole-SDFG
                     walk and a measurable share of the pipeline on a large graph.
    :param validate_all: Validate the SDFG after EVERY stage -- a debugging bisect aid, off by
                         default. Set ``validate`` to check the result, and this as well to find
                         WHICH stage broke it.
    :param unroll_limit: Fully unroll constant-trip loops with at most this many
                         iterations (0 disables).
    :param peel_limit: Best-effort loop peeling before parallelize (0 disables;
                       off by default -- the per-loop search is expensive).
    :param break_anti_dependence: Snapshot-rename pure read-ahead anti-dependence
                                  loops before parallelize (off by default).
    :param perfect_loop_nesting: Run ``PerfectLoopNesting`` at the fission stage. ON by
                                 default -- see the ruling at the fission stage below.
    :param target: ``'cpu'`` (default) or ``'gpu'``. Picks the per-target knob
                   preset (see ``CPU_DEFAULTS`` / ``GPU_DEFAULTS``). Any
                   explicit knob argument (e.g. ``interchange_carry_with_map=...``)
                   overrides the preset for that knob.
    :param interchange_carry_with_map: ``LoopToScan`` knob: relocate the carry
                                       ``LoopRegion`` INTO the per-column Map so
                                       the scan runs sequential-per-thread.
                                       ``None`` (default) -> per-target preset.
    :param reconstruct_wavefront_nest: Run ``ReconstructWavefrontNest`` right before
                                       ``WavefrontSkew`` in the ``loop_fuse`` stage.
                                       ``None`` (default) -> per-target preset (off on
                                       both targets; see ``CPU_DEFAULTS``).
    :param normalize_loop_and_map_origin: Run ``NormalizeLoopAndMapOrigin`` right before the
                                          ``loop_to_x`` stage. ``None`` (default) -> per-target
                                          preset (off on both targets; see ``CPU_DEFAULTS``).
    :param specialize_constants: Optional ``{symbol or scalar argument: value}`` map (e.g.
                             CloudSC's ``{'nclv': 5, 'yrecldp_nssopt': 1}``, or a kernel's shape
                             symbols like ``{'Norb': 3}``) baked into the SDFG via
                             ``specialize_symbols`` (recursively, dropping the symbol) or
                             ``specialize_scalars`` (folding the reads, keeping the argument)
                             BEFORE canonicalization -- the same specialization the
                             cloudsc parallelization pipeline does. Symbolic trip
                             counts that become concrete then unroll under
                             ``ShortLoopUnroll``, and concrete matmul extents let
                             ``canonicalize_set_fast_implementations`` pick the inlined
                             ``'pure'`` GEMM for known-small dims. ``None`` leaves every
                             symbol symbolic.
    :param stages: Stage labels to run, in recipe order (see :func:`stage_labels`); ``None``
                   (default) runs every stage. A label that does not occur in the recipe
                   raises ``ValueError``.
    """

    CATEGORY: str = 'Canonicalization'

    validate = properties.Property(
        dtype=bool,
        default=False,
        desc='Validate the SDFG once at the end. OFF by default, with validate_all: validation is a '
        'whole-SDFG walk and it is a measurable share of the pipeline on a large graph. Turn validate on '
        'to check the result, and validate_all as well to find WHICH stage broke it.')
    validate_all = properties.Property(
        dtype=bool,
        default=False,
        desc='Validate the SDFG after EVERY stage that reported a modification, which pinpoints the '
        'stage that produced an invalid SDFG instead of only catching it at the end. OFF by default: '
        'it is one whole-SDFG walk per modifying stage, and the recipe has 235 of them, so on a large '
        'graph it costs more than the passes it watches. Turn it on to bisect a stage that produced '
        'an invalid SDFG; the end-of-pipeline ``validate`` still catches the break either way.')
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
        'Off by default on both targets (unproven corpus benefit; see CPU_DEFAULTS).')
    normalize_loop_and_map_origin = properties.Property(
        dtype=bool,
        default=False,
        desc='Run NormalizeLoopAndMapOrigin right before the loop_to_x stage, rebasing every Map range / '
        'LoopRegion counter to a 0-based begin while keeping the stride. Off by default on both targets '
        '(see CPU_DEFAULTS).')
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
    stages = properties.ListProperty(
        element_type=str,
        allow_none=True,
        default=None,
        desc='Stage labels to run, in recipe order; None (default) runs every stage. A label '
        'absent from the recipe raises ValueError.')

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
                 reconstruct_wavefront_nest: Optional[bool] = None,
                 normalize_loop_and_map_origin: Optional[bool] = None,
                 assume_parallel_guards: bool = False,
                 perfect_loop_nesting: bool = True,
                 specialize_constants: Optional[Dict[str, int]] = None,
                 lift: bool = True,
                 lift_copy: bool = True,
                 semantic_lifting: bool = True,
                 dump_dir: Optional[str] = None,
                 stages: Optional[Sequence[str]] = None):
        if target not in TARGET_DEFAULTS:
            raise ValueError(f"target must be one of {sorted(TARGET_DEFAULTS)}; got {target!r}")
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
        self.stages = list(stages) if stages is not None else None
        self._specialize_constants = specialize_constants or {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def build_stages(self) -> List[Tuple[str, ppl.Pass]]:
        """Build this pipeline's flat recipe, honoring every knob property.

        :returns: ``(stage_label, pass)`` pairs, in recipe order, fresh instances each call.
        """
        return _build_stages(unroll_limit=self.unroll_limit,
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
            from dace.sdfg.utils import specialize_scalars, specialize_symbols
            # A scalar argument (a run-time flag such as CloudSC's ``yrecldp_nssopt``) is data, not a
            # symbol, so it folds through ``specialize_scalars``; the argument stays in the signature.
            scalars = {
                name: value
                for name, value in self._specialize_constants.items() if isinstance(sdfg.arrays.get(name), data.Scalar)
            }
            specialize_scalars(sdfg, scalars)
            specialize_symbols(sdfg, {n: v for n, v in self._specialize_constants.items() if n not in scalars})
        stages = self.build_stages()
        if self.stages is not None:
            known_labels = OrderedSet(label for label, _ in stages)
            unknown = [label for label in self.stages if label not in known_labels]
            if unknown:
                raise ValueError(f'unknown canonicalization stage label(s): {unknown}')
            wanted = OrderedSet(self.stages)
            stages = [(label, unit) for label, unit in stages if label in wanted]
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            sdfg.save(os.path.join(self.dump_dir, '000_input.sdfgz'), compress=True)
        # The cleanup block is spliced in at every stage boundary, and on a settled graph every
        # member of it reports no change. Skip it unless a transformation has touched the graph
        # since the last one; the graph starts dirty, and only NON-cleanup units re-dirty it, so a
        # cleanup that tidies something does not thereby earn the next one.
        dirty = True
        for index, (label, unit) in enumerate(stages, start=1):
            is_cleanup = isinstance(unit, StructuralCleanup)
            if is_cleanup and not dirty:
                continue
            _assert_self_contained(unit)
            # A Pipeline validates its own members when asked, so scope validation to the
            # sub-pass that actually changed something instead of re-walking the whole SDFG here.
            is_pipeline = isinstance(unit, ppl.Pipeline)
            if self.validate_all and is_pipeline:
                unit.validate_subpasses = True
            result = unit.apply_pass(sdfg, {})
            if is_cleanup:
                dirty = False
            elif changed_the_graph(unit, result):
                dirty = True
            # apply_pass returns non-None iff it modified the SDFG, so an unchanged SDFG needs no
            # re-validation -- that is what makes validate_all affordable by default.
            if self.validate_all and result is not None and not is_pipeline:
                sdfg.validate()
            # Dump AFTER every stage, including no-ops: a bisect needs a dense index, and a stage
            # that reports no change can still have rewritten the graph (5 units do exactly that).
            if self.dump_dir:
                sdfg.save(os.path.join(self.dump_dir, f'{index:03d}_{label}.sdfgz'), compress=True)
        disable_openmp_sections(sdfg)
        if self.validate:
            sdfg.validate()
        return len(stages)


def stage_labels(target: str = 'cpu') -> List[str]:
    """Ordered, de-duplicated stage labels of the canonicalization recipe for ``target``.

    Lets a caller slice :func:`canonicalize` -- run every stage up to (not including) a chosen
    label, do its own thing, then run the rest via ``CanonicalizationPipeline(stages=...)``.

    :param target: ``'cpu'`` (default) or ``'gpu'``.
    :returns: Stage labels in first-occurrence recipe order.
    """
    seen = OrderedSet(label for label, _ in CanonicalizationPipeline(target=target).build_stages())
    return list(seen)


def canonicalize(sdfg: SDFG,
                 validate: bool = False,
                 validate_all: bool = False,
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
                 dump_dir: Optional[str] = None,
                 stages: Optional[Sequence[str]] = None) -> SDFG:
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
    :param validate: Validate the SDFG after canonicalization. OFF by default -- validation is a
                     whole-SDFG walk and a measurable share of the pipeline on a large graph.
    :param validate_all: Validate the SDFG after EVERY stage -- a debugging bisect aid, off by
                         default. Set ``validate`` to check the result, and this as well to find
                         WHICH stage broke it.
    :param unroll_limit: Unroll constant-trip loops <= this many iterations (0 disables).
    :param peel_limit: Best-effort loop peeling before parallelize; ``None``
                       (default) -> per-target preset (CPU=4, GPU=4).
    :param break_anti_dependence: Snapshot-rename read-ahead anti-dependence
                                  loops (TSVC s121 shape:
                                  ``a[i] = a[i+1] + b[i]``); ``None``
                                  (default) -> per-target preset
                                  (CPU=True, GPU=True).
    :param target: ``'cpu'`` (default) or ``'gpu'``. Picks the per-target knob
                   preset (see ``CPU_DEFAULTS`` / ``GPU_DEFAULTS``). Explicit
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
    :param specialize_constants: Optional ``{symbol or scalar argument: value}`` baked in via
                             ``specialize_symbols`` / ``specialize_scalars`` (cloudsc-style,
                             recursive into nested SDFGs) before canonicalization, so symbolic trip counts
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
    :param stages: Stage labels to run, in recipe order (see :func:`stage_labels`); ``None``
                   (default) runs every stage. A label that does not occur in the recipe
                   raises ``ValueError``.
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
                                            lift_copy, semantic_lifting, dump_dir, stages)


def canonicalize_under_authority(sdfg: SDFG,
                                 validate,
                                 validate_all,
                                 unroll_limit,
                                 peel_limit,
                                 break_anti_dependence,
                                 target,
                                 interchange_carry_with_map,
                                 scatter_to_guarded_maps,
                                 privatize_scatter_reductions,
                                 reconstruct_wavefront_nest,
                                 normalize_loop_and_map_origin,
                                 assume_parallel_guards,
                                 perfect_loop_nesting,
                                 specialize_constants,
                                 lift,
                                 lift_copy,
                                 semantic_lifting,
                                 dump_dir,
                                 stages: Optional[Sequence[str]] = None) -> SDFG:
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
                             dump_dir=dump_dir,
                             stages=stages).apply_pass(sdfg, {})
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
