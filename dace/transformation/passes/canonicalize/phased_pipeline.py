# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The phased canonicalization recipe: one ordered list of phases, each run once.

See ``PIPELINE.md`` in this package for the phase order, the reason for every position, and the
pass-to-phase table. :func:`build_phased_stages` returns the same ``(label, pass)`` shape as the
legacy builder, so :class:`~dace.transformation.passes.canonicalize.pipeline.CanonicalizationPipeline`
runs either one; the label is the phase name.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from dace import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.distribute_tasklet_into_map import DistributeTaskletIntoMap
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.mapreduce import MapReduceFusion, MapWCRFusion
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
from dace.transformation.dataflow.prune_connectors import PruneConnectors
from dace.transformation.dataflow.redundant_array import RedundantArray
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy)
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.buffer_expansion import BufferExpansion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import AssumeSymbolConstraints
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp
from dace.transformation.passes.canonicalize.collapse_noop_cast import CollapseNoOpCast
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import EliminateTrivialTasklets
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.forward_store_to_load import ForwardStoreToLoad
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops
from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import (InductionVariableSubstitution,
                                                                                     LoopCarriedRotationSubstitution)
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.loop_to_rank_k_update import LoopToRankKUpdate
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import MaterializeLoopExitSymbols
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting
from dace.transformation.passes.canonicalize.pipeline import (IvSubstitutionFissionFixpoint, PrivatizeArraysStage,
                                                              PrivatizeScalarsStage, PropagateAndPrune,
                                                              StructuralCleanup, TERMINAL_SYMBOL_ROUNDS,
                                                              changed_the_graph, disable_unit_validation,
                                                              fold_scalar_slices, inline_single_state)
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import PruneUnreferencedTransients
from dace.transformation.passes.canonicalize.reconstruct_wavefront_nest import ReconstructWavefrontNest
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation.passes.canonicalize.require_structured_control_flow import RequireStructuredControlFlow
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
from dace.transformation.passes.canonicalize.reverse_map_traversal import ReverseMapTraversal
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import RevertNonReductionWCR
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.fusion_inline import InlineControlFlowRegions, InlineSDFGs
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.loop_to_reduce import (AccumulatorCopyChainToWCR, LoopToReduce, PinCarriedTopLevelLoops,
                                                       RetargetWCRAccumulator)
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling, ShortLoopUnroll
from dace.transformation.passes.parallelize_loops import ParallelizeLoops
from dace.transformation.passes.parallelize_under_constraint import ParallelizeUnderConstraint
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere, PatternMatchAndApplyRepeated
from dace.transformation.passes.privatize_scatter_reduction import PrivatizeScatterReduction
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.propagate_memlets import PropagateMemlets
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.scalar_fission import ArrayFission, ScalarFission
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.scatter_to_guarded_maps import ScatterToGuardedMaps
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.symbol_ssa import SymbolSSA
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod

#: A stage list: ``(phase label, pass)`` pairs in run order.
Stages = List[Tuple[str, ppl.Pass]]

#: Run ``SimplifyPass`` once between the normalize and lift phases.
SIMPLIFY_AFTER_NORMALIZE = True
#: Rounds cap of the inner map-fusion fixpoint (collapse, fuse, fuse states).
FUSION_INNER_ROUNDS = 4
#: Rounds cap of the outer fusion fixpoint (permute, inner fusion, condition fusion).
FUSION_OUTER_ROUNDS = 3


def prepared(units: List[ppl.Pass]) -> List[ppl.Pass]:
    """Switch off per-unit validation and progress bars, as the legacy builder does for its units."""
    for unit in units:
        disable_unit_validation(unit)
        if isinstance(unit, PatternMatchAndApplyRepeated):
            unit.progress = False
    return units


@transformation.explicit_cf_compatible
class PhaseFixpoint(ppl.Pass):
    """Run a fixed list of units in order until one round changes nothing, at most ``max_rounds``.

    Fresh units every round: several units keep per-apply state. A ``SimplifyPass`` member skips
    the confirming tail of its own sweep (see ``IvSubstitutionFissionFixpoint``).
    """

    CATEGORY: str = 'Canonicalization'

    def __init__(self, name: str, factory: Callable[[], List[ppl.Pass]], max_rounds: int) -> None:
        super().__init__()
        self.name = name
        self.factory = factory
        self.max_rounds = max_rounds

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def units(self) -> List[ppl.Pass]:
        """One round's fresh units, in order."""
        return prepared(self.factory())

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """:returns: The number of rounds that changed the graph, or ``None``."""
        rounds = 0
        for _ in range(self.max_rounds):
            changed = False
            for unit in self.units():
                if isinstance(unit, SimplifyPass):
                    unit_changed = IvSubstitutionFissionFixpoint.simplify_until_settled(unit, sdfg)
                else:
                    unit_changed = changed_the_graph(unit, unit.apply_pass(sdfg, {}))
                changed = changed or unit_changed
            if not changed:
                break
            rounds += 1
        return rounds or None


def privatization_units() -> List[ppl.Pass]:
    """Split one container into per-scope versions where a dominating write allows it."""
    return [
        PrivatizeScalarsStage(),
        PrivatizeArraysStage(),
        ppl.Pipeline([ScalarFission()]),
        ppl.Pipeline([ArrayFission()]),
        PromoteConstantIndexAccess(),
        BufferExpansion(),
    ]


def loop_to_map_units() -> List[ppl.Pass]:
    """Loop2Map on every loop it accepts, with the scope summaries it reads rebuilt first."""
    return [PropagateMemlets(), ParallelizeLoops(propagate=False), PatternApplyOnceEverywhere([PruneConnectors()])]


def ingest_phase(semantic_lifting: bool, lift: bool, privatize_scatter_reductions: bool) -> Stages:
    """Phase 0: structured control flow, one spelling per statement, every map lowered to a loop.

    The recognizers here match the FRONTEND map / nested-SDFG shape, which lowering destroys.
    """
    label = 'ingest'
    s: Stages = [(label, ControlFlowRaising()), (label, RequireStructuredControlFlow()), (label, RemoveViews())]
    if semantic_lifting and lift:
        s += [(label, LoopToSymm()), (label, LiftInv())]
    if privatize_scatter_reductions:
        s += [(label, PrivatizeScatterReduction())]
    s += [(label, NormalizeWCR()), (label, SupplyNumThreads())]
    s += [(label, CollapseNoOpCast()), (label, RewriteModuloToPyMod()), (label, NormalizeNegativeStride()),
          (label, UniqueLoopIterators(assign_loop_iterator_post_value=False)), (label, ContinueToCondition()),
          (label, SimplifyPass())]
    if semantic_lifting and lift:
        s += [(label, LoopToSymm()), (label, LoopToRankKUpdate())]
    s += [(label, ConvertLengthOneArraysToScalars())]
    s += [(label, RevertNonReductionWCR())]
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True
    s += [(label, PatternApplyOnceEverywhere([lower_maps])), (label, PatternApplyOnceEverywhere([PruneConnectors()])),
          (label, InlineSDFGs())]
    s += fold_scalar_slices(label)
    s += [(label, NormalizeNegativeStride()), (label, EliminateTrivialTasklets())]
    s += [(label, StructuralCleanup())]
    return s


def normalize_phase(unroll_limit: int, iv_split_rounds: int, perfect_loop_nesting: bool, target: str,
                    normalize_loop_and_map_origin: bool) -> Stages:
    """Phases 1-6: reroll, untile, IV substitution, LICM, WCR normalization, privatization, split,
    perfect nesting, minimum-stride permutation."""
    s: Stages = [('reroll', RerollUnrolledLoops())]
    s += [('untile', UntileLoops()), ('untile', FuseConsecutiveLoops())]
    if unroll_limit > 0:
        s += [('untile', ShortLoopUnroll(unroll_limit)),
              ('untile', UniqueLoopIterators(assign_loop_iterator_post_value=False)), ('untile', SymbolSSA())]

    def statement_round() -> List[ppl.Pass]:
        promote = ScalarToSymbolPromotion()
        promote.transients_only = False
        units: List[ppl.Pass] = [
            promote, SimplifyPass(),
            HoistInductionVariableUpdates(),
            InductionVariableSubstitution()
        ]
        units += [PropagateIndexSubsets(), RemoveUnusedSymbols(), PropagateAndPrune()]
        units += [LoopInvariantCodeMotion(), MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu'))]
        units += [
            ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]),
            DeadCarriedStoreElimination()
        ]
        units += [RevertNonReductionWCR(), NormalizeWCR()]
        units += privatization_units()
        units += [ForwardStoreToLoad(), SplitStatements()]
        return units

    s += [('normalize', PhaseFixpoint('normalize', statement_round, iv_split_rounds))]
    s += [('normalize', MaterializeLoopExitSymbols()), ('normalize', StructuralCleanup())]
    label = 'perfect_nest'
    s += [(label, DistributeProducerConsumerLoop()), (label, MoveIfIntoLoop()),
          (label, CascadeInterstateEdgeAssignmentsUp())]
    if perfect_loop_nesting:
        s += [(label, PerfectLoopNesting(target=target))]
    s += [(label, UniqueLoopIterators(assign_loop_iterator_post_value=False)),
          (label, PatternApplyOnceEverywhere([TrivialLoopElimination()]))]
    s += [('permute', LoopStridePermutation())]
    if normalize_loop_and_map_origin:
        s += [('permute', NormalizeLoopAndMapOrigin())]
    if SIMPLIFY_AFTER_NORMALIZE:
        s += [('simplify', SimplifyPass())]
    return s


def lift_phase(semantic_lifting: bool, lift: bool, lift_copy: bool, interchange_carry_with_map: bool,
               target: str) -> Stages:
    """Phase 7: recognize library operators on the normalized loop nests."""
    label = 'lift'
    s: Stages = fold_scalar_slices(label)
    s += [(label, LoopToSymmetrize())]
    if semantic_lifting and lift_copy:
        s += [(label, AssignmentAndCopyKernelToMemsetAndMemcpy())]
    if semantic_lifting and lift:
        s += [(label, LoopToTranspose())]
    s += [(label, LoopToEinsum()), (label, RevertNonReductionWCR()), (label, LoopToReduce()), (label, LiftPreprocess()),
          (label, LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target)),
          (label, ArgMaxLift()), (label, LoopToConditionalReduce()), (label, LoopToStreamCompaction()),
          (label, LiftLoopCarriedReduction())]
    s += [(label, StructuralCleanup())]
    return s


def parallelize_phase(peel_limit: int, break_anti_dependence: bool, scatter_to_guarded_maps: bool,
                      assume_parallel_guards: bool, target: str) -> Stages:
    """Phase 8: Loop2Map, rescue of the refused loops, Loop2Map retry, guarded and reduction lifts."""
    label = 'loop2map'
    s: Stages = [(label, CascadeInterstateEdgeAssignmentsUp()),
                 (label, UniqueLoopIterators(assign_loop_iterator_post_value=False))]
    s += [(label, unit) for unit in loop_to_map_units()]
    label = 'rescue'
    if peel_limit > 0:
        s += [(label, LoopCarriedRotationSubstitution(peel_limit)), (label, BestEffortLoopPeeling(peel_limit))]
    if break_anti_dependence:
        s += [(label, BreakAntiDependence(forward_reads=True)), (label, BreakAntiDependence())]
    s += [(label, ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]))]
    s += [(label, PrivatizeScalarsStage()), (label, PrivatizeArraysStage()), (label, SymbolPropagation()),
          (label, ConstantPropagation())]
    s += [(label, CascadeInterstateEdgeAssignmentsUp()),
          (label, UniqueLoopIterators(assign_loop_iterator_post_value=False))]
    s += [(label, unit) for unit in loop_to_map_units()]
    label = 'loop2map_guarded'
    s += [(label, ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards))]
    label = 'loop2map_reduction'
    s += [(label, FuseChainedScalarReductions()), (label, RevertNonReductionWCR()), (label, PinCarriedTopLevelLoops()),
          (label, AccumulatorCopyChainToWCR()), (label, RetargetWCRAccumulator())]
    s += [(label, PropagateMemlets()), (label, ParallelizeLoops(propagate=False))]
    s += [(label, PrivatizeScalarsStage()), (label, PrivatizeArraysStage())]
    s += inline_single_state(label)
    s += [(label, StructuralCleanup()), (label, NormalizeWCR())]
    if scatter_to_guarded_maps:
        s += [('loop2map_guarded', ScatterToGuardedMaps(assume_no_conflicts=assume_parallel_guards))]
    if target == 'gpu':
        s += [(label, PatternApplyOnceEverywhere([PerfLoopNesting()]))]
    return s


def wavefront_phase(reconstruct_wavefront_nest: bool, target: str) -> Stages:
    """Phase 9: skew the sequential nests Loop2Map refused, then map the exposed inner axis."""
    label = 'wavefront'
    s: Stages = []
    if reconstruct_wavefront_nest:
        s += [(label, RevertNonReductionWCR())]
        s += inline_single_state(label)
        s += [(label, ReconstructWavefrontNest())]
    if target == 'gpu':
        s += [(label, ppl.Pipeline([ReorderStateForLoopFusion()]))]
    s += [(label, FuseLoops()), (label, WavefrontSkew(target=target))]
    s += [(label, PropagateMemlets()), (label, ParallelizeLoops(propagate=False))]
    s += inline_single_state(label)
    return s


def lift_maps_phase(semantic_lifting: bool, lift: bool, lift_copy: bool) -> Stages:
    """Phase 7 again, on maps: the operators whose recognizers only match the fused map form."""
    label = 'lift_maps'
    s: Stages = [(label, InsertAssignTaskletsAtMapBoundary())]
    s += inline_single_state(label)
    s += [(label, PropagateMemlets())]
    if semantic_lifting and lift_copy:
        s += [(label, AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += inline_single_state(label)
    if semantic_lifting and lift:
        s += [(label, PatternApplyOnceEverywhere([LiftEinsum()]))]
    return s


def fusion_phase(target: str) -> Stages:
    """Phase 10: (MapFusion x MapCollapse x FuseStates) to a fixpoint, inside an outer fixpoint that
    re-permutes and fuses conditions."""
    label = 'fusion'
    s: Stages = [(label, NormalizeWCRSource()), (label, RevertNonReductionWCR())]
    s += [(label, CascadeInterstateEdgeAssignmentsUp()), (label, EmptyStateElimination()),
          (label, PatternApplyOnceEverywhere([TrivialMapElimination()])), (label, EmptyLoopElimination()),
          (label, PatternApplyOnceEverywhere([MoveIfIntoMap()]))]
    s += inline_single_state(label)
    s += [(label, StructuralCleanup())]

    def inner_round() -> List[ppl.Pass]:
        units: List[ppl.Pass] = [
            PatternApplyOnceEverywhere([MapCollapse()]),
            PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]),
            ppl.Pipeline([FuseMaps()]),
            PatternApplyOnceEverywhere([MapReduceFusion(), MapWCRFusion()])
        ]
        units += [unit for _, unit in inline_single_state(label)]
        units += [StructuralCleanup()]
        return units

    def outer_round() -> List[ppl.Pass]:
        units: List[ppl.Pass] = [
            ReverseMapTraversal(),
            MinimizeStridePermutation(),
            MoveLoopIntoMapGated(target=target)
        ]
        units += [unit for _, unit in inline_single_state(label)]
        units += [PhaseFixpoint('fusion_inner', inner_round, FUSION_INNER_ROUNDS)]
        units += [PatternApplyOnceEverywhere([ConditionFusion()]), LiftTrivialIf(), NormalizeMapBody()]
        units += [unit for _, unit in inline_single_state(label)]
        return units

    s += [(label, PhaseFixpoint('fusion', outer_round, FUSION_OUTER_ROUNDS))]
    s += [(label, MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu'))),
          (label, MoveMapInvariantIfUp(require_full_hoist=(target == 'gpu')))]
    s += inline_single_state(label)
    return s


def seal_phase() -> Stages:
    """Phase 11: reclaim what the phases left dead, and record the facts the output relies on."""
    label = 'seal'
    s: Stages = [(label, RelaxIntegerPowers())]
    s += [(label, ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]))]
    s += inline_single_state(label)
    s += [(label, PruneEmptyConditionalBranches())]
    s += [(label, PatternApplyOnceEverywhere([RedundantArray()])), (label, RematerializeDerivedTemporaries())]
    s += [(label, StructuralCleanup())]
    for _ in range(TERMINAL_SYMBOL_ROUNDS):
        s += [(label, SymbolDedup()), (label, SymbolPropagation()), (label, ConstantPropagation())]
    s += [(label, RemoveUnusedSymbols()), (label, OptionalArrayInference()), (label, ConvertLengthOneArraysToScalars()),
          (label, PruneUnreferencedTransients()), (label, InlineControlFlowRegions()),
          (label, AssumeSymbolConstraints()), (label, NormalizeFloorDivision()), (label, AnnotateLoopKinds()),
          (label, ppl.Pipeline([ArrayElimination()]))]
    return s


def build_phased_stages(unroll_limit: int, peel_limit: int, break_anti_dependence: bool,
                        interchange_carry_with_map: bool, scatter_to_guarded_maps: bool,
                        privatize_scatter_reductions: bool, reconstruct_wavefront_nest: bool,
                        normalize_loop_and_map_origin: bool, assume_parallel_guards: bool, perfect_loop_nesting: bool,
                        iv_split_rounds: int, target: str, lift: bool, lift_copy: bool,
                        semantic_lifting: bool) -> Stages:
    """The phased recipe, with the knobs of the legacy builder.

    :returns: ``(phase label, pass)`` pairs in run order, fresh instances each call.
    """
    s = ingest_phase(semantic_lifting, lift, privatize_scatter_reductions)
    s += normalize_phase(unroll_limit, iv_split_rounds, perfect_loop_nesting, target, normalize_loop_and_map_origin)
    s += lift_phase(semantic_lifting, lift, lift_copy, interchange_carry_with_map, target)
    s += parallelize_phase(peel_limit, break_anti_dependence, scatter_to_guarded_maps, assume_parallel_guards, target)
    s += wavefront_phase(reconstruct_wavefront_nest, target)
    s += fusion_phase(target)
    s += lift_maps_phase(semantic_lifting, lift, lift_copy)
    s += seal_phase()
    prepared([unit for _, unit in s])
    return s
