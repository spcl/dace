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

#: Phase labels, in the paper's names and order (Fig. "fig:phases" and appendix "The Schedule in Detail").
NORMALIZE_STATEMENTS = 'normalize_statements'
LIFT_SEMANTICS = 'lift_semantics'
DERIVE_PARALLELISM = 'derive_parallelism'
RECOMPOSE = 'recompose'

#: Rounds cap of the statement-normalization fixpoint.
NORMALIZE_ROUNDS = 3
#: The unit that decides whether the statement-normalization fixpoint takes another round. Invariant motion is
#: the rewrite that re-exposes work for splitting; a round in which it changed nothing ends the fixpoint.
NORMALIZE_DRIVER = 'LoopInvariantCodeMotion'
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
    """Run a fixed list of units in order until a round changes nothing, at most ``max_rounds``.

    With ``drivers`` set, only a change by a unit of one of those class names earns another round.
    Fresh units every round: several units keep per-apply state. A ``SimplifyPass`` member skips
    the confirming tail of its own sweep (see ``IvSubstitutionFissionFixpoint``).
    """

    CATEGORY: str = 'Canonicalization'

    def __init__(self, name: str, factory: Callable[[], List[ppl.Pass]], max_rounds: int,
                 drivers: Tuple[str, ...] = ()) -> None:
        super().__init__()
        self.name = name
        self.factory = factory
        self.max_rounds = max_rounds
        self.drivers = drivers

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
            again = False
            for unit in self.units():
                if isinstance(unit, SimplifyPass):
                    unit_changed = IvSubstitutionFissionFixpoint.simplify_until_settled(unit, sdfg)
                else:
                    unit_changed = changed_the_graph(unit, unit.apply_pass(sdfg, {}))
                changed = changed or unit_changed
                again = again or (unit_changed and (not self.drivers or type(unit).__name__ in self.drivers))
            if changed:
                rounds += 1
            if not again:
                break
        return rounds or None


def privatization_units() -> List[ppl.Pass]:
    """Split one container into per-scope versions where a dominating write allows it."""
    return [PrivatizeScalarsStage(), PrivatizeArraysStage(), PromoteConstantIndexAccess(), BufferExpansion()]


def loop_to_map_units() -> List[ppl.Pass]:
    """Loop2Map on every loop it accepts, with the scope summaries it reads rebuilt first."""
    return [PropagateMemlets(), ParallelizeLoops(propagate=False), PatternApplyOnceEverywhere([PruneConnectors()])]


def dead_dataflow_units() -> List[ppl.Pass]:
    """Dead-code elimination, the one permitted reduction in work."""
    return [ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]), DeadCarriedStoreElimination()]


def statement_round(target: str) -> List[ppl.Pass]:
    """One round of statement normalization: IV substitution, split, privatize, invariant motion, WCR
    normalization, dead dataflow.

    Split precedes invariant motion because motion hoists whole statements. Privatization precedes it too, so a
    split temporary rewritten every iteration is not hoisted.
    """
    promote = ScalarToSymbolPromotion()
    promote.transients_only = False
    units: List[ppl.Pass] = [promote, SimplifyPass(), HoistInductionVariableUpdates(), InductionVariableSubstitution()]
    units += [PropagateIndexSubsets(), RemoveUnusedSymbols(), PropagateAndPrune()]
    units += [ForwardStoreToLoad(), SplitStatements()]
    units += privatization_units()
    units += [LoopInvariantCodeMotion(), MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu'))]
    units += [RevertNonReductionWCR(), NormalizeWCR()]
    units += dead_dataflow_units()
    return units


def normalize_statements_phase(semantic_lifting: bool, lift: bool, privatize_scatter_reductions: bool,
                               unroll_limit: int, perfect_loop_nesting: bool, target: str,
                               normalize_loop_and_map_origin: bool) -> Stages:
    """Normalize Statements: lower to loops, reroll, untile, the statement fixpoint, perfect nesting, permutation.

    Also holds the recognizers that match the frontend map / nested-SDFG shape (labelled ``lift_semantics``):
    lowering and splitting consume that shape.
    """
    label = NORMALIZE_STATEMENTS
    s: Stages = [(label, ControlFlowRaising()), (label, RequireStructuredControlFlow()), (label, RemoveViews())]
    if semantic_lifting and lift:
        s += [(LIFT_SEMANTICS, LoopToSymm()), (LIFT_SEMANTICS, LiftInv())]
    if privatize_scatter_reductions:
        s += [(label, PrivatizeScatterReduction())]
    s += [(label, NormalizeWCR()), (label, SupplyNumThreads())]
    s += [(label, CollapseNoOpCast()), (label, RewriteModuloToPyMod()), (label, NormalizeNegativeStride()),
          (label, UniqueLoopIterators(assign_loop_iterator_post_value=False)), (label, ContinueToCondition()),
          (label, SimplifyPass())]
    if semantic_lifting and lift:
        s += [(LIFT_SEMANTICS, LoopToSymm()), (LIFT_SEMANTICS, LoopToRankKUpdate())]
    s += [(label, ConvertLengthOneArraysToScalars()), (label, RevertNonReductionWCR())]
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True
    s += [(label, PatternApplyOnceEverywhere([lower_maps])), (label, PatternApplyOnceEverywhere([PruneConnectors()])),
          (label, InlineSDFGs())]
    s += fold_scalar_slices(label)
    s += [(label, NormalizeNegativeStride()), (label, EliminateTrivialTasklets()), (label, StructuralCleanup())]

    s += [(label, RerollUnrolledLoops()), (label, UntileLoops()), (label, FuseConsecutiveLoops())]
    if unroll_limit > 0:
        s += [(label, ShortLoopUnroll(unroll_limit)),
              (label, UniqueLoopIterators(assign_loop_iterator_post_value=False)), (label, SymbolSSA())]
    s += [(label,
           PhaseFixpoint('statements',
                         lambda: statement_round(target),
                         NORMALIZE_ROUNDS,
                         drivers=(NORMALIZE_DRIVER, 'MoveLoopInvariantIfUp')))]
    s += [(label, MaterializeLoopExitSymbols()), (label, StructuralCleanup())]

    s += [(label, DistributeProducerConsumerLoop()), (label, MoveIfIntoLoop()),
          (label, CascadeInterstateEdgeAssignmentsUp())]
    if perfect_loop_nesting:
        s += [(label, PerfectLoopNesting(target=target))]
    s += [(label, UniqueLoopIterators(assign_loop_iterator_post_value=False)),
          (label, PatternApplyOnceEverywhere([TrivialLoopElimination()]))]
    s += [(label, LoopStridePermutation())]
    if normalize_loop_and_map_origin:
        s += [(label, NormalizeLoopAndMapOrigin())]
    s += [(label, SimplifyPass())]
    return s


def lift_semantics_phase(semantic_lifting: bool, lift: bool, lift_copy: bool, interchange_carry_with_map: bool,
                         target: str) -> Stages:
    """Lift Semantics: recognize library operators on the normalized loop nests."""
    label = LIFT_SEMANTICS
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


def derive_parallelism_phase(peel_limit: int, break_anti_dependence: bool, scatter_to_guarded_maps: bool,
                             assume_parallel_guards: bool, reconstruct_wavefront_nest: bool, semantic_lifting: bool,
                             lift_copy: bool, target: str) -> Stages:
    """Derive Parallelism: Loop2Map, rescue of the loops it refused, Loop2Map retry, guarded and reduction loops,
    wavefront skew."""
    label = DERIVE_PARALLELISM
    s: Stages = [(label, CascadeInterstateEdgeAssignmentsUp()),
                 (label, UniqueLoopIterators(assign_loop_iterator_post_value=False))]
    s += [(label, unit) for unit in loop_to_map_units()]
    # Rescue: each rewrite probes Loop2Map and acts only on a loop it refuses; maps are never touched.
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
    # Copy/fill on the maps just minted, before a later cleanup fuses a second writer into their state.
    if semantic_lifting and lift_copy:
        s += [(LIFT_SEMANTICS, AssignmentAndCopyKernelToMemsetAndMemcpy())]
    s += [(label, ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards))]
    # Accumulators become reduction loops, then derive again.
    s += [(label, FuseChainedScalarReductions()), (label, RevertNonReductionWCR()), (label, PinCarriedTopLevelLoops()),
          (label, AccumulatorCopyChainToWCR()), (label, RetargetWCRAccumulator())]
    s += [(label, PropagateMemlets()), (label, ParallelizeLoops(propagate=False))]
    s += [(label, PrivatizeScalarsStage()), (label, PrivatizeArraysStage())]
    s += inline_single_state(label)
    s += [(label, StructuralCleanup()), (label, NormalizeWCR())]
    if scatter_to_guarded_maps:
        s += [(label, ScatterToGuardedMaps(assume_no_conflicts=assume_parallel_guards))]
    if target == 'gpu':
        s += [(label, PatternApplyOnceEverywhere([PerfLoopNesting()]))]
    # Wavefront: skew the nests every level left sequential, then map the exposed axis.
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


def recompose_phase(semantic_lifting: bool, lift: bool, lift_copy: bool, target: str) -> Stages:
    """Recompose: (MapFusion x MapCollapse x FuseStates) to a fixpoint inside a fixpoint that re-permutes and fuses
    conditions; hoist guards; lift what fusion exposed; final simplification and dead code."""
    label = RECOMPOSE
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
        units += [PhaseFixpoint('fusion', inner_round, FUSION_INNER_ROUNDS)]
        units += [PatternApplyOnceEverywhere([ConditionFusion()]), LiftTrivialIf(), NormalizeMapBody()]
        units += [unit for _, unit in inline_single_state(label)]
        return units

    s += [(label, PhaseFixpoint('recompose', outer_round, FUSION_OUTER_ROUNDS))]
    s += [(label, MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu'))),
          (label, MoveMapInvariantIfUp(require_full_hoist=(target == 'gpu')))]
    s += inline_single_state(label)

    # The lifts whose matchers need the fused map form.
    s += [(LIFT_SEMANTICS, InsertAssignTaskletsAtMapBoundary())]
    s += inline_single_state(LIFT_SEMANTICS)
    s += [(LIFT_SEMANTICS, PropagateMemlets())]
    if semantic_lifting and lift_copy:
        s += [(LIFT_SEMANTICS, AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += inline_single_state(LIFT_SEMANTICS)
    if semantic_lifting and lift:
        s += [(LIFT_SEMANTICS, PatternApplyOnceEverywhere([LiftEinsum()]))]

    s += [(label, RelaxIntegerPowers())]
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
    """The phased recipe, with the knobs of the legacy builder. ``iv_split_rounds`` is unused: the statement
    fixpoint has its own cap (:data:`NORMALIZE_ROUNDS`).

    :returns: ``(phase label, pass)`` pairs in run order, fresh instances each call.
    """
    s = normalize_statements_phase(semantic_lifting, lift, privatize_scatter_reductions, unroll_limit,
                                   perfect_loop_nesting, target, normalize_loop_and_map_origin)
    s += lift_semantics_phase(semantic_lifting, lift, lift_copy, interchange_carry_with_map, target)
    s += derive_parallelism_phase(peel_limit, break_anti_dependence, scatter_to_guarded_maps, assume_parallel_guards,
                                  reconstruct_wavefront_nest, semantic_lifting, lift_copy, target)
    s += recompose_phase(semantic_lifting, lift, lift_copy, target)
    prepared([unit for _, unit in s])
    return s
