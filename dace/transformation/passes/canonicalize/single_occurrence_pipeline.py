# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""WORK IN PROGRESS: the phased recipe with every pass at exactly one position.

See ``docs/design/canonicalize_phase_order.md``.

Selectable as ``order='single'``. Not at parity with legacy yet: 17 of 321 kernels are less parallel.
"""
from typing import List

from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.distribute_tasklet_into_map import DistributeTaskletIntoMap
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.mapreduce import MapReduceFusion, MapWCRFusion
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
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
from dace.transformation.passes.canonicalize.prune_and_inline_nested_sdfgs import PruneAndInlineNestedSDFGs
from dace.transformation.passes.canonicalize.pipeline import (PrivatizeArraysStage, PrivatizeScalarsStage,
                                                              StructuralCleanup, fold_scalar_slices)
from dace.transformation.passes.canonicalize.reconstruct_wavefront_nest import ReconstructWavefrontNest
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation.passes.canonicalize.require_structured_control_flow import RequireStructuredControlFlow
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
from dace.transformation.passes.canonicalize.reverse_map_traversal import ReverseMapTraversal
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import RevertNonReductionWCR
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
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
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling, ShortLoopUnroll
from dace.transformation.passes.parallelize_loops import ParallelizeLoops
from dace.transformation.passes.parallelize_under_constraint import ParallelizeUnderConstraint
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere
from dace.transformation.passes.privatize_scatter_reduction import PrivatizeScatterReduction
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.propagate_memlets import PropagateMemlets
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.scatter_to_guarded_maps import ScatterToGuardedMaps
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.symbol_ssa import SymbolSSA
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod

from dace.transformation.passes.canonicalize.phased_pipeline import (DERIVE_PARALLELISM, FUSION_INNER_ROUNDS,
                                                                     FUSION_OUTER_ROUNDS, LIFT_SEMANTICS,
                                                                     NORMALIZE_ROUNDS, NORMALIZE_STATEMENTS, RECOMPOSE,
                                                                     PhaseFixpoint, Stages, prepared)

#: Rounds cap of the lifting fixpoint: a lift can emit a loop or map another lift claims.
LIFT_ROUNDS = 2
#: Rounds cap of the derivation fixpoint: round two is the Loop2Map retry on what round one rewrote.
DERIVE_ROUNDS = 3


def cleanup_rail() -> List[ppl.Pass]:
    """The value-preserving tidy that may run between phases: inline single-state nested bodies, forward interstate
    assignments, make iterator names unique, then the structural cleanup."""
    return [
        PruneAndInlineNestedSDFGs(),
        CascadeInterstateEdgeAssignmentsUp(),
        UniqueLoopIterators(assign_loop_iterator_post_value=False),
        StructuralCleanup()
    ]


def labelled(label: str, units: List[ppl.Pass]) -> Stages:
    return [(label, unit) for unit in units]


def statement_round(privatize_scatter_reductions: bool, unroll_limit: int, perfect_loop_nesting: bool, target: str,
                    normalize_loop_and_map_origin: bool) -> List[ppl.Pass]:
    """One round of statement normalization, in the target order: WCR normalization and lowering to loops, reroll,
    untile, IV substitution, split, privatize, invariant motion, dead dataflow, perfect nesting, stride permutation."""
    units: List[ppl.Pass] = []
    # WCR normalization and simplification read the frontend map form, which lowering replaces.
    if privatize_scatter_reductions:
        units += [PrivatizeScatterReduction()]
    units += [NormalizeWCR(), SimplifyPass(), RevertNonReductionWCR()]
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True
    units += [PatternApplyOnceEverywhere([lower_maps]), InlineSDFGs()]
    units += [stage[1] for stage in fold_scalar_slices(NORMALIZE_STATEMENTS)]
    units += [NormalizeNegativeStride(), ConvertLengthOneArraysToScalars(), EliminateTrivialTasklets()]
    units += cleanup_rail()
    units += [RerollUnrolledLoops(), UntileLoops(), FuseConsecutiveLoops()]
    if unroll_limit > 0:
        units += [ShortLoopUnroll(unroll_limit), SymbolSSA()]
    promote = ScalarToSymbolPromotion()
    promote.transients_only = False
    units += [promote, HoistInductionVariableUpdates(), InductionVariableSubstitution(), PropagateIndexSubsets()]
    units += [ForwardStoreToLoad(), SplitStatements()]
    units += [PrivatizeScalarsStage(), PrivatizeArraysStage(), PromoteConstantIndexAccess(), BufferExpansion()]
    units += [LoopInvariantCodeMotion(), MoveLoopInvariantIfUp(require_full_hoist=(target == 'gpu'))]
    units += [ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination()]), DeadCarriedStoreElimination()]
    units += [MaterializeLoopExitSymbols(), DistributeProducerConsumerLoop()]
    if perfect_loop_nesting:
        units += [PerfectLoopNesting(target=target)]
    units += [PatternApplyOnceEverywhere([TrivialLoopElimination()]), LoopStridePermutation()]
    if normalize_loop_and_map_origin:
        units += [NormalizeLoopAndMapOrigin()]
    return units


def normalize_statements_phase(semantic_lifting: bool, lift: bool, privatize_scatter_reductions: bool,
                               unroll_limit: int, perfect_loop_nesting: bool, target: str,
                               normalize_loop_and_map_origin: bool) -> Stages:
    """Normalize Statements: structured control flow and one spelling, then the statement fixpoint."""
    units: List[ppl.Pass] = [ControlFlowRaising(), RequireStructuredControlFlow(), RemoveViews()]
    if semantic_lifting and lift:
        units += [LiftInv()]
    units += [SupplyNumThreads(), CollapseNoOpCast(), RewriteModuloToPyMod(), ContinueToCondition()]
    units += [
        PhaseFixpoint(
            'statements', lambda: statement_round(privatize_scatter_reductions, unroll_limit, perfect_loop_nesting,
                                                  target, normalize_loop_and_map_origin), NORMALIZE_ROUNDS)
    ]
    units += cleanup_rail()
    return labelled(NORMALIZE_STATEMENTS, units)


def lift_semantics_phase(semantic_lifting: bool, lift: bool, lift_copy: bool, interchange_carry_with_map: bool,
                         target: str) -> Stages:
    """Lift Semantics: recognize library operators on the normalized nests."""

    def lift_round() -> List[ppl.Pass]:
        units: List[ppl.Pass] = []
        if semantic_lifting and lift:
            units += [LoopToSymm(), LoopToRankKUpdate()]
        units += [LoopToSymmetrize()]
        if semantic_lifting and lift_copy:
            units += [AssignmentAndCopyKernelToMemsetAndMemcpy()]
        if semantic_lifting and lift:
            units += [LoopToTranspose()]
        units += [
            LoopToEinsum(),
            LoopToReduce(),
            LiftPreprocess(),
            LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target),
            ArgMaxLift(),
            LoopToConditionalReduce(),
            LoopToStreamCompaction(),
            LiftLoopCarriedReduction()
        ]
        return units

    units: List[ppl.Pass] = [PhaseFixpoint('lift', lift_round, LIFT_ROUNDS)]
    units += cleanup_rail()
    return labelled(LIFT_SEMANTICS, units)


def derive_parallelism_phase(peel_limit: int, break_anti_dependence: bool, scatter_to_guarded_maps: bool,
                             assume_parallel_guards: bool, reconstruct_wavefront_nest: bool, semantic_lifting: bool,
                             lift_copy: bool, target: str) -> Stages:
    """Derive Parallelism, as a fixpoint over its levels: Loop2Map, rescue of the refused loops, guarded loops,
    accumulators to reduction loops, wavefront skew. The next round derives again on what the last one rewrote."""

    def derive_round() -> List[ppl.Pass]:
        units: List[ppl.Pass] = [PropagateMemlets(), ParallelizeLoops(propagate=False)]
        units += [PrivatizeScalarsStage(), PrivatizeArraysStage()]
        units += cleanup_rail()
        if peel_limit > 0:
            units += [LoopCarriedRotationSubstitution(peel_limit), BestEffortLoopPeeling(peel_limit)]
        if break_anti_dependence:
            units += [BreakAntiDependence(forward_reads=True)]
        units += [ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards)]
        units += [
            FuseChainedScalarReductions(),
            PinCarriedTopLevelLoops(),
            AccumulatorCopyChainToWCR(),
            RetargetWCRAccumulator()
        ]
        if scatter_to_guarded_maps:
            units += [ScatterToGuardedMaps(assume_no_conflicts=assume_parallel_guards)]
        if target == 'gpu':
            units += [PatternApplyOnceEverywhere([PerfLoopNesting()])]
        if reconstruct_wavefront_nest:
            units += [ReconstructWavefrontNest()]
        if target == 'gpu':
            units += [ppl.Pipeline([ReorderStateForLoopFusion()])]
        units += [FuseLoops(), WavefrontSkew(target=target)]
        return units

    units: List[ppl.Pass] = [PhaseFixpoint('derive', derive_round, DERIVE_ROUNDS)]
    units += cleanup_rail()
    return labelled(DERIVE_PARALLELISM, units)


def recompose_phase(semantic_lifting: bool, lift: bool, lift_copy: bool, target: str) -> Stages:
    """Recompose: (MapFusion x MapCollapse x FuseStates) to a fixpoint inside a fixpoint that re-permutes and fuses
    conditions; hoist map-invariant guards; seal."""

    def inner_round() -> List[ppl.Pass]:
        return [
            PatternApplyOnceEverywhere([MapCollapse()]),
            PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]),
            ppl.Pipeline([FuseMaps()]),
            PatternApplyOnceEverywhere([MapReduceFusion(), MapWCRFusion()])
        ] + cleanup_rail()

    def outer_round() -> List[ppl.Pass]:
        return [
            ReverseMapTraversal(),
            MinimizeStridePermutation(),
            MoveLoopIntoMapGated(target=target),
            PhaseFixpoint('fusion', inner_round, FUSION_INNER_ROUNDS),
            PatternApplyOnceEverywhere([ConditionFusion()]),
            LiftTrivialIf(),
            NormalizeMapBody()
        ]

    units: List[ppl.Pass] = [
        NormalizeWCRSource(),
        EmptyLoopElimination(),
        PatternApplyOnceEverywhere([TrivialMapElimination()]),
        PatternApplyOnceEverywhere([MoveIfIntoMap()])
    ]
    units += [PhaseFixpoint('recompose', outer_round, FUSION_OUTER_ROUNDS)]
    units += [MoveMapInvariantIfUp(require_full_hoist=(target == 'gpu')), InsertAssignTaskletsAtMapBoundary()]
    if semantic_lifting and lift:
        units += [PatternApplyOnceEverywhere([LiftEinsum()])]
    units += [
        RelaxIntegerPowers(),
        PruneEmptyConditionalBranches(),
        PatternApplyOnceEverywhere([RedundantArray()]),
        RematerializeDerivedTemporaries(),
        InlineControlFlowRegions()
    ]
    units += cleanup_rail()
    units += [OptionalArrayInference(), AssumeSymbolConstraints(), NormalizeFloorDivision(), AnnotateLoopKinds()]
    return labelled(RECOMPOSE, units)


def build_single_occurrence_stages(unroll_limit: int, peel_limit: int, break_anti_dependence: bool,
                                   interchange_carry_with_map: bool, scatter_to_guarded_maps: bool,
                                   privatize_scatter_reductions: bool, reconstruct_wavefront_nest: bool,
                                   normalize_loop_and_map_origin: bool, assume_parallel_guards: bool,
                                   perfect_loop_nesting: bool, iv_split_rounds: int, target: str, lift: bool,
                                   lift_copy: bool, semantic_lifting: bool) -> Stages:
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
    prepared([stage[1] for stage in s])
    return s
