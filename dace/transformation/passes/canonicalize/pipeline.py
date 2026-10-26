# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline.

Rewrites an SDFG into a deterministic canonical form so later passes (fusion,
vectorization, scheduling, equivalence checks) observe one shape per
computation.
"""

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.ordered import OrderedSet
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.distribute_tasklet_into_map import DistributeTaskletIntoMap
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.mapreduce import MapReduceFusion, MapWCRFusion
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
from dace.transformation.dataflow.prune_connectors import PruneConnectors
from dace.transformation.dataflow.redundant_array import RedundantArray, RedundantMapReadCopy
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.analysis import canon_stats
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy,
)
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.buffer_expansion import BufferExpansion
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (
    AssumeSymbolConstraints,
    SetSymbolNonnegativeAssumptions,
    declare_nonnegative_symbols,
    set_symbol_nonnegative_assumptions,
)
from dace.transformation.passes.canonicalize.carried_accumulation import CarriedAccumulationToWCR
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import (
    CascadeInterstateEdgeAssignmentsUp,
    HoistAssignmentsToLoopBodyHead,
)
from dace.transformation.passes.canonicalize.collapse_noop_cast import CollapseNoOpCast
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dace.transformation.passes.canonicalize.dead_loop_initialization import DeadLoopInitialization
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import EliminateTrivialTasklets
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.expand_index_remaps import ExpandIndexRemaps
from dace.transformation.passes.canonicalize.fold_scalar_read_copies import FoldScalarReadCopies
from dace.transformation.passes.canonicalize.forward_iota_reads import ForwardIotaReads
from dace.transformation.passes.canonicalize.forward_offset_copies import ForwardOffsetCopies
from dace.transformation.passes.canonicalize.forward_store_to_load import ForwardStoreToLoad
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops
from dace.transformation.passes.canonicalize.fuse_elementwise_into_gemv import FuseElementwiseIntoGemv
from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops
from dace.transformation.passes.canonicalize.fuse_maps_at_shared_axes import FuseMapsAtSharedAxes
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import (
    InductionVariableSubstitution,
    LoopCarriedRotationSubstitution,
)
from dace.transformation.passes.canonicalize.lift_break_to_find_first import LiftBreakToFindFirst
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.loop_to_getrf_nopivot import LoopToGetrfNoPivot
from dace.transformation.passes.canonicalize.loop_to_rank_k_update import LoopToRankKUpdate
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import MaterializeLoopExitSymbols
from dace.transformation.passes.canonicalize.merge_guards import MergeGuards
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from dace.transformation.passes.canonicalize.nest_to_contraction import NestToContraction
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting
from dace.transformation.passes.canonicalize.prune_and_inline_nested_sdfgs import PruneAndInlineNestedSDFGs
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import PruneUnreferencedTransients
from dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination import RedundantOrderingEdgeElimination
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation.passes.canonicalize.require_structured_control_flow import RequireStructuredControlFlow
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
from dace.transformation.passes.canonicalize.reverse_map_traversal import ReverseMapTraversal
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import RevertNonReductionWCR
from dace.transformation.passes.canonicalize.route_ordering_through_data import RouteOrderingThroughData
from dace.transformation.passes.canonicalize.segment_confined_loops import SegmentConfinedLoops
from dace.transformation.passes.canonicalize.select_branch_maps import SelectBranchMaps
from dace.transformation.passes.canonicalize.sink_last_only_scan_store import SinkLastOnlyScanStore
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.canonicalize.supply_num_threads import SupplyNumThreads
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.canonicalize.while_to_for import WhileToFor
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import (
    CleanAccessNodeToScalarSliceToTaskletPattern,
)
from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import (
    CleanTaskletToScalarSliceToAccessNodePattern,
)
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.dead_fill_elimination import DeadFillElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.demote_non_structural_symbols import DemoteNonStructuralSymbols
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination
from dace.transformation.passes.equalize_symbol_dtypes import EqualizeSymbolDtypes, equalize
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.fusion_inline import FuseStates, InlineControlFlowRegions, InlineSDFGs
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.loop_to_reduce import (
    AccumulatorCopyChainToWCR,
    LoopToReduce,
    PinCarriedTopLevelLoops,
    RetargetWCRAccumulator,
)
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.loop_to_state_scan import LoopToStateScan
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.parallelization.contracts.guarded_scatter import ScatterToGuardedMaps
from dace.transformation.passes.parallelization.contracts.under_constraint import ParallelizeUnderConstraint
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops
from dace.transformation.passes.parallelization.sibling_chains import ParallelizeLoopsHoldingSiblingChains
from dace.transformation.passes.parallelization.wavefront.reconstruct_nest import ReconstructWavefrontNest
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew
from dace.transformation.passes.parallelization_prep import (
    DEFAULT_PEEL_LIMIT,
    DEFAULT_UNROLL_LIMIT,
    BestEffortLoopPeeling,
    ShortLoopUnroll,
)
from dace.transformation.passes.pattern_matching import (
    PatternApplyOnceEverywhere,
    PatternMatchAndApply,
    PatternMatchAndApplyRepeated,
)
from dace.transformation.passes.privatize_scatter_reduction import PrivatizeScatterReduction
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.propagate_memlets import PropagateMemlets
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.scalar_fission import PrivatizeArrays, PrivatizeScalars
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.symbol_ssa import SymbolSSA
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod

#: Canonicalization's simplify fuses states with StateFusionExtended's extra merges.
CANON_SIMPLIFY_OPTIONS: dict[str, Any] = {"FuseStates.aggressive_merges": True}


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


def _structural_cleanup(label: str) -> list[tuple[str, ppl.Pass]]:
    """Tidy symbols, then the state machine, between phases; never ``SimplifyPass`` mid-pipeline.

    Symbol phase: ``SymbolDedup``, ``SymbolPropagation``, ``ConstantPropagation``,
    ``RemoveUnusedSymbols``, then ``SymbolDedup`` again LAST (propagation and the prune both mint
    new mergeable pairs; 7 kernels kept one otherwise). It runs at every boundary because two
    names for one address make syntactic subset tests (``AugAssignToWCR``) silently decline; that
    once turned an indirect accumulate into a scatter that aborted at run time
    (``scatter_accum_dup`` is the canary). Dedup costs 0.8% of canonicalize.

    Structural phase: ``StateFusionExtended`` once everywhere (cheap per boundary, not a fixpoint),
    ``RedundantOrderingEdgeElimination`` last, since fusion is what makes ordering edges redundant.
    Fusion unions assignments, so duplicates it mints wait for the next boundary's symbol phase.

    Placed only where a phase ends: ``coalesce`` (x2; map fusion re-nests bodies, seidel_2d),
    ``lower``, ``loop_to_scan``, ``reduction_to_wcr_map``, ``fuse`` (x2) and ``end``. ``SymbolSSA``
    is not part of it: the runs after ``ShortLoopUnroll`` and at ``ssa`` already cover it.

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

    CATEGORY: str = "Canonicalization"

    #: Rounds to run. Two is the measured requirement, not a guess.
    ROUNDS: int = 2

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return True

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
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

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return True

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def units(self) -> list[ppl.Pass]:
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
        walk_fuse = FuseStates(aggressive_merges=True)
        walk_fuse.progress = False
        return [
            EqualizeSymbolDtypes(),
            SymbolDedup(),
            SymbolPropagation(),
            ConstantPropagation(),
            RemoveUnusedSymbols(),
            SymbolDedup(),
            FoldScalarReadCopies(),
            walk_fuse,
            EmptyStateElimination(),
            DeadStateElimination(),
            RedundantOrderingEdgeElimination(),
            # After the state fusion above, so a copy and its readers share a state: redundant copies fold away
            ppl.Pipeline([ArrayElimination(fold_ordered_copies=True)]),
            # A shifted copy array elimination keeps (read in several states) is read at the shifted source index
            ForwardOffsetCopies(),
            PruneUnreferencedTransients(),
        ]

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        changed = 0
        settled = Settled()
        for unit in self.units():
            if settled.skip(unit):
                continue
            unit_changed = changed_the_graph(unit, unit.apply_pass(sdfg, {}))
            settled.record(unit, unit_changed)
            changed += unit_changed
        return changed or None


def run_structural_cleanup(sdfg: SDFG) -> None:
    """Apply the between-phase structural cleanup once, to a finished SDFG.

    The recipe's own helper, for callers that run their own stage after the pipeline returns
    (:func:`~dace.transformation.passes.canonicalize.finalize.offload_to_gpu` does) and need the
    same tidy-up on the graph they were handed. One source of truth for what "structural cleanup"
    means, rather than a second copy of the list.

    :param sdfg: The SDFG to clean up in place.
    """
    for label, unit in _structural_cleanup("structural_cleanup"):
        unit.apply_pass(sdfg, {})


def _inline_single_state(label: str) -> list[tuple[str, ppl.Pass]]:
    """Flatten single-state NestedSDFG bodies; un-inlined, they report whole-array memlets and
    every dependence test refuses on the box (seidel_2d). ``PruneConnectors`` shares the fixpoint
    because a dead connector is a hard ``InlineSDFG`` refusal; ``RouteOrderingThroughData`` runs first
    because an ordering edge to or from a non-scope node is another.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [(label, RouteOrderingThroughData()), (label, PruneAndInlineNestedSDFGs())]


def _fold_scalar_slices(label: str) -> list[tuple[str, ppl.Pass]]:
    """Fold the frontend scalar-slice bridge; behind the transient a matcher has no index to shift,
    which costs tsvc s252 its ``_remat`` clone and its map.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs, in order.
    """
    return [
        (label, CleanAccessNodeToScalarSliceToTaskletPattern()),
        (label, CleanTaskletToScalarSliceToAccessNodePattern()),
    ]


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

    CATEGORY: str = "Canonicalization"

    max_rounds = properties.Property(
        dtype=int,
        default=IV_SPLIT_MAX_ROUNDS,
        desc="Cap on alternation rounds; the loop breaks earlier when a round changes nothing.",
    )

    def __init__(self, max_rounds: int = IV_SPLIT_MAX_ROUNDS) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def round_units(self) -> tuple[ppl.Pass, ...]:
        """The passes one round runs, in order.

        A method rather than a literal buried in ``apply_pass`` because this round OWNS a
        ``SimplifyPass``: a recipe invariant that asks where simplification happens can only be
        checked if the composite says what it contains.
        """
        promote = ScalarToSymbolPromotion()
        promote.transients_only = False
        return (
            promote,
            SimplifyPass(pass_options=CANON_SIMPLIFY_OPTIONS),
            HoistInductionVariableUpdates(),
            InductionVariableSubstitution(),
            SplitStatements(),
        )

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
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
        state: dict[str, Any] = {}
        settled: dict[str, None] = {}
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


class _ThenIfChanged(ppl.Pass):
    """The ``first`` units in order, then the ``then`` units only when a watched one of them rewrote the graph: the
    lift that follows a rewrite re-probes every loop, which a whole-program kernel pays for nothing when there was
    nothing to rewrite.

    :param first: ``(unit, watched)`` pairs; only a watched unit's result decides.
    """

    CATEGORY: str = "Canonicalization"

    def __init__(self, first: list[tuple[ppl.Pass, bool]], then: list[ppl.Pass]) -> None:
        super().__init__()
        self.first = first
        self.then = then

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> bool | None:
        changed = False
        for unit, watched in self.first:
            result = unit.apply_pass(sdfg, {})
            changed = changed or (watched and result is not None)
        if not changed:
            return None
        for unit in self.then:
            unit.apply_pass(sdfg, {})
        return True


# Per-target knob presets; explicit knob args to ``canonicalize(..., target=...)`` override them.
#
# ``interchange_carry_with_map``: ``Loop[jk]{Map[jl]}`` -> ``Map[jl]{Loop[jk]}``, carry per thread.
#   CPU 4.6-5.2x, GPU 0.73-0.80x (BW-bound; short coalesced kernels win).
# ``scatter_to_guarded_maps``: sort + duplicate guard, ~1.04x CPU / ~1.03x GPU.
# ``privatize_scatter_reductions``: whole-buffer map WCR so CPU emits an OpenMP array-section
#   reduction instead of atomics (azimint_hist ~200x -> ~1x vs numpy). CPU only.
# ``reconstruct_wavefront_nest``: rebuild seidel_2d's Map+copy+Loop row body into the one loop
#   ``WavefrontSkew`` needs; commits only where the skew then provably fires. ON for both.
# ``normalize_loop_and_map_origin``: rebase ranges to 0 keeping the stride (``NormalizeLoopsAndMaps``
#   folded it and lost TSVC s172's map); see the AB note at its wiring below.
CPU_DEFAULTS: dict[str, Any] = {
    "interchange_carry_with_map": True,
    "scatter_to_guarded_maps": True,
    "privatize_scatter_reductions": True,
    "reconstruct_wavefront_nest": True,
    "normalize_loop_and_map_origin": False,
}
GPU_DEFAULTS: dict[str, Any] = {
    "interchange_carry_with_map": False,
    "scatter_to_guarded_maps": True,
    "privatize_scatter_reductions": False,
    "reconstruct_wavefront_nest": True,
    "normalize_loop_and_map_origin": False,
}
TARGET_DEFAULTS: dict[str, dict[str, Any]] = {"cpu": CPU_DEFAULTS, "gpu": GPU_DEFAULTS}


def _resolve_target_default(target: str, knob: str, explicit: Any | None, fallback: Any) -> Any:
    """Pick ``explicit`` if not ``None``, else the per-target preset, else
    ``fallback``. Used to resolve every per-target knob in one place."""
    if explicit is not None:
        return explicit
    return TARGET_DEFAULTS.get(target, {}).get(knob, fallback)


#: Indices one find-first chunk searches before the body below its hit runs (``LiftBreakToFindFirst``): on CPU a
#: chunk whose arrays stay in the last-level cache for the body's second read, on GPU about four times a full grid.
FIND_FIRST_CHUNK: dict[str, int] = {"cpu": 1 << 20, "gpu": 1 << 21}


# The recipe
# The phased canonicalization recipe: the paper's phases in order, with a structural cleanup between them.
#
# Phases (mpr paper, Fig. "phases"): normalize statements, lift semantics, derive parallelism, then recompose
# (fusion and the final tidy). Specialization is a separate call (``finalize_for_target``).
#
# - ``StructuralCleanup`` runs at the start and between every two phases, and nowhere else is it the reason a
#   phase ends.
# - ``SimplifyPass`` (itself a fixpoint) runs twice: on the raised input, and once after normalization.
# - A fixpoint of several passes (:class:`PhaseFixpoint`) is allowed inside a phase.
# - Guards are pushed inward (``MoveIfIntoLoop``, ``MoveIfIntoMap``) and never hoisted here: how far up an invariant
#   guard goes is a target decision, made by ``hoist_invariant_guards`` in ``cpu_specialize`` / the GPU
#   specialization.
#
# Units whose matcher needs a shape another phase destroys stay where that shape exists but carry the
# ``lift_semantics`` label, so ``stage_labels`` shows every out-of-phase placement; past maximal fusion they carry
# ``after_fusion`` instead. No label before ``fuse`` recurs after it, so ``stages`` sliced at ``fuse`` (or any
# phase) runs exactly the recipe prefix and suffix.

Stages = list[tuple[str, ppl.Pass]]

NORMALIZE_STATEMENTS = "normalize_statements"
LIFT_SEMANTICS = "lift_semantics"
DERIVE_PARALLELISM = "derive_parallelism"
RECOMPOSE = "recompose"
#: Maximal fusion alone, so a caller can own the fusion decisions (NestForge stage 2).
FUSE = "fuse"
#: Everything recompose does once fusion settled, out-of-phase lifts included.
AFTER_FUSION = "after_fusion"
#: The between-phase rail, and the cleanup of the raised input.
CLEANUP = "cleanup"
#: The terminal bookkeeping.
FINISH = "finish"

#: Round cap of the statement fixpoint; another round only when invariant motion changed the graph.
NORMALIZE_ROUNDS = 3
NORMALIZE_DRIVERS = ("LoopInvariantCodeMotion",)
#: Round caps of the inner (collapse x fuse x fuse states) and outer (permute, inner, condition fusion) fixpoints.
FUSION_INNER_ROUNDS = 4
FUSION_OUTER_ROUNDS = 3


def prepared(units: list[ppl.Pass]) -> list[ppl.Pass]:
    """Switch off per-unit validation and progress bars on every unit."""
    for unit in units:
        disable_unit_validation(unit)
        if isinstance(unit, PatternMatchAndApplyRepeated):
            unit.progress = False
    return units


@transformation.explicit_cf_compatible
class PhaseFixpoint(ppl.Pass):
    """Run fresh instances of a list of units, in order, until a round changes nothing, at most ``max_rounds``.

    With ``drivers`` set, only a change by a unit of one of those class names earns another round. Fresh units
    every round: several units keep per-apply state.
    """

    CATEGORY: str = "Canonicalization"

    def __init__(
        self, name: str, factory: Callable[[], list[ppl.Pass]], max_rounds: int, drivers: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self.name = name
        self.factory = factory
        self.max_rounds = max_rounds
        self.drivers = drivers

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: the number of rounds that changed the graph, or ``None``."""
        rounds = 0
        settled = Settled()
        for round_index in range(self.max_rounds):
            changed = again = False
            for unit in prepared(self.factory()):
                if settled.skip(unit):
                    continue
                unit_changed = changed_the_graph(unit, unit.apply_pass(sdfg, {}))
                settled.record(unit, unit_changed)
                changed = changed or unit_changed
                again = again or (unit_changed and (not self.drivers or type(unit).__name__ in self.drivers))
            rounds += changed
            if not again:
                break
        return rounds or None


def rail(label: str = CLEANUP) -> Stages:
    """The between-phase cleanup: flatten single-state bodies (an un-inlined body reports whole-array memlets),
    then the structural cleanup."""
    return [*_inline_single_state(label), (label, StructuralCleanup())]


def unique_iterators() -> ppl.Pass:
    return UniqueLoopIterators()


def clean_anti_dependency_units() -> list[ppl.Pass]:
    """CleanAntiDependency (WAR, WAW) by renaming: per-scope versions of a container wherever a dominating write
    allows it, per-iteration copies of a constant slot or scratch buffer a loop rewrites before reading. The
    read-ahead WAR a snapshot breaks is the other half, ``BreakAntiDependence`` in :func:`derive_parallelism`: its
    offset proof reads guard symbols that only SymbolSSA and propagation there settle."""
    return [
        PrivatizeScalars(),
        PrivatizeArrays(),
        PromoteConstantIndexAccess(),
        BufferExpansion(),
    ]


def dead_dataflow_units() -> list[ppl.Pass]:
    return [
        ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination(fold_ordered_copies=True)]),
        DeadCarriedStoreElimination(),
    ]


def loop_to_map_units(hold_sibling_chains: bool = False) -> list[ppl.Pass]:
    """LoopToMap on every loop it accepts, the scope summaries it reads rebuilt first.

    :param hold_sibling_chains: Leave a sibling-loop chain with a refused sibling whole, for ``WavefrontSkew``.
    """
    parallelize = ParallelizeLoopsHoldingSiblingChains if hold_sibling_chains else ParallelizeLoops
    return [
        PropagateMemlets(),
        # A scalar argument promoted to a symbol since ingestion is under the contract too (its step orients a map)
        SetSymbolNonnegativeAssumptions(),
        parallelize(propagate=False),
        PatternApplyOnceEverywhere([PruneConnectors()]),
    ]


def statement_round() -> list[ppl.Pass]:
    """One round of statement normalization: IV substitution, split, privatize, invariant motion, WCR
    normalization, dead dataflow. Split and privatization precede invariant motion: it hoists whole statements,
    and a temporary rewritten every iteration must not be hoisted."""
    promote = ScalarToSymbolPromotion()
    promote.transients_only = False
    # Fold symbol copies before the IV matchers: the frontend spells ``k = k + inc`` as ``t = k + inc; k = t``, and
    # induction-variable substitution matches the folded update only (tsvc s318).
    units: list[ppl.Pass] = [promote, PropagateAndPrune()]
    # A slice update ``A[S] += e`` writes ``A[S]`` in place, a read-modify-write ``AccumulatorCopyChainToWCR`` reduces
    units += [EliminateCopyBack()]
    units += [HoistInductionVariableUpdates(), InductionVariableSubstitution()]
    units += [PropagateIndexSubsets(), RemoveUnusedSymbols(), PropagateAndPrune()]
    units += [ForwardStoreToLoad(), SplitStatements()]
    units += [PhaseFixpoint("clean_anti_dependency", clean_anti_dependency_units, 1)]
    units += [LoopInvariantCodeMotion()]
    units += [RevertNonReductionWCR(), NormalizeWCR()]
    units += dead_dataflow_units()
    return units


def ingest(target: str) -> Stages:
    """Raise structured control flow and settle one spelling, then the first ``SimplifyPass`` and cleanup."""
    s: Stages = [
        # The contract's compile-time half from the start: every proof below reads the sizes as nonnegative (a
        # symbolic loop step ``n3`` orients its map); ``AssumeSymbolConstraints`` adds the runtime check at the end
        (CLEANUP, SetSymbolNonnegativeAssumptions()),
        (CLEANUP, ControlFlowRaising()),
        (CLEANUP, RequireStructuredControlFlow()),
        (CLEANUP, RemoveViews()),
        (CLEANUP, EliminateTrivialTasklets()),
        (CLEANUP, FoldScalarReadCopies()),
        (CLEANUP, CollapseNoOpCast()),
        (CLEANUP, RewriteModuloToPyMod()),
        (CLEANUP, NormalizeNegativeStride()),
        (CLEANUP, unique_iterators()),
        (CLEANUP, ContinueToCondition()),
        (CLEANUP, SupplyNumThreads()),
        # np.roll / broadcast_to / np.where as their pure maps from the start, so every normalization below sees a map
        # and not an opaque library node; the SimplifyPass right after inlines their nested SDFGs
        (CLEANUP, ExpandIndexRemaps()),
        (CLEANUP, SimplifyPass(pass_options=CANON_SIMPLIFY_OPTIONS)),
        # A while loop that counts gets its counter here, so every later pass sees a counted loop.
        (CLEANUP, WhileToFor()),
        # After simplification: the raised guard of a break still reads its condition through a separate
        # preparation region before it, which the matcher does not see through.
        (CLEANUP, LiftBreakToFindFirst(chunk=FIND_FIRST_CHUNK[target])),
    ]
    return s + rail()


def normalize_statements(
    lift: bool,
    privatize_scatter_reductions: bool,
    unroll_limit: int,
    perfect_loop_nesting: bool,
    target: str,
    normalize_loop_and_map_origin: bool,
) -> Stages:
    """Lower every map to a loop, reroll, untile, unroll tiny loops, the statement fixpoint, perfect nesting and
    stride permutation, then ``SimplifyPass``."""
    label = NORMALIZE_STATEMENTS
    s: Stages = []
    # Out of phase: this matcher reads the frontend identity-construction map that lowering consumes.
    if lift:
        s += [(LIFT_SEMANTICS, LiftInv())]
    # Before lowering: NormalizeWCR after MapToForLoop computes azimint_naive wrong (an in-nsdfg WCR lowered into a
    # write-only connector), and the scatter privatizer must precede NormalizeWCR.
    if privatize_scatter_reductions:
        s += [(label, PrivatizeScatterReduction())]
    s += [(label, NormalizeWCR())]
    s += [
        (label, PatternApplyOnceEverywhere([MoveIfIntoMap()])),
        (label, ConvertLengthOneArraysToScalars()),
        (label, RevertNonReductionWCR()),
    ]
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True
    # While updates are still maps: once lowered and unrolled, an update's add and its copy-back can land in different
    # states (gromacs's ``fshift[ish, :] += fi[i, :]``), out of the fold's reach. Again in every statement round.
    s += [(label, EliminateCopyBack())]
    s += [
        (label, PatternApplyOnceEverywhere([lower_maps], state_local=True)),
        (label, PatternApplyOnceEverywhere([PruneConnectors()])),
        (label, InlineSDFGs()),
    ]
    s += _fold_scalar_slices(label)
    s += [(label, NormalizeNegativeStride()), (label, EliminateTrivialTasklets()), (label, StructuralCleanup())]
    s += [(label, RerollUnrolledLoops()), (label, UntileLoops()), (label, FuseConsecutiveLoops())]
    if unroll_limit > 0:
        s += [(label, ShortLoopUnroll(unroll_limit)), (label, unique_iterators()), (label, SymbolSSA())]
    s += [
        (
            label,
            PhaseFixpoint("statements", statement_round, NORMALIZE_ROUNDS, drivers=NORMALIZE_DRIVERS),
        )
    ]
    s += [(label, MaterializeLoopExitSymbols()), (label, CascadeInterstateEdgeAssignmentsUp())]
    # Again after the round: an unrolled body indexed through a strength-reduced counter shows its ``a[4*i + c]``
    # copies only once induction-variable substitution has closed the counter.
    s += [(label, RerollUnrolledLoops())]
    s += [
        (label, DistributeProducerConsumerLoop()),
        (label, MoveIfIntoLoop()),
        (label, CascadeInterstateEdgeAssignmentsUp()),
    ]
    if perfect_loop_nesting:
        s += [(label, PerfectLoopNesting(target=target))]
    s += [(label, unique_iterators()), (label, PatternApplyOnceEverywhere([TrivialLoopElimination()]))]
    s += [(label, LoopStridePermutation())]
    if normalize_loop_and_map_origin:
        s += [(label, NormalizeLoopAndMapOrigin())]
    # The special cleanup after normalization.
    s += [(label, SimplifyPass(pass_options=CANON_SIMPLIFY_OPTIONS))]
    return s


def lift_semantics(lift: bool, lift_copy: bool, interchange_carry_with_map: bool, target: str) -> Stages:
    """Recognize library operators on the normalized loop nests."""
    label = LIFT_SEMANTICS
    s: Stages = _fold_scalar_slices(label)
    # Before LoopToEinsum, which would take the inner product out of the nest.
    if lift:
        s += [(label, LoopToGetrfNoPivot(target=target)), (label, LoopToSymm()), (label, LoopToRankKUpdate())]
    s += [(label, LoopToSymmetrize())]
    if lift_copy:
        s += [(label, AssignmentAndCopyKernelToMemsetAndMemcpy())]
    if lift:
        s += [(label, LoopToTranspose())]
    s += [
        (label, LoopToEinsum()),
        (label, NestToContraction()),
        (label, RevertNonReductionWCR()),
        (label, SinkLastOnlyScanStore()),
        (label, LoopToReduce()),
        (label, LiftPreprocess()),
        (label, LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target)),
        (label, LoopToStateScan()),
        (label, ArgMaxLift()),
        (label, LoopToConditionalReduce()),
        (label, LoopToStreamCompaction()),
        (label, LiftLoopCarriedReduction()),
    ]
    return s


def recurrence_lifts(interchange_carry_with_map: bool, target: str, only: list | None = None) -> list[ppl.Pass]:
    """The lifts of a loop-carried recurrence, in :func:`lift_semantics` order; ``only`` restricts them to those loops."""
    return [
        LoopToReduce(only=only),
        LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target, only=only),
        LoopToStateScan(only=only),
        LoopToConditionalReduce(only=only),
        LiftLoopCarriedReduction(only=only),
    ]


def derive_parallelism(
    peel_limit: int,
    scatter_to_guarded_maps: bool,
    assume_parallel_guards: bool,
    reconstruct_wavefront_nest: bool,
    lift_copy: bool,
    interchange_carry_with_map: bool,
    target: str,
    validate: bool = False,
) -> Stages:
    """LoopToMap, rescue the loops it refused, LoopToMap again, guarded and reduction loops, wavefront skew.

    :param validate: Whether the pipeline validates its result; best-effort peeling then validates its own once.
    """
    label = DERIVE_PARALLELISM
    s: Stages = [(label, CascadeInterstateEdgeAssignmentsUp()), (label, unique_iterators()), (label, SymbolSSA())]
    # SSA leaves one definition per name, which is when propagation can substitute it (``j = j_0 + 1`` into
    # ``b[j]``); demoted first, its operand is a scalar read and the symbol stays.
    s += [(label, SymbolPropagation()), (label, ConstantPropagation())]
    # Before demotion: the anti-dependence proof reads a guard's symbols (``A[P]`` under ``P > i`` reads ahead), which
    # a demoted guard turns into opaque scalar reads.
    s += [(label, BreakAntiDependence(forward_reads=True)), (label, BreakAntiDependence())]
    s += [(label, DemoteNonStructuralSymbols()), (label, PrivatizeScalars())]
    s += [(label, DeadCarriedStoreElimination())]
    # Before the lift: a workspace whose pre-loop fill no iteration observes is then the loop's own, privatized per
    # iteration
    s += [(label, DeadLoopInitialization())]
    s += [(label, unit) for unit in loop_to_map_units(hold_sibling_chains=True)]
    # Rescue: each rewrite acts only on a loop LoopToMap refuses.
    segments: list = []
    if peel_limit > 0:
        s += [
            (label, LoopCarriedRotationSubstitution(peel_limit)),
            (label, BestEffortLoopPeeling(peel_limit, segments=segments, validate=validate)),
        ]
    s += [(label, unit) for unit in dead_dataflow_units()]
    s += [(label, PrivatizeScalars()), (label, PrivatizeArrays())]
    s += [(label, SymbolPropagation()), (label, ConstantPropagation())]
    # Peeling copies a loop with the symbols its body assigns; SSA names each copy's apart, or the lifted copy's
    # inner symbols shadow the other's and memlet propagation carries them out of the nested SDFG.
    s += [(label, CascadeInterstateEdgeAssignmentsUp()), (label, unique_iterators()), (label, SymbolSSA())]
    s += [(label, unit) for unit in loop_to_map_units(hold_sibling_chains=True)]
    # A split or peel can leave a plain recurrence the lifting segment never saw (``y[i] = s[i] + y[i - 1]`` on the
    # ``[K:N)`` half of an ``if i < K`` body); lift it, on the split's segments only, before the levels below
    # settle it as sequential.
    s += [(LIFT_SEMANTICS, unit) for unit in recurrence_lifts(interchange_carry_with_map, target, only=segments)]
    # Out of phase: the fill map LoopToScan's stride specialization emits is lifted only before a cleanup fuses a
    # second writer into its state (tsvc_2_5 scan_strided_sym).
    if lift_copy:
        s += [(LIFT_SEMANTICS, AssignmentAndCopyKernelToMemsetAndMemcpy())]
    if target == "gpu":
        s += [(label, PatternApplyOnceEverywhere([PerfLoopNesting()]))]
    s += [(label, ParallelizeUnderConstraint(assume_constraint=assume_parallel_guards))]
    # Accumulators become reduction loops, then derive again.
    s += [
        (label, FuseChainedScalarReductions()),
        (label, RevertNonReductionWCR()),
        (label, PinCarriedTopLevelLoops()),
        (label, AccumulatorCopyChainToWCR()),
        (label, RetargetWCRAccumulator()),
        (label, PropagateMemlets()),
        (label, SetSymbolNonnegativeAssumptions()),
        (label, ParallelizeLoopsHoldingSiblingChains(propagate=False)),
        (label, PrivatizeScalars()),
        (label, PrivatizeArrays()),
    ]
    # Inlined bodies show their in-place accumulations; as WCR they let a loop refused on them map, ahead of the
    # guarded-scatter versioning below. Either rewrite, the inlining or the WCR, can leave a loop that now maps.
    s += [
        (
            label,
            _ThenIfChanged(
                [
                    *[(unit, isinstance(unit, PruneAndInlineNestedSDFGs)) for _, unit in _inline_single_state(label)],
                    (StructuralCleanup(), False),
                    (NormalizeWCR(), False),
                    (CarriedAccumulationToWCR(), True),
                ],
                [
                    PropagateMemlets(),
                    ParallelizeLoopsHoldingSiblingChains(propagate=False),
                    PrivatizeScalars(),
                    PrivatizeArrays(),
                ],
            ),
        )
    ]
    if scatter_to_guarded_maps:
        # An ``arange`` fill is an inlined map by now: its interstate reads become their index, so a scatter keyed
        # through it shows the loop variable as its slot and needs no conflict check.
        s += [(label, ForwardIotaReads()), (label, SymbolPropagation())]
        s += [
            (
                label,
                ScatterToGuardedMaps(emit_unparallelized_else_branch=True, assume_no_conflicts=assume_parallel_guards),
            )
        ]
    # Wavefront: skew the nests every level left sequential, then map the exposed axis.
    if reconstruct_wavefront_nest:
        s += [(label, RevertNonReductionWCR()), *_inline_single_state(label), (label, ReconstructWavefrontNest())]
    if target == "gpu":
        s += [(label, ppl.Pipeline([ReorderStateForLoopFusion()]))]
    s += [(label, FuseLoops()), (label, WavefrontSkew(target=target))]
    s += [(label, unit) for unit in loop_to_map_units()]
    return s


def maximal_fusion(target: str) -> PhaseFixpoint:
    """Maximal fusion: (FuseStates x FuseLoops x MapCollapse x FuseMaps, single-state bodies inlined first) to a fixpoint
    inside a fixpoint that re-permutes, interchanges and fuses conditions. The recompose phase runs it, and so does
    specialization after hoisting guards (``hoist_invariant_guards``): one definition of "fuse everything"."""

    def inner_round() -> list[ppl.Pass]:
        return [
            # States first: map fusion only sees maps that share a state
            HoistAssignmentsToLoopBodyHead(),
            FuseStates(aggressive_merges=True, order_array_seeds=True),
            # Loops next: two fused loops put their maps in one body
            FuseLoops(),
            PatternApplyOnceEverywhere([MapCollapse()]),
            PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]),
            ppl.Pipeline(
                [
                    FuseMaps(allow_intermediates_with_symbolic_size=True, partial_reduce_fusion=True),
                    # A producer over more axes than its consumer fuses in as an inner map over the rest
                    FuseMapsAtSharedAxes(),
                ]
            ),
            PatternApplyOnceEverywhere([MapReduceFusion(), MapWCRFusion()]),
            *(unit for stage_label, unit in _inline_single_state(RECOMPOSE)),
            StructuralCleanup(),
        ]

    def outer_round() -> list[ppl.Pass]:
        return [
            ReverseMapTraversal(),
            MinimizeStridePermutation(),
            MoveLoopIntoMapGated(target=target),
            *(unit for stage_label, unit in _inline_single_state(RECOMPOSE)),
            PhaseFixpoint("fusion", inner_round, FUSION_INNER_ROUNDS),
            MergeGuards(),
            # An if/else of two same-range maps becomes one selecting map, its guard trivial for the lift below
            SelectBranchMaps(),
            LiftTrivialIf(),
            NormalizeMapBody(),
            *(unit for stage_label, unit in _inline_single_state(RECOMPOSE)),
        ]

    return PhaseFixpoint("maximal_fusion", outer_round, FUSION_OUTER_ROUNDS)


def recompose(lift: bool, lift_copy: bool, target: str) -> Stages:
    """(MapCollapse x FuseMaps x FuseStates) to a fixpoint inside a fixpoint that re-permutes, interchanges and
    fuses conditions; lift what fusion exposed; the final tidy. Invariant guards stay pushed in: hoisting them is
    the specialization step ``hoist_invariant_guards``."""
    label = RECOMPOSE
    s: Stages = [(label, NormalizeWCRSource()), (label, RevertNonReductionWCR())]
    # Before fusion: the WCR map it leaves fuses with the matrix's producer
    s += [(label, FuseElementwiseIntoGemv())]
    s += [
        (label, CascadeInterstateEdgeAssignmentsUp()),
        # A per-iteration assignment between two body states keeps them, and the maps in them, apart
        (label, HoistAssignmentsToLoopBodyHead()),
        (label, EmptyStateElimination()),
        (label, PatternApplyOnceEverywhere([TrivialMapElimination()])),
        (label, EmptyLoopElimination()),
        (label, PatternApplyOnceEverywhere([MoveIfIntoMap()])),
    ]
    # Index remaps (np.roll, broadcast_to, np.where) as gather maps, so fusion inlines them instead of keeping a full-size
    # transient per call (laplacian_stencil_3d's 24 rolls); ingest expands them already, this catches the ones the
    # lifting phases made since. The inlining below dissolves their nested SDFGs.
    s += [(label, ExpandIndexRemaps())]
    s += [*_inline_single_state(label), (label, StructuralCleanup())]
    # A memset the control flow after it overwrites unread (velocity's zeta_value, swept level by level) goes before
    # fusion, so its state does not stand between the maps; the finish runs it again for the fills lifted after.
    s += [(label, DeadFillElimination())]

    s += [(FUSE, maximal_fusion(target))]
    label = AFTER_FUSION
    # Out of phase: these matchers need the fused map form (contractions split over maps, durbin's unary copy).
    if lift_copy:
        s += [(label, AssignmentAndCopyKernelToMemsetAndMemcpy())]
        s += _inline_single_state(label)
    if lift:
        s += [(label, PatternApplyOnceEverywhere([LiftEinsum()]))]
    s += [(label, NormalizeWCRSource()), (label, RelaxIntegerPowers())]
    s += [(label, ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination(fold_ordered_copies=True)]))]
    s += [*_inline_single_state(label), (label, PruneEmptyConditionalBranches())]
    s += [
        (label, PatternApplyOnceEverywhere([RedundantArray()])),
        (label, RematerializeDerivedTemporaries()),
    ]
    # Fusion frees loops the pre-fusion sweep had to keep: a level loop whose body was a fill, a map and a copy of one
    # scratch plane is a single map after it (velocity_tendencies' nine level loops), so sweep once more and collapse.
    s += [(label, ParallelizeLoopsHoldingSiblingChains()), (label, PatternApplyOnceEverywhere([MapCollapse()]))]
    return s


def finish() -> Stages:
    """The terminal bookkeeping: symbols, derived annotations, assumption guard, hints; the views inlining minted."""
    s: Stages = rail(FINISH)
    # On the settled form: a loop over the segments of a pointer array is lifted behind the entry guard that checks
    # the data facts its segment confinement rests on (nfa_frontier's components).
    s += [(FINISH, SegmentConfinedLoops())]
    for _ in range(TERMINAL_SYMBOL_ROUNDS):
        s += [(FINISH, SymbolDedup()), (FINISH, SymbolPropagation()), (FINISH, ConstantPropagation())]
    s += [
        (FINISH, RemoveUnusedSymbols()),
        (FINISH, OptionalArrayInference()),
        (FINISH, ConvertLengthOneArraysToScalars()),
        (FINISH, RevertNonReductionWCR()),
        (FINISH, DeadFillElimination()),
        (FINISH, PruneUnreferencedTransients()),
        (FINISH, InlineControlFlowRegions()),
        (FINISH, AssumeSymbolConstraints()),
        (FINISH, NormalizeFloorDivision()),
        # A staging copy sits in its readers' map scope, fed from the map entry, only once the terminal inlining has
        # run; the reclaim stays last for the arrays the fold leaves.
        (FINISH, PatternApplyOnceEverywhere([RedundantMapReadCopy()])),
        (FINISH, ppl.Pipeline([ArrayElimination(fold_ordered_copies=True)])),
    ]
    return s


def build_stages(
    unroll_limit: int = DEFAULT_UNROLL_LIMIT,
    peel_limit: int = DEFAULT_PEEL_LIMIT,
    interchange_carry_with_map: bool = True,
    scatter_to_guarded_maps: bool = True,
    privatize_scatter_reductions: bool = True,
    reconstruct_wavefront_nest: bool = True,
    normalize_loop_and_map_origin: bool = False,
    assume_parallel_guards: bool = False,
    perfect_loop_nesting: bool = True,
    target: str = "cpu",
    lift: bool = True,
    lift_copy: bool = True,
    semantic_lifting: bool = True,
    validate: bool = False,
) -> Stages:
    """Build the canonicalization recipe: the paper's phases in order (see the section comment above).

    :param validate: The pipeline's ``validate``, for the units that validate their own rewrite only when asked.

    :returns: ``(label, pass)`` pairs in run order, fresh instances each call.
    """
    lift, lift_copy = semantic_lifting and lift, semantic_lifting and lift_copy
    s = ingest(target)
    s += normalize_statements(
        lift, privatize_scatter_reductions, unroll_limit, perfect_loop_nesting, target, normalize_loop_and_map_origin
    )
    s += rail()
    s += lift_semantics(lift, lift_copy, interchange_carry_with_map, target)
    s += rail()
    s += derive_parallelism(
        peel_limit,
        scatter_to_guarded_maps,
        assume_parallel_guards,
        reconstruct_wavefront_nest,
        lift_copy,
        interchange_carry_with_map,
        target,
        validate,
    )
    s += rail()
    s += recompose(lift, lift_copy, target)
    # Once more after the lifts recompose ends with: what they exposed fuses too
    s += [(FUSE, maximal_fusion(target))]
    s += finish()
    prepared([unit for stage_label, unit in s])
    return s


def disable_unit_validation(unit: ppl.Pass) -> None:
    """Turn off a stage unit's own validation, nested pipeline members included.

    This pipeline's ``validate`` / ``validate_all`` own validation: a unit validating the whole SDFG itself
    repeats that walk, and validation is not read-only (``Fill.validate`` drops an unconnected input).
    """
    if isinstance(unit, (PatternMatchAndApply, FuseMaps)):
        unit.validate = False
        unit.validate_all = False
    if isinstance(unit, (NormalizeWCR, NormalizeWCRSource)):
        unit.validate = False
    if isinstance(unit, ppl.Pipeline):
        for member in unit.passes:
            disable_unit_validation(member)


#: A stage factory returns that stage's fresh passes, in order.
StageFactory = Callable[[], list[ppl.Pass]]


def _stage_runs() -> list[tuple[str, int, int]]:
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
    runs: list[list] = []
    for i, (lbl, unit) in enumerate(build_stages()):
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
    return lambda: [p for lbl, p in build_stages()[start:stop]]


#: Grouped view of :func:`build_stages`: ``(label,
#: factory)`` per stage, in order, where ``factory()`` builds that stage's
#: fresh passes. ``build_stages`` (flat ``(label, pass)``) is the source of
#: truth used by the pipeline; this view exists for callers that iterate
#: stage-by-stage (``for name, factory in CANONICALIZE_STAGES:
#: for unit in factory(): ...``).
CANONICALIZE_STAGES: list[tuple[str, StageFactory]] = [
    (label, _stage_factory(start, stop)) for label, start, stop in _stage_runs()
]


def _assert_self_contained(unit: ppl.Pass) -> None:
    """Guard the empty-``pipeline_results`` invariant.

    Every unit is applied with an empty results dict, so it must either have
    no dependencies or be a self-resolving ``Pipeline`` (e.g. ``SimplifyPass``,
    or the ``Pipeline`` wrapping ``FuseMaps``). A bare dependency-bearing
    pass placed directly in a stage would silently lose its inputs.

    :param unit: The pass about to be applied.
    :raises AssertionError: If ``unit`` has unresolved dependencies.
    """
    assert not unit.depends_on() or isinstance(unit, ppl.Pipeline), (
        f"{type(unit).__name__} has dependencies but is not a self-resolving "
        f"Pipeline; wrap it so its depends_on() is satisfied."
    )


def changed_the_graph(unit: ppl.Pass, result: Any) -> bool:
    """Whether ``result`` reports a rewrite: plain ``Pipeline`` returns its whole dict even when
    unchanged (unlike ``FixedPointPipeline``), which would pin the caller's dirty flag forever."""
    if result is None:
        return False
    if type(unit) is ppl.Pipeline:
        return any(name in unit._pass_names for name in result)
    return True


#: Units a rerun leaves unchanged on a graph no unit changed since their last run; a run then skips them.
SETTLED_UNITS: tuple[type[ppl.Pass], ...] = (
    StructuralCleanup,
    RouteOrderingThroughData,
    PruneAndInlineNestedSDFGs,
    PropagateAndPrune,
    SymbolDedup,
)


class Settled:
    """The change clock of one run of units: a unit of :data:`SETTLED_UNITS` is skipped when no unit has
    changed the graph since it last ran. Keyed by class: these units take no options."""

    __slots__ = ("clock", "ran_at")

    def __init__(self) -> None:
        self.clock = 0
        self.ran_at: dict[type, int] = {}

    def skip(self, unit: ppl.Pass) -> bool:
        return type(unit) in SETTLED_UNITS and self.ran_at.get(type(unit)) == self.clock

    def record(self, unit: ppl.Pass, changed: bool, dirties: bool = True) -> None:
        """Note that ``unit`` ran; ``dirties`` is whether its change earns the settled units another run."""
        if changed and dirties:
            self.clock += 1
        if type(unit) in SETTLED_UNITS:
            self.ran_at[type(unit)] = self.clock


@properties.make_properties
@transformation.explicit_cf_compatible
class CanonicalizationPipeline(ppl.Pass):
    """Rewrite an SDFG into its canonical form.

    The recipe (:func:`build_stages`) is one flat ordered list of passes
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
    :param peel_limit: Best-effort loop peeling before parallelize (0 disables); defaults to
                       ``optimizer.canonicalization.peel_limit``.
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

    CATEGORY: str = "Canonicalization"

    validate = properties.Property(
        dtype=bool,
        default=False,
        desc="Validate the SDFG once at the end. OFF by default, with validate_all: validation is a "
        "whole-SDFG walk and it is a measurable share of the pipeline on a large graph. Turn validate on "
        "to check the result, and validate_all as well to find WHICH stage broke it.",
    )
    validate_all = properties.Property(
        dtype=bool,
        default=False,
        desc="Validate the SDFG after EVERY stage that reported a modification, which pinpoints the "
        "stage that produced an invalid SDFG instead of only catching it at the end. OFF by default: "
        "it is one whole-SDFG walk per modifying stage, and the recipe has 235 of them, so on a large "
        "graph it costs more than the passes it watches. Turn it on to bisect a stage that produced "
        "an invalid SDFG; the end-of-pipeline ``validate`` still catches the break either way.",
    )
    dump_dir = properties.Property(
        dtype=str,
        default=None,
        allow_none=True,
        desc="Directory to save the SDFG into after every stage, for bisecting which stage broke it. "
        "None (default) disables dumping entirely.",
    )
    unroll_limit = properties.Property(
        dtype=int, default=DEFAULT_UNROLL_LIMIT, desc="Unroll constant-trip loops <= this many iterations (0 disables)."
    )
    peel_limit = properties.Property(
        dtype=int, default=DEFAULT_PEEL_LIMIT, desc="Best-effort loop peeling before parallelize (0 disables)."
    )
    target = properties.Property(
        dtype=str, default="cpu", choices=["cpu", "gpu"], desc="Per-target knob preset selector ('cpu' or 'gpu')."
    )
    interchange_carry_with_map = properties.Property(
        dtype=bool,
        default=True,
        desc="LoopToScan: relocate the carry LoopRegion INTO the per-column Map (on for CPU, off for GPU).",
    )
    scatter_to_guarded_maps = properties.Property(
        dtype=bool,
        default=True,
        desc="Run ScatterToGuardedMaps in the scatter stage to lift scatter loops with a sort-based guard.",
    )
    privatize_scatter_reductions = properties.Property(
        dtype=bool,
        default=True,
        desc="Run PrivatizeScatterReduction to surface a data-dependent scatter reduction "
        "(azimint histogram) to an OpenMP array-section reduction clause (CPU-only; off for GPU).",
    )
    reconstruct_wavefront_nest = properties.Property(
        dtype=bool,
        default=True,
        desc="Run ReconstructWavefrontNest right before WavefrontSkew in the loop_fuse stage, rebuilding "
        "an imperfect Map-plus-LoopRegion stencil body into the single loop WavefrontSkew requires. "
        "On by default on both targets (see CPU_DEFAULTS).",
    )
    normalize_loop_and_map_origin = properties.Property(
        dtype=bool,
        default=False,
        desc="Run NormalizeLoopAndMapOrigin right before the loop_to_x stage, rebasing every Map range / "
        "LoopRegion counter to a 0-based begin while keeping the stride. Off by default on both targets "
        "(see CPU_DEFAULTS).",
    )
    perfect_loop_nesting = properties.Property(
        dtype=bool,
        default=True,
        desc="Distribute a loop over its data-independent statement groups (PerfectLoopNesting) so "
        "each statement parallelizes on its own axes. On by default since the pass stopped grouping "
        "through LoopFission -- see the fission-stage comment for what that changed and for the "
        "CloudSC gate that has not yet been re-run.",
    )

    assume_parallel_guards = properties.Property(
        dtype=bool,
        default=False,
        desc="Assume every parallel-guard condition holds: ParallelizeUnderConstraint + "
        "ScatterToGuardedMaps emit only the parallel Map (no if-else fallback, no sort/trap). "
        "Unsound if a condition is violated at runtime; default keeps the sound guards.",
    )
    lift = properties.Property(
        dtype=bool,
        default=True,
        desc="Lift tensor-contraction (matmul) maps to Einsum library nodes (False keeps them as WCR loop nests).",
    )
    lift_copy = properties.Property(
        dtype=bool,
        default=True,
        desc="Lift contiguous copy/zero-init maps to Copy/Fill library nodes (False keeps them as maps).",
    )
    semantic_lifting = properties.Property(
        dtype=bool,
        default=True,
        desc="Master gate for the post-LoopToMap map->library-node lifts (Einsum + Copy/Fill). "
        "False (set by the vectorizer) keeps the residual as raw maps it can lower.",
    )
    stages = properties.ListProperty(
        element_type=str,
        allow_none=True,
        default=None,
        desc="Stage labels to run, in recipe order; None (default) runs every stage. A label "
        "absent from the recipe raises ValueError.",
    )

    def __init__(
        self,
        validate: bool = False,
        validate_all: bool = False,
        unroll_limit: int = DEFAULT_UNROLL_LIMIT,
        peel_limit: int | None = None,
        target: str = "cpu",
        interchange_carry_with_map: bool | None = None,
        scatter_to_guarded_maps: bool | None = None,
        privatize_scatter_reductions: bool | None = None,
        reconstruct_wavefront_nest: bool | None = None,
        normalize_loop_and_map_origin: bool | None = None,
        assume_parallel_guards: bool = False,
        perfect_loop_nesting: bool = True,
        specialize_constants: dict[str, int] | None = None,
        lift: bool = True,
        lift_copy: bool = True,
        semantic_lifting: bool = True,
        dump_dir: str | None = None,
        stages: Sequence[str] | None = None,
    ) -> None:
        if target not in TARGET_DEFAULTS:
            raise ValueError(f"target must be one of {sorted(TARGET_DEFAULTS)}; got {target!r}")
        self.validate = validate
        self.validate_all = validate_all
        self.dump_dir = dump_dir
        self.unroll_limit = unroll_limit
        self.target = target
        # Per-target knobs: ``None`` -> preset; explicit value overrides preset.
        self.peel_limit = DEFAULT_PEEL_LIMIT if peel_limit is None else peel_limit
        self.interchange_carry_with_map = _resolve_target_default(
            target, "interchange_carry_with_map", interchange_carry_with_map, fallback=True
        )
        self.scatter_to_guarded_maps = _resolve_target_default(
            target, "scatter_to_guarded_maps", scatter_to_guarded_maps, fallback=True
        )
        self.privatize_scatter_reductions = _resolve_target_default(
            target, "privatize_scatter_reductions", privatize_scatter_reductions, fallback=(target == "cpu")
        )
        self.reconstruct_wavefront_nest = _resolve_target_default(
            target, "reconstruct_wavefront_nest", reconstruct_wavefront_nest, fallback=True
        )
        self.normalize_loop_and_map_origin = _resolve_target_default(
            target, "normalize_loop_and_map_origin", normalize_loop_and_map_origin, fallback=False
        )
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

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def build_stages(self) -> list[tuple[str, ppl.Pass]]:
        """Build this pipeline's flat recipe, honoring every knob property.

        :returns: ``(stage_label, pass)`` pairs, in recipe order, fresh instances each call.
        """
        return build_stages(
            unroll_limit=self.unroll_limit,
            peel_limit=self.peel_limit,
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
            semantic_lifting=self.semantic_lifting,
            validate=self.validate,
        )

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
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
                for name, value in self._specialize_constants.items()
                if isinstance(sdfg.arrays.get(name), data.Scalar)
            }
            specialize_scalars(sdfg, scalars)
            specialize_symbols(sdfg, {n: v for n, v in self._specialize_constants.items() if n not in scalars})
        stages = self.build_stages()
        if self.stages is not None:
            known_labels = OrderedSet(label for label, _ in stages)
            unknown = [label for label in self.stages if label not in known_labels]
            if unknown:
                raise ValueError(f"unknown canonicalization stage label(s): {unknown}")
            wanted = OrderedSet(self.stages)
            stages = [(label, unit) for label, unit in stages if label in wanted]
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            sdfg.save(os.path.join(self.dump_dir, "000_input.sdfgz"), compress=True)
        # The cleanup block and the inline step are spliced in at every stage boundary, and on a settled
        # graph they report no change: skip one unless a unit touched the graph since it last ran. Only
        # non-cleanup units re-dirty it, so a cleanup that tidies something does not earn the next one.
        settled = Settled()
        recorder = canon_stats.active()
        for index, (label, unit) in enumerate(stages, start=1):
            is_cleanup = isinstance(unit, StructuralCleanup)
            if settled.skip(unit):
                continue
            _assert_self_contained(unit)
            # A Pipeline validates its own members when asked, so scope validation to the
            # sub-pass that actually changed something instead of re-walking the whole SDFG here.
            is_pipeline = isinstance(unit, ppl.Pipeline)
            if self.validate_all and isinstance(unit, ppl.Pipeline):
                unit.validate_subpasses = True
            if recorder is not None:
                recorder.stage_start(label, unit)
            result = unit.apply_pass(sdfg, {})
            if recorder is not None:
                recorder.stage_done(sdfg)
            settled.record(unit, changed_the_graph(unit, result), dirties=not is_cleanup)
            # apply_pass returns non-None iff it modified the SDFG, so an unchanged SDFG needs no
            # re-validation -- that is what makes validate_all affordable by default.
            if self.validate_all and result is not None and not is_pipeline:
                sdfg.validate()
            # Dump AFTER every stage, including no-ops: a bisect needs a dense index, and a stage
            # that reports no change can still have rewritten the graph (5 units do exactly that).
            if self.dump_dir:
                sdfg.save(os.path.join(self.dump_dir, f"{index:03d}_{label}.sdfgz"), compress=True)
        disable_openmp_sections(sdfg)
        equalize(sdfg)
        if self.validate:
            sdfg.validate()
        return len(stages)


def stage_labels(target: str = "cpu") -> list[str]:
    """Ordered, de-duplicated stage labels of the canonicalization recipe for ``target``.

    Lets a caller slice :func:`canonicalize` -- run every stage up to (not including) a chosen
    label, do its own thing, then run the rest via ``CanonicalizationPipeline(stages=...)``.

    :param target: ``'cpu'`` (default) or ``'gpu'``.
    :returns: Stage labels in first-occurrence recipe order.
    """
    seen = OrderedSet(label for label, _ in CanonicalizationPipeline(target=target).build_stages())
    return list(seen)


def canonicalize(
    sdfg: SDFG,
    validate: bool = False,
    validate_all: bool = False,
    unroll_limit: int = DEFAULT_UNROLL_LIMIT,
    peel_limit: int | None = None,
    target: str = "cpu",
    interchange_carry_with_map: bool | None = None,
    scatter_to_guarded_maps: bool | None = None,
    privatize_scatter_reductions: bool | None = None,
    reconstruct_wavefront_nest: bool | None = None,
    normalize_loop_and_map_origin: bool | None = None,
    assume_parallel_guards: bool = False,
    perfect_loop_nesting: bool = True,
    specialize_constants: dict[str, int] | None = None,
    lift: bool = True,
    lift_copy: bool = True,
    semantic_lifting: bool = True,
    dump_dir: str | None = None,
    stages: Sequence[str] | None = None,
    nonnegative_symbols: Iterable[str] | None = None,
    stats: dict | str | None = None,
) -> SDFG:
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
                       (default) -> ``optimizer.canonicalization.peel_limit``.
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
                                   ``None`` (default) -> per-target preset (on for both).
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
    :param nonnegative_symbols: The symbols (and scalar arguments promoted to symbols) the caller knows are never
                                negative. Registered in the SDFG's facts before the first pass, so every proof reads
                                them, and checked at program entry. Any other symbol has an unknown sign. ``None``
                                (default) keeps what an earlier call declared, else none.
    :param stats: Record which stage settled each loop and the time spent, in total and inside Z3 and isl
                  (:mod:`~dace.transformation.passes.analysis.canon_stats`): a dict is filled with the record, a
                  path gets it appended as one JSON line. ``None`` (default) records nothing.
    :returns: The same ``sdfg`` instance, canonicalized.
    """
    arguments = (
        validate,
        validate_all,
        unroll_limit,
        peel_limit,
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
        stages,
    )
    if stats is None:
        return canonicalize_recorded(sdfg, nonnegative_symbols, *arguments)
    recorder = canon_stats.CanonStats(sdfg.name)
    with recorder.recording(sdfg):
        canonicalize_recorded(sdfg, nonnegative_symbols, *arguments)
    recorder.finish(sdfg)
    if isinstance(stats, dict):
        stats.update(recorder.as_dict())
    else:
        recorder.write(stats)
    return sdfg


def canonicalize_recorded(sdfg: SDFG, nonnegative_symbols: Iterable[str] | None, *arguments) -> SDFG:
    """:func:`canonicalize` after its ``stats`` argument."""
    if nonnegative_symbols is not None:
        declare_nonnegative_symbols(sdfg, nonnegative_symbols)
    set_symbol_nonnegative_assumptions(sdfg)
    # Every stage below recovers loop bounds from STRING-backed properties, which means re-parsing
    # them -- and an unscoped parse mints ``DEFAULT_SYMBOL_TYPE``. Declaring the SDFG's own symbol
    # table as the authority for the whole run is what keeps one name spelled as ONE symbol, so a
    # bound and a descriptor shape still cancel (see ``sympy_to_dace``). Nested SDFGs are covered
    # by name: a nested scope re-declares the same names, and only names this table does not carry
    # keep the default.
    authority = {name: dtype for nested in sdfg.all_sdfgs_recursive() for name, dtype in nested.symbols.items()}
    with symbolic.serialization_symbol_dtypes(authority):
        return canonicalize_under_authority(sdfg, *arguments)


def canonicalize_under_authority(
    sdfg: SDFG,
    validate: bool,
    validate_all: bool,
    unroll_limit: int,
    peel_limit: int | None,
    target: str,
    interchange_carry_with_map: bool | None,
    scatter_to_guarded_maps: bool | None,
    privatize_scatter_reductions: bool | None,
    reconstruct_wavefront_nest: bool | None,
    normalize_loop_and_map_origin: bool | None,
    assume_parallel_guards: bool,
    perfect_loop_nesting: bool,
    specialize_constants: dict[str, int] | None,
    lift: bool,
    lift_copy: bool,
    semantic_lifting: bool,
    dump_dir: str | None,
    stages: Sequence[str] | None = None,
) -> SDFG:
    """The body of :func:`canonicalize`, run with the SDFG's symbol dtypes already in scope."""
    CanonicalizationPipeline(
        validate=validate,
        validate_all=validate_all,
        unroll_limit=unroll_limit,
        peel_limit=peel_limit,
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
        stages=stages,
    ).apply_pass(sdfg, {})
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
