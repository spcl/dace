# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""SDFG canonicalization pipeline."""

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
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
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (
    AssumeSymbolConstraints,
    SetSymbolNonnegativeAssumptions,
    declare_nonnegative_symbols,
    set_symbol_nonnegative_assumptions,
)
from dace.transformation.passes.canonicalize.batch_gathered_products import BatchGatheredProducts
from dace.transformation.passes.canonicalize.carried_accumulation import CarriedAccumulationToWCR
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import (
    CascadeInterstateEdgeAssignmentsUp,
    HoistAssignmentsToLoopBodyHead,
)
from dace.transformation.passes.canonicalize.collapse_noop_cast import CollapseNoOpCast
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import EliminateTrivialTasklets
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.expand_fill_copy import ExpandFillCopy
from dace.transformation.passes.canonicalize.expand_index_remaps import ExpandIndexRemaps
from dace.transformation.passes.canonicalize.fill_through_copies import FillThroughCopies
from dace.transformation.passes.canonicalize.fold_scalar_read_copies import FoldScalarReadCopies
from dace.transformation.passes.canonicalize.forward_copy_maps import ForwardCopyMaps
from dace.transformation.passes.canonicalize.forward_offset_copies import ForwardOffsetCopies
from dace.transformation.passes.canonicalize.forward_store_to_load import ForwardStoreToLoad
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops
from dace.transformation.passes.canonicalize.fuse_elementwise_into_gemv import FuseElementwiseIntoGemv
from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops
from dace.transformation.passes.canonicalize.fuse_maps_at_shared_axes import FuseMapsAtSharedAxes
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.in_bounds_facts import DeclareInBoundsExtents, declare_in_bounds_extents
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
from dace.transformation.passes.consolidate_edges import ConsolidateEdges
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.dead_fill_elimination import DeadFillElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.demote_non_structural_symbols import DemoteNonStructuralSymbols
from dace.transformation.passes.detach_disjoint_reads import DetachDisjointReads
from dace.transformation.passes.empty_loop_elimination import EmptyLoopElimination
from dace.transformation.passes.equalize_symbol_dtypes import EqualizeSymbolDtypes, equalize
from dace.transformation.passes.fold_constant_tables import FoldConstantTables, read_constant_tables
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.fusion_inline import FuseStates, InlineControlFlowRegions
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.loop_to_reduce import (
    AccumulatorCopyChainToWCR,
    LoopToReduce,
)
from dace.transformation.passes.loop_to_scan import LoopToScan
from dace.transformation.passes.loop_to_state_scan import LoopToStateScan
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.parallelization.contracts.guarded_scatter import ScatterToGuardedMaps
from dace.transformation.passes.parallelization.contracts.under_constraint import ParallelizeUnderConstraint
from dace.transformation.passes.parallelization.parallelize import Parallelize
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops
from dace.transformation.passes.parallelization_prep import (
    DEFAULT_PEEL_LIMIT,
    BestEffortLoopPeeling,
    ModuloBandSplit,
    ShortLoopUnroll,
)
from dace.transformation.passes.pattern_matching import (
    PatternApplyOnceEverywhere,
    PatternMatchAndApply,
    PatternMatchAndApplyRepeated,
)
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.rematerialize_derived_temporaries import RematerializeDerivedTemporaries
from dace.transformation.passes.remove_views import RemoveViews
from dace.transformation.passes.scalar_fission import PrivatizeScalars, PrivatizeScalarsAndArrays
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.single_element_library_nodes import SingleElementLibraryNodes
from dace.transformation.passes.symbol_propagation import SymbolPropagation
from dace.transformation.passes.symbol_ssa import SymbolSSA
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.vectorization.propagate_index_subsets import PropagateIndexSubsets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod

#: Canonicalization's simplify fuses states with StateFusionExtended's extra merges.
CANON_SIMPLIFY_OPTIONS: dict[str, Any] = {"FuseStates.aggressive_merges": True}


def disable_openmp_sections(sdfg: SDFG) -> None:
    """Opt ``sdfg`` and every nested SDFG out of ``#pragma omp parallel sections``."""
    for nested in sdfg.all_sdfgs_recursive():
        nested.openmp_sections = False


def _structural_cleanup(label: str) -> list[tuple[str, ppl.Pass]]:
    """Tidy symbols, then the state machine, between phases; never ``SimplifyPass`` mid-pipeline."""
    return [(label, StructuralCleanup())]


@properties.make_properties
@transformation.explicit_cf_compatible
class PropagateSymbols(ppl.Pass):
    """``SymbolPropagation`` then ``ConstantPropagation``, the order the recipe always wants them in, as one unit."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return SymbolPropagation().modifies() | ConstantPropagation().modifies()

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified != ppl.Modifies.Nothing

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> bool | None:
        tables = read_constant_tables(sdfg)
        symbols = SymbolPropagation().apply_pass(sdfg, {})
        constants = ConstantPropagation().apply_pass(sdfg, {})
        return True if tables or symbols is not None or constants is not None else None


def symbol_phase() -> list[ppl.Pass]:
    """Merge the symbols that are one value, fold what is known, drop what that left unread."""
    return [SymbolDedup(), PropagateSymbols(), RemoveUnusedSymbols()]


@properties.make_properties
@transformation.explicit_cf_compatible
class PropagateAndPrune(ppl.Pass):
    """Fold symbols and constants, then drop the dataflow that folding made dead -- TWICE."""

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
        for _ in range(self.ROUNDS):
            # A Pipeline per round, not bare passes: DeadDataflowElimination declares a
            # ControlFlowBlockReachability dependency that only a Pipeline resolves.
            pipeline = ppl.Pipeline([PropagateSymbols(), DeadDataflowElimination()])
            # A round that modified nothing hands every later round the identical graph.
            if not changed_the_graph(pipeline, pipeline.apply_pass(sdfg, {})):
                break
            changed += 1
        return changed or None


@properties.make_properties
@transformation.explicit_cf_compatible
class StructuralCleanup(ppl.Pass):
    """The between-phase structural cleanup, as ONE unit."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return True

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def units(self) -> list[ppl.Pass]:
        """The block's members, in order."""
        walk_fuse = FuseStates(aggressive_merges=True)
        walk_fuse.progress = False
        return [
            EqualizeSymbolDtypes(),
            *symbol_phase(),
            SymbolDedup(),
            FoldScalarReadCopies(),
            walk_fuse,
            EmptyStateElimination(),
            DeadStateElimination(),
            RedundantOrderingEdgeElimination(),
            # A View a later pass made (inlining a call, nesting a loop body) reads its array directly again
            RemoveViews(),
            # A library node over one element per operand is the tasklet it computes
            SingleElementLibraryNodes(),
            # After the state fusion above, so a copy and its readers share a state: redundant copies fold away
            ppl.Pipeline([ArrayElimination(fold_ordered_copies=True, nested=True)]),
            # A shifted copy array elimination keeps (read in several states) is read at the shifted source index
            ForwardOffsetCopies(),
            # A transient a map copies element by element (nbody's x.T) is read from the array it copies
            ForwardCopyMaps(),
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
    """Apply the between-phase structural cleanup once, to a finished SDFG."""
    for label, unit in _structural_cleanup("structural_cleanup"):
        unit.apply_pass(sdfg, {})


def _inline_single_state(label: str) -> list[tuple[str, ppl.Pass]]:
    """Flatten single-state NestedSDFG bodies; un-inlined, they report whole-array memlets and every dependence test
    refuses on the box."""
    return [(label, RouteOrderingThroughData()), (label, PruneAndInlineNestedSDFGs())]


#: Cap on the IV-substitution / statement-fission alternation.
IV_SPLIT_MAX_ROUNDS = 2


@properties.make_properties
@transformation.explicit_cf_compatible
class IvSubstitutionFissionFixpoint(ppl.Pass):
    """Alternate induction-variable substitution with statement fission until neither fires."""

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
        """The passes one round runs, in order."""
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
        """Run the round up to ``max_rounds`` times, stopping early on a no-op round."""
        units = self.round_units()
        rounds = 0
        for _ in range(self.max_rounds):
            # Each pass reports differently
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
        """``simplify.apply_pass(sdfg, {})`` without the tail of its confirming sweep."""
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
    """The ``first`` units in order, then the ``then`` units only when a watched one of them rewrote the graph: the lift
    that follows a rewrite re-probes every loop, which a whole-program kernel pays for nothing when there was nothing to
    rewrite."""

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
        rewrote = run_then = False
        for unit, watched in self.first:
            unit_changed = changed_the_graph(unit, unit.apply_pass(sdfg, {}))
            rewrote = rewrote or unit_changed
            run_then = run_then or (watched and unit_changed)
        if not run_then:
            return True if rewrote else None
        for unit in self.then:
            unit.apply_pass(sdfg, {})
        return True


# Per-target knob presets; explicit knob args to ``canonicalize(..., target=...)`` override them.
CPU_DEFAULTS: dict[str, Any] = {
    "interchange_carry_with_map": True,
    "scatter_to_guarded_maps": True,
    "normalize_loop_and_map_origin": False,
}
GPU_DEFAULTS: dict[str, Any] = {
    "interchange_carry_with_map": False,
    "scatter_to_guarded_maps": True,
    "normalize_loop_and_map_origin": False,
}
TARGET_DEFAULTS: dict[str, dict[str, Any]] = {"cpu": CPU_DEFAULTS, "gpu": GPU_DEFAULTS}


def _resolve_target_default(target: str, knob: str, explicit: Any | None, fallback: Any) -> Any:
    """Pick ``explicit`` if not ``None``, else the per-target preset, else ``fallback``."""
    if explicit is not None:
        return explicit
    return TARGET_DEFAULTS.get(target, {}).get(knob, fallback)


#: Indices one find-first chunk searches before the body below its hit runs (``LiftBreakToFindFirst``): on CPU a
#: chunk whose arrays stay in the last-level cache for the body's second read, on GPU about four times a full grid.
FIND_FIRST_CHUNK: dict[str, int] = {"cpu": 1 << 20, "gpu": 1 << 21}


# The recipe The phased canonicalization recipe: the paper's phases in order, with a structural cleanup between them.

Stages = list[tuple[str, ppl.Pass]]

NORMALIZE_STATEMENTS = "normalize_statements"
LIFT_SEMANTICS = "lift_semantics"
DERIVE_PARALLELISM = "derive_parallelism"
RECOMPOSE = "recompose"
#: Maximal fusion alone, so a caller can own the fusion decisions (NestForge stage 2).
FUSE = "fuse"
#: The raised and simplified input, before any phase.
INGEST = "ingest"
#: The between-phase rail.
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
    """Run fresh instances of a list of units, in order, until a round changes nothing, at most ``max_rounds``."""

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


@transformation.explicit_cf_compatible
class SimplifyAsParsed(ppl.Pass):
    """The frontend's ``to_sdfg(simplify=True)``: each nested SDFG simplified before its parent."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> bool | None:
        nested = [node.sdfg for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.NestedSDFG)]
        changed = [SimplifyPass().apply_pass(sd, {}) is not None for sd in [*reversed(nested), sdfg]]
        return any(changed) or None


def rail(label: str = CLEANUP) -> Stages:
    """The between-phase cleanup: flatten single-state bodies (an un-inlined body reports whole-array memlets),
    then the structural cleanup."""
    return [*_inline_single_state(label), (label, StructuralCleanup())]


def clean_anti_dependency_units() -> list[ppl.Pass]:
    """CleanAntiDependency (WAR, WAW) by renaming: per-scope versions of a container wherever a dominating write allows
    it, per-iteration copies of a constant slot a loop rewrites before reading."""
    return [
        PrivatizeScalarsAndArrays(),
        PromoteConstantIndexAccess(),
    ]


def dead_dataflow_units() -> list[ppl.Pass]:
    return [
        ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination(fold_ordered_copies=True, nested=True)]),
        DeadCarriedStoreElimination(),
    ]


def first_lift() -> ParallelizeLoops:
    """LoopToMap on every loop it accepts, sibling chains held for ``WavefrontSkew``."""
    lift = ParallelizeLoops(propagate=False, hold_sibling_chains=True)
    # A loop refused on a scratch buffer only lifts with the buffer expanded per iteration
    lift.expand_buffers = True
    return lift


def statement_round() -> list[ppl.Pass]:
    """One round of statement normalization: IV substitution, split, privatize, invariant motion, WCR normalization,
    dead dataflow."""
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
    units += clean_anti_dependency_units()
    units += [LoopInvariantCodeMotion()]
    units += dead_dataflow_units()
    return units


def ingest(target: str) -> Stages:
    """Raise structured control flow and settle one spelling, then the first ``SimplifyPass`` and cleanup."""
    s: Stages = [
        # The contract's compile-time half from the start: every proof below reads the sizes as nonnegative (a
        # symbolic loop step ``n3`` orients its map); ``AssumeSymbolConstraints`` adds the runtime check at the end
        (INGEST, SetSymbolNonnegativeAssumptions()),
        (INGEST, ControlFlowRaising()),
        (INGEST, RequireStructuredControlFlow()),
        # Once the control flow is structured: simplify does not converge on a graph that is not
        (INGEST, SimplifyAsParsed()),
        (INGEST, RemoveViews()),
        (INGEST, EliminateTrivialTasklets()),
        (INGEST, FoldScalarReadCopies()),
        (INGEST, CollapseNoOpCast()),
        (INGEST, NormalizeNegativeStride()),
        (INGEST, UniqueLoopIterators()),
        (INGEST, ContinueToCondition()),
        (INGEST, SupplyNumThreads()),
        # np.roll / broadcast_to / np.where as their pure maps from the start, so every normalization below sees a map
        # and not an opaque library node; the SimplifyPass right after inlines their nested SDFGs
        (INGEST, ExpandIndexRemaps()),
        (INGEST, SimplifyPass(pass_options=CANON_SIMPLIFY_OPTIONS)),
        # A table filled once with literals (CloudSC's ``iphase``) is a constant host code and every kernel read alike,
        # before a fusion moves its fill into a loop
        (INGEST, FoldConstantTables()),
        # A while loop that counts gets its counter here, so every later pass sees a counted loop.
        (INGEST, WhileToFor()),
        # After simplification: the raised guard of a break still reads its condition through a separate
        # preparation region before it, which the matcher does not see through.
        (INGEST, LiftBreakToFindFirst(chunk=FIND_FIRST_CHUNK[target])),
    ]
    return s + [(INGEST, StructuralCleanup())]


def normalize_statements(
    lift: bool,
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
    s += [
        (label, PatternApplyOnceEverywhere([MoveIfIntoMap()])),
        (label, ConvertLengthOneArraysToScalars()),
        # The frontend writes every ``+=`` in a map body as WCR; one that conflicts with nothing is a plain update
        (label, RevertNonReductionWCR()),
    ]
    lower_maps = MapToForLoop()
    lower_maps.keep_reductions_parallel = True
    # While updates are still maps: once lowered and unrolled, an update's add and its copy-back can land in different
    # states (gromacs's ``fshift[ish, :] += fi[i, :]``), out of the fold's reach. Again in every statement round.
    s += [(label, EliminateCopyBack())]
    s += [
        (label, PatternApplyOnceEverywhere([lower_maps], state_local=True)),
        (label, PruneAndInlineNestedSDFGs(multistate=True)),
    ]
    s += [(label, NormalizeNegativeStride()), (label, EliminateTrivialTasklets()), (label, StructuralCleanup())]
    s += [(label, RerollUnrolledLoops()), (label, UntileLoops()), (label, FuseConsecutiveLoops())]
    if unroll_limit > 0:
        s += [(label, ShortLoopUnroll(unroll_limit))]
    # Whether or not short loops unrolled: the statement round's induction-variable substitution closes a counter
    # bumped several times per iteration (``k = k + 1`` before each unrolled copy) only once it is in SSA form.
    s += [(label, UniqueLoopIterators()), (label, SymbolSSA())]
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
    ]
    if perfect_loop_nesting:
        s += [(label, PerfectLoopNesting(target=target))]
    s += [(label, UniqueLoopIterators()), (label, PatternApplyOnceEverywhere([TrivialLoopElimination()]))]
    s += [(label, LoopStridePermutation())]
    if normalize_loop_and_map_origin:
        s += [(label, NormalizeLoopAndMapOrigin())]
    # The special cleanup after normalization.
    s += [(label, SimplifyPass(pass_options=CANON_SIMPLIFY_OPTIONS))]
    return s


def lift_semantics(lift: bool, lift_copy: bool, interchange_carry_with_map: bool, target: str) -> Stages:
    """Recognize library operators on the normalized loop nests."""
    label = LIFT_SEMANTICS
    s: Stages = []
    # Before LoopToEinsum, which would take the inner product out of the nest.
    if lift:
        s += [(label, LoopToGetrfNoPivot(target=target)), (label, LoopToSymm()), (label, LoopToRankKUpdate())]
    s += [(label, LoopToSymmetrize())]
    if lift_copy:
        s += [(label, AssignmentAndCopyKernelToMemsetAndMemcpy())]
    if lift:
        s += [(label, LoopToTranspose()), (label, LoopToEinsum()), (label, NestToContraction())]
    s += [
        (label, SinkLastOnlyScanStore()),
        (label, LoopToReduce()),
        (label, LiftPreprocess()),
        (label, LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target)),
        (label, LoopToStateScan()),
        (label, ArgMaxLift()),
        (label, LoopToConditionalReduce()),
    ]
    if target == "gpu":
        # One guard per nest, then its gathered products become one batched call the compaction below feeds
        s += [(label, MergeGuards()), (label, BatchGatheredProducts())]
    s += [(label, LoopToStreamCompaction()), (label, LiftLoopCarriedReduction())]
    return s


def recurrence_lifts(interchange_carry_with_map: bool, target: str, only: list | None = None) -> list[ppl.Pass]:
    """The lifts of a loop-carried recurrence, in :func:`lift_semantics` order; ``only`` restricts them to those loops."""
    return [
        LoopToReduce(only=only),
        LoopToScan(interchange_carry_with_map=interchange_carry_with_map, target=target, only=only),
        LoopToStateScan(only=only),
        ArgMaxLift(only=only),
        LoopToConditionalReduce(only=only),
        LiftLoopCarriedReduction(only=only),
    ]


def derive_parallelism(
    peel_limit: int,
    scatter_to_guarded_maps: bool,
    assume_parallel_guards: bool,
    lift_copy: bool,
    interchange_carry_with_map: bool,
    target: str,
    validate: bool = False,
    wavefront: bool = True,
) -> Stages:
    """LoopToMap, rescue the loops it refused, LoopToMap again, guarded and reduction loops, wavefront skew."""
    label = DERIVE_PARALLELISM
    s: Stages = [(label, SymbolSSA())]
    # SSA leaves one definition per name, which is when propagation can substitute it (``j = j_0 + 1`` into ``b[j]``)
    s += [(label, PropagateSymbols())]
    # Before demotion: the anti-dependence proof reads a guard's symbols (``A[P]`` under ``P > i`` reads ahead), which
    # a demoted guard turns into opaque scalar reads.
    s += [(label, BreakAntiDependence(forward_reads=True)), (label, BreakAntiDependence())]
    s += [(label, DemoteNonStructuralSymbols()), (label, PrivatizeScalars())]
    s += [(label, DeadCarriedStoreElimination())]
    # Before the lift: a workspace whose pre-loop fill no iteration observes is then the loop's own, privatized per
    # iteration
    s += [(label, DeadFillElimination())]
    # Before the lift: a wrapping modulo the dependence proof would map as a modular scatter maps as its affine bands.
    segments: list = []
    split: list[tuple[ppl.Pass, bool]] = []
    rescue: list[tuple[ppl.Pass, bool]] = []
    if peel_limit > 0:
        split = [(ModuloBandSplit(peel_limit, segments=segments, validate=validate), True)]
        rescue = [
            (LoopCarriedRotationSubstitution(peel_limit), True),
            (BestEffortLoopPeeling(peel_limit, segments=segments, validate=validate), True),
        ]
    # The symbol units tidy what a band split, rotation or peel left: they run only when one happened.
    s += [
        (
            label,
            _ThenIfChanged(
                [
                    *split,
                    (first_lift(), False),
                    *rescue,
                    *((unit, False) for unit in dead_dataflow_units()),
                    (PrivatizeScalarsAndArrays(), False),
                ],
                # Peeling copies a loop with the symbols its body assigns
                [PropagateSymbols(), CascadeInterstateEdgeAssignmentsUp(), UniqueLoopIterators(), SymbolSSA()],
            ),
        )
    ]
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
        (label, AccumulatorCopyChainToWCR()),
        (label, ParallelizeLoops(propagate=False, hold_sibling_chains=True)),
        (label, PrivatizeScalarsAndArrays()),
    ]
    # Inlined bodies show their in-place accumulations
    s += [
        (
            label,
            _ThenIfChanged(
                [
                    *[(unit, isinstance(unit, PruneAndInlineNestedSDFGs)) for _, unit in _inline_single_state(label)],
                    (StructuralCleanup(), False),
                    # Once LoopToMap made the reductions maps': in-body ones surface onto the map boundary, which
                    # the lifts below and fusion read, and every WCR is sourced from an access node.
                    (NormalizeWCR(), False),
                    (CarriedAccumulationToWCR(), True),
                ],
                [
                    ParallelizeLoops(propagate=False, hold_sibling_chains=True),
                    PrivatizeScalarsAndArrays(),
                ],
            ),
        )
    ]
    if scatter_to_guarded_maps:
        s += [(label, SymbolPropagation())]
        s += [
            (
                label,
                ScatterToGuardedMaps(emit_unparallelized_else_branch=True, assume_no_conflicts=assume_parallel_guards),
            )
        ]
    # The third guarded derivation: a loop over the segments of a pointer array is lifted behind the entry guard that
    # checks the data facts its segment confinement rests on (nfa_frontier's components).
    s += [(label, _ThenIfChanged([(SegmentConfinedLoops(), True)], symbol_phase()))]
    # Wavefront: map what now maps, then skew the nests left sequential on a carried dependence.
    s += [(label, RevertNonReductionWCR()), *_inline_single_state(label)]
    # The accumulator steps leave the increment's tasklet a plain copy
    s += [(label, EliminateTrivialTasklets())]
    if target == "gpu":
        s += [(label, ppl.Pipeline([ReorderStateForLoopFusion()]))]
    s += [(label, FuseLoops())]
    s += [(label, Parallelize(target=target, hold_sibling_chains=False, wavefront=wavefront))]
    # The time loop a wavefront skew made keeps the skew's name until here
    s += [(label, UniqueLoopIterators())]
    return s


def maximal_fusion(target: str, after: Callable[[], list[ppl.Pass]] | None = None) -> PhaseFixpoint:
    """Maximal fusion: (FuseStates x FuseLoops x MapCollapse x FuseMaps, single-state bodies inlined first) to a
    fixpoint inside a fixpoint that re-permutes, interchanges and fuses conditions."""

    def inner_round() -> list[ppl.Pass]:
        return [
            # States first: map fusion only sees maps that share a state, and consolidated edges let more states fuse
            ConsolidateEdges(),
            HoistAssignmentsToLoopBodyHead(),
            FuseStates(aggressive_merges=True, order_array_seeds=True),
            # Loops next: two fused loops put their maps in one body
            FuseLoops(),
            PatternApplyOnceEverywhere([MapCollapse()]),
            PatternApplyOnceEverywhere([DistributeTaskletIntoMap()]),
            DetachDisjointReads(),
            ppl.Pipeline(
                [
                    FuseMaps(
                        allow_intermediates_with_symbolic_size=True,
                        partial_reduce_fusion=True,
                        relaxed_dataflow=True,
                    ),
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
            *(after() if after is not None else []),
        ]

    return PhaseFixpoint("maximal_fusion", outer_round, FUSION_OUTER_ROUNDS)


def after_fusion_units(lift: bool, lift_copy: bool, target: str) -> list[ppl.Pass]:
    """Fresh units that lift and tidy what fusion exposed: they close every round of maximal fusion, and a caller that
    chose the fusions itself runs them once before the ``finish`` stages."""
    # A reduction fused over one element leaves its init map
    units: list[ppl.Pass] = [PatternApplyOnceEverywhere([TrivialMapElimination()])]
    # Out of phase: these matchers need the fused map form (contractions split over maps, durbin's unary copy).
    if lift_copy:
        units += [AssignmentAndCopyKernelToMemsetAndMemcpy()]
    if lift:
        units += [PatternApplyOnceEverywhere([LiftEinsum(per_slice=target == "cpu")])]
    # Fusion joined a slice update's map with the copy of its result, across the states the frontend gave them.
    units += [
        EliminateCopyBack(),
        ppl.FixedPointPipeline([DeadDataflowElimination(), ArrayElimination(fold_ordered_copies=True, nested=True)]),
        PruneEmptyConditionalBranches(),
        PatternApplyOnceEverywhere([RedundantArray()]),
        RematerializeDerivedTemporaries(),
        PatternApplyOnceEverywhere([MapCollapse()]),
    ]
    return units


def recompose(lift: bool, lift_copy: bool, target: str) -> Stages:
    """(MapCollapse x FuseMaps x FuseStates) to a fixpoint inside a fixpoint that re-permutes, interchanges and fuses
    conditions; lift what fusion exposed; the final tidy."""
    label = RECOMPOSE
    # The maps' propagated memlets show always-executed accesses the frontend form hides, and fusion proves with them
    s: Stages = [(label, DeclareInBoundsExtents())]
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
    s += rail(label)
    # A memset the control flow after it overwrites unread (velocity's zeta_value, swept level by level) goes before
    # fusion, so its state does not stand between the maps; the finish runs it again for the fills lifted after.
    # A copy of a fill is a fill of its own (warpx's i_old = np.copy(i_new) after i_new = np.zeros(n)), dead or not.
    s += [(label, FillThroughCopies())]
    s += [(label, DeadFillElimination())]
    # The fills and copies left are maps from here on, so fusion joins them with their producers and consumers
    s += [(label, ExpandFillCopy())]

    s += [(FUSE, maximal_fusion(target, lambda: after_fusion_units(lift, lift_copy, target)))]
    return s


def finish() -> Stages:
    """The terminal bookkeeping: symbols, derived annotations, assumption guard, hints; the views inlining minted."""
    s: Stages = [(FINISH, StructuralCleanup())]
    s += [
        (FINISH, OptionalArrayInference()),
        (FINISH, ConvertLengthOneArraysToScalars(single_element=True)),
        (FINISH, DeadFillElimination()),
        (FINISH, PruneUnreferencedTransients()),
        (FINISH, InlineControlFlowRegions()),
        (FINISH, AssumeSymbolConstraints()),
        (FINISH, NormalizeFloorDivision()),
        # Spelled for the code generator once nothing analyzes the form any more: the runtime call is opaque to the
        # symbolic and SMT reasoning that sees a floored ``Mod`` through
        (FINISH, RewriteModuloToPyMod()),
        # Fusion leaves plain copies through its intermediates
        (FINISH, EliminateTrivialTasklets()),
        (FINISH, FoldScalarReadCopies(scopes=True)),
        # A staging copy sits in its readers' map scope, fed from the map entry, only once the terminal inlining has run
        (FINISH, PatternApplyOnceEverywhere([RedundantMapReadCopy()])),
        (FINISH, ppl.Pipeline([ArrayElimination(fold_ordered_copies=True, nested=True)])),
    ]
    return s


def build_stages(
    unroll_limit: int = 0,
    peel_limit: int = DEFAULT_PEEL_LIMIT,
    interchange_carry_with_map: bool = True,
    scatter_to_guarded_maps: bool = True,
    normalize_loop_and_map_origin: bool = False,
    assume_parallel_guards: bool = False,
    perfect_loop_nesting: bool = True,
    target: str = "cpu",
    lift: bool = True,
    lift_copy: bool = True,
    semantic_lifting: bool = True,
    validate: bool = False,
    wavefront: bool = True,
) -> Stages:
    """Build the canonicalization recipe: the paper's phases in order (see the section comment above)."""
    lift, lift_copy = semantic_lifting and lift, semantic_lifting and lift_copy
    s = ingest(target)
    s += normalize_statements(lift, unroll_limit, perfect_loop_nesting, target, normalize_loop_and_map_origin)
    s += lift_semantics(lift, lift_copy, interchange_carry_with_map, target)
    s += rail()
    s += derive_parallelism(
        peel_limit,
        scatter_to_guarded_maps,
        assume_parallel_guards,
        lift_copy,
        interchange_carry_with_map,
        target,
        validate,
        wavefront,
    )
    s += recompose(lift, lift_copy, target)
    s += finish()
    prepared([unit for stage_label, unit in s])
    return s


def disable_unit_validation(unit: ppl.Pass) -> None:
    """Turn off a stage unit's own validation, nested pipeline members included."""
    if isinstance(unit, (PatternMatchAndApply, FuseMaps)):
        unit.validate = False
        unit.validate_all = False
    if isinstance(unit, NormalizeWCR):
        unit.validate = False
    if isinstance(unit, ppl.Pipeline):
        for member in unit.passes:
            disable_unit_validation(member)
    if isinstance(unit, _ThenIfChanged):
        for member in (*(pair[0] for pair in unit.first), *unit.then):
            disable_unit_validation(member)


#: A stage factory returns that stage's fresh passes, in order.
StageFactory = Callable[[], list[ppl.Pass]]


def _stage_runs() -> list[tuple[str, int, int]]:
    """Split the flat recipe into contiguous runs of one stage label."""
    runs: list[list] = []
    for i, (lbl, unit) in enumerate(build_stages()):
        if runs and runs[-1][0] == lbl and runs[-1][2] == i:
            runs[-1][2] = i + 1
        else:
            runs.append([lbl, i, i + 1])
    return [(lbl, a, b) for lbl, a, b in runs]


def _stage_factory(start: int, stop: int) -> StageFactory:
    """Return a factory yielding fresh passes for one run of the recipe."""
    return lambda: [p for lbl, p in build_stages()[start:stop]]


#: Grouped view of :func:`build_stages`: ``(label, factory)`` per stage, in order, where ``factory()`` builds that
#: stage's fresh passes.
CANONICALIZE_STAGES: list[tuple[str, StageFactory]] = [
    (label, _stage_factory(start, stop)) for label, start, stop in _stage_runs()
]


def _assert_self_contained(unit: ppl.Pass) -> None:
    """Guard the empty-``pipeline_results`` invariant."""
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
    PrivatizeScalars,
    PrivatizeScalarsAndArrays,
)


class Settled:
    """The change clock of one run of units: a unit of :data:`SETTLED_UNITS` is skipped when no unit has changed the
    graph since it last ran."""

    __slots__ = ("clock", "ran_at")

    def __init__(self) -> None:
        self.clock = 0
        self.ran_at: dict[type, int] = {}

    @staticmethod
    def tracked(unit: ppl.Pass) -> bool:
        return type(unit) in SETTLED_UNITS and not getattr(unit, "multistate", False)

    def skip(self, unit: ppl.Pass) -> bool:
        return self.tracked(unit) and self.ran_at.get(type(unit)) == self.clock

    def record(self, unit: ppl.Pass, changed: bool, dirties: bool = True) -> None:
        """Note that ``unit`` ran; ``dirties`` is whether its change earns the settled units another run."""
        if changed and dirties:
            self.clock += 1
        if self.tracked(unit):
            # A non-dirtying composite (``StructuralCleanup``) runs a single sweep rather than a
            # fixpoint, so it is settled only once a sweep reports no change.
            if not changed or dirties:
                self.ran_at[type(unit)] = self.clock
            else:
                self.ran_at.pop(type(unit), None)


@properties.make_properties
@transformation.explicit_cf_compatible
class CanonicalizationPipeline(ppl.Pass):
    """Rewrite an SDFG into its canonical form."""

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
        "it is one whole-SDFG walk per modifying stage, so on a large graph it costs more than the passes "
        "it watches. Turn it on to bisect a stage that produced "
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
        dtype=int,
        default=0,
        desc="Unroll constant-trip loops <= this many iterations (0, the default, leaves them to the specializations).",
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
        desc="Run ScatterToGuardedMaps in derive_parallelism to lift scatter loops with a sort-based guard.",
    )
    normalize_loop_and_map_origin = properties.Property(
        dtype=bool,
        default=False,
        desc="Run NormalizeLoopAndMapOrigin at the end of normalize_statements, rebasing every Map range / "
        "LoopRegion counter to a 0-based begin while keeping the stride. Off by default on both targets "
        "(see CPU_DEFAULTS).",
    )
    perfect_loop_nesting = properties.Property(
        dtype=bool,
        default=True,
        desc="Distribute a loop over its data-independent statement groups (PerfectLoopNesting) so "
        "each statement parallelizes on its own axes. On by default.",
    )
    wavefront = properties.Property(
        dtype=bool,
        default=True,
        desc="Skew loops refused on a carried dependence into wavefronts (WavefrontSkew). Off, they stay sequential "
        "loops, for a caller that applies the skew itself. On by default.",
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
        unroll_limit: int = 0,
        peel_limit: int | None = None,
        target: str = "cpu",
        interchange_carry_with_map: bool | None = None,
        scatter_to_guarded_maps: bool | None = None,
        normalize_loop_and_map_origin: bool | None = None,
        assume_parallel_guards: bool = False,
        perfect_loop_nesting: bool = True,
        specialize_constants: dict[str, int] | None = None,
        lift: bool = True,
        lift_copy: bool = True,
        semantic_lifting: bool = True,
        dump_dir: str | None = None,
        stages: Sequence[str] | None = None,
        wavefront: bool = True,
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
        self.normalize_loop_and_map_origin = _resolve_target_default(
            target, "normalize_loop_and_map_origin", normalize_loop_and_map_origin, fallback=False
        )
        self.assume_parallel_guards = assume_parallel_guards
        self.perfect_loop_nesting = perfect_loop_nesting
        self.lift = lift
        self.lift_copy = lift_copy
        self.semantic_lifting = semantic_lifting
        self.stages = list(stages) if stages is not None else None
        self.wavefront = wavefront
        self._specialize_constants = specialize_constants or {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def build_stages(self) -> list[tuple[str, ppl.Pass]]:
        """Build this pipeline's flat recipe, honoring every knob property."""
        return build_stages(
            unroll_limit=self.unroll_limit,
            peel_limit=self.peel_limit,
            interchange_carry_with_map=self.interchange_carry_with_map,
            scatter_to_guarded_maps=self.scatter_to_guarded_maps,
            normalize_loop_and_map_origin=self.normalize_loop_and_map_origin,
            assume_parallel_guards=self.assume_parallel_guards,
            perfect_loop_nesting=self.perfect_loop_nesting,
            target=self.target,
            lift=self.lift,
            lift_copy=self.lift_copy,
            semantic_lifting=self.semantic_lifting,
            validate=self.validate,
            wavefront=self.wavefront,
        )

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Canonicalize ``sdfg`` in place."""
        disable_openmp_sections(sdfg)
        # Specialize chosen symbols to constants first (e.g. ``nclv = 5``), so the otherwise-symbolic species-loop trip
        # counts become concrete and unroll.
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
        # The cleanup block and the inline step are spliced in at every stage boundary, and on a settled graph they
        # report no change: skip one unless a unit touched the graph since it last ran.
        settled = Settled()
        recorder = canon_stats.of(sdfg)
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
            # apply_pass returns non-None iff it modified the SDFG, so an unchanged SDFG needs no re-validation
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
    """Ordered, de-duplicated stage labels of the canonicalization recipe for ``target``."""
    seen = OrderedSet(label for label, _ in CanonicalizationPipeline(target=target).build_stages())
    return list(seen)


def canonicalize(
    sdfg: SDFG,
    validate: bool = False,
    validate_all: bool = False,
    unroll_limit: int = 0,
    peel_limit: int | None = None,
    target: str = "cpu",
    interchange_carry_with_map: bool | None = None,
    scatter_to_guarded_maps: bool | None = None,
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
    wavefront: bool = True,
) -> SDFG:
    """Canonicalize ``sdfg`` in place and return it."""
    arguments = (
        validate,
        validate_all,
        unroll_limit,
        peel_limit,
        target,
        interchange_carry_with_map,
        scatter_to_guarded_maps,
        normalize_loop_and_map_origin,
        assume_parallel_guards,
        perfect_loop_nesting,
        specialize_constants,
        lift,
        lift_copy,
        semantic_lifting,
        dump_dir,
        stages,
        wavefront,
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
    declare_in_bounds_extents(sdfg)
    # Every stage below recovers loop bounds from STRING-backed properties, which means re-parsing them
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
    normalize_loop_and_map_origin: bool | None,
    assume_parallel_guards: bool,
    perfect_loop_nesting: bool,
    specialize_constants: dict[str, int] | None,
    lift: bool,
    lift_copy: bool,
    semantic_lifting: bool,
    dump_dir: str | None,
    stages: Sequence[str] | None = None,
    wavefront: bool = True,
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
        normalize_loop_and_map_origin=normalize_loop_and_map_origin,
        assume_parallel_guards=assume_parallel_guards,
        perfect_loop_nesting=perfect_loop_nesting,
        specialize_constants=specialize_constants,
        lift=lift,
        lift_copy=lift_copy,
        semantic_lifting=semantic_lifting,
        dump_dir=dump_dir,
        stages=stages,
        wavefront=wavefront,
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
    # Canonicalized output opts in to OpenMP array-section reduction codegen (whole-buffer WCR accumulators of a
    # parallel map -> ``reduction(op:A[0:n])`` instead of per-element atomics; complex via ``declare reduction``).
    for nested in sdfg.all_sdfgs_recursive():
        nested.openmp_array_reductions = True
    # Enforced, not hoped for: the pipeline opts out of omp sections at entry and exit, and a
    # caller who flipped ``compiler.cpu.openmp_sections`` globally must not be able to undo it.
    assert not any(nested.openmp_sections for nested in sdfg.all_sdfgs_recursive())
    return sdfg
