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

from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.conditional_component_fission import ConditionalComponentFission
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary

from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
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
    """Between-pass structural cleanup (never ``SimplifyPass`` mid-pipeline):
    ``StateFusionExtended``, then both inliners to a fixpoint.

    ``InlineSDFG`` only flattens a single-``SDFGState`` NestedSDFG;
    ``InlineMultistateSDFG`` flattens the control-flow-bearing NestedSDFGs
    that map->loop lowering produces (a NestedSDFG wrapping a ``LoopRegion``
    / ``ConditionalBlock``). Without the latter those NestedSDFGs are
    permanent, burying loops so ``MoveIfIntoLoop`` and cross-nest fusion
    cannot see them. Both run so single- and multi-state nestings collapse.

    :param label: The owning stage label.
    :returns: ``(stage_label, pass)`` pairs for the cleanup, in order.
    """
    return [(label, PatternMatchAndApplyRepeated([StateFusionExtended()])),
            (label, PatternMatchAndApplyRepeated([InlineMultistateSDFG()])),
            (label, PatternMatchAndApplyRepeated([InlineSDFG()]))]


def _build_stages() -> List[Tuple[str, ppl.Pass]]:
    """Build the loop-centric canonicalization recipe as one flat list.

    Every map is lowered to a ``LoopRegion`` up front so all canonicalization
    runs on a single representation (one fission/normalize/reduce path, no
    map/loop duplication, no hybrids); ``LoopToMap`` recovers parallelism near
    the end, then maps are fused. Returns ``(stage_label, pass)`` pairs with
    fresh instances each call.

    ``SimplifyPass`` runs **only** at the very start, after the cleaning
    passes (unique loop iterators, split tasklets, trivial-tasklet cleanup),
    and once at the end -- never between transforming stages. Between-stage
    structural cleanup is ``StateFusionExtended`` + ``InlineSDFG`` instead.
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

    # clean: unique loop iterators -> split tasklets -> drop trivial tasklets
    # -> the single SimplifyPass (only here and at the end).
    s += [('clean', _uniq), ('clean', SplitTasklets()),
          ('clean', PatternMatchAndApplyRepeated([TrivialTaskletElimination()])), ('clean', SimplifyPass())]

    # prep (still maps): push guarding conditionals into maps, then replicate
    # a conditional per independent output so it can fission later.
    s += [('prep', PatternMatchAndApplyRepeated([MoveIfIntoMap()])), ('prep', ConditionalComponentFission())]

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

    # move_if_into_loop: push guarding conditionals into loop bodies. The
    # genuine inner imperfect nest (a bare tasklet beside an inner loop,
    # inside an enclosing loop) takes the free-state path: the bare sibling
    # is wrapped in a trivial single-iteration loop and the guard duplicated
    # into every sibling loop. The wrapper is spliced out again by the
    # 'untrivialize' stage before LoopToMap can mangle it.
    s += [('move_if_into_loop', MoveIfIntoLoop())]

    # cascade_iedges_up (post-move-if): MoveIfIntoLoop may have buried an
    # invariant interstate-edge assignment (e.g. ``kfdia_plus_1 = kfdia + 1``,
    # the Python-frontend bound promotion) inside the loop it pushed the
    # guard into. Lift it back out *past every enclosing loop* (all-or-
    # nothing upward, see ``CASCADE_UP_DESIGN.md``) so LoopToMap's
    # body-assigns-loop-range-symbol refuse-check does not later block
    # parallelization. Standalone and idempotent -- can be invoked again.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # fission: loop distribution + block-level perfect-loop-nesting. Fission
    # clones a loop into siblings that keep the same ``_loop_it_<N>`` name;
    # re-running UniqueLoopIterators here disambiguates those duplicates so the
    # later LoopToMap is not blocked by a sibling appearing to read the shared
    # iterator after a parallelized loop.
    s += [('fission', LoopFission()), ('fission', _uniq_fis)]

    # normalize: every loop range -> 0:trip:1.
    s += [('normalize', NormalizeLoopsAndMaps())]

    # reduce / ssa: lift accumulator loops, unique loop iterators.
    s += [('reduce', LoopToReduce()), ('ssa', _uniq2)]

    # loop_stride_permutation (before LoopToMap): no-op stub. A loop-level
    # interchange would need a loop-interchange primitive (none exists) and
    # loop-carried-dependence analysis; instead the loops that *can* become
    # maps are permuted as maps right after LoopToMap (see 'reorder' below),
    # which is dependence-free by the Map contract and reuses the proven,
    # symbolic-safe MinimizeStridePermutation.
    s += [('loop_stride_permutation', LoopStridePermutation())]

    # untrivialize: the perfect-nesting scaffold (the single-iteration trivial
    # loops MoveIfIntoLoop wrapped bare siblings in) has done its job for
    # fission/normalize/reduce/ssa. Splice it out *while still a LoopRegion*
    # (reusing TrivialLoopElimination) -- before LoopToMap, which would
    # otherwise turn it into a sticky NestedSDFG that breaks idempotence and
    # re-lowering.
    s += [('untrivialize', PatternMatchAndApplyRepeated([TrivialLoopElimination()]))]

    # cascade_iedges_up (pre-parallelize): re-run after fission / normalize /
    # ssa, since each of those rewrites the CFG and may expose new hoisting
    # opportunities (e.g. an iedge that was previously inside a body block
    # is now the only block on a clean linear chain). Critically: this MUST
    # run before LoopToMap so the refuse-check on body-assigned range
    # symbols sees the cleaned-up shape.
    s += [('cascade_iedges_up', CascadeInterstateEdgeAssignmentsUp())]

    # NOTE: MoveLoopInvariantIfUp is deliberately NOT wired at this
    # pre-parallelize point. It is the dual of the earlier ``MoveIfIntoLoop``
    # stage (which pushes guards INTO loops to enable sibling-loop fusion), so
    # hoisting guards back out here would simply undo that work and ping-pong.
    # The terminal ``hoist_guards`` stage runs MLIU once, AFTER fuse, where the
    # fusion it would otherwise undo has already happened.

    # parallelize: canonical loops -> parallel maps.
    s += [('parallelize', PatternMatchAndApplyRepeated([LoopToMap()]))]

    # post_l2m: insert assign tasklets at map boundary, then structural
    # cleanup (state fusion + inline SDFG) -- after LoopToMap.
    s += [('post_l2m', InsertAssignTaskletsAtMapBoundary())]
    s += _structural_cleanup('post_l2m')

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

    # end: the final SimplifyPass.
    s += [('end', SimplifyPass())]
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
    :param validate_all: Validate the SDFG after each stage.
    """

    CATEGORY: str = 'Canonicalization'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')

    def __init__(self, validate: bool = False, validate_all: bool = False):
        self.validate = validate
        self.validate_all = validate_all

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
        stages = _build_stages()
        for _label, unit in stages:
            _assert_self_contained(unit)
            unit.apply_pass(sdfg, {})
            if self.validate_all:
                sdfg.validate()
        if self.validate:
            sdfg.validate()
        return len(stages)


def canonicalize(sdfg: SDFG, validate: bool = True, validate_all: bool = False) -> SDFG:
    """Canonicalize ``sdfg`` in place and return it.

    One-call recipe analogous to ``auto_optimize``.

    :param sdfg: The SDFG to canonicalize.
    :param validate: Validate the SDFG after canonicalization.
    :param validate_all: Validate the SDFG after each stage.
    :returns: The same ``sdfg`` instance, canonicalized.
    """
    CanonicalizationPipeline(validate=validate, validate_all=validate_all).apply_pass(sdfg, {})
    return sdfg
