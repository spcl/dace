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
from dace.transformation.passes.ssa_loop_iterators import SSALoopIterators
from dace.transformation.passes.simplify_induction_variables import SimplifyInductionVariables
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.full_map_fusion import FullMapFusion
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.insert_unit_copy_assign_tasklets import InsertAssignTaskletsForUnitCopies

from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.tasklet_fusion import TaskletFusion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign

from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap


def _cleanup(loop_to_reduce: bool) -> List[ppl.Pass]:
    """Copy/WCR normalization cleanup, reused in preparation and between
    transforming stages.

    Decomposes write-conflict resolutions into explicit augmented-assignment
    subgraphs (``WCRToAugAssign``) and removes copy-shaped edges that later
    subset-substituting passes (e.g. ``NormalizeLoopsAndMaps``) would have to
    special-case: map-boundary staging and ``other_subset`` copies
    (``InsertAssignTaskletsAtMapBoundary``) and single-element
    ``AccessNode -> AccessNode`` copies (``InsertAssignTaskletsForUnitCopies``).

    ``LoopToReduce`` is gated by ``loop_to_reduce`` because it is only sound
    *after* maximal fission has isolated each accumulator into its own
    stride-1 loop (it must never run in early preparation); the
    ``maximal_fission`` / ``perfect_loop_nesting`` / ``normalize`` stages are
    the re-run points where a fresh reducible loop can appear (see
    ``DESIGN.md``).

    :param loop_to_reduce: Also lift scalar-accumulator loops to ``Reduce``.
    """
    passes: List[ppl.Pass] = [
        PatternMatchAndApplyRepeated([WCRToAugAssign()]),
        InsertAssignTaskletsAtMapBoundary(),
        InsertAssignTaskletsForUnitCopies(),
    ]
    if loop_to_reduce:
        passes.append(LoopToReduce())
    return passes


def _build_stages() -> List[Tuple[str, ppl.Pass]]:
    """Build the canonicalization recipe as one flat ordered list.

    Returns ``(stage_label, pass)`` pairs in application order, with fresh
    pass instances each call (passes are stateful). The label repeats per
    pass and is only used for ``validate_all`` reporting. Ordering is
    load-bearing; ``DESIGN.md`` is the authoritative rationale, summarized in
    one comment per stage boundary here. ``Untile`` (Stage 1) and the Stage-2
    loop-fission step are intentionally absent until their code lands (it
    lives outside this subpackage).

    Composites (``SimplifyPass``, ``PatternMatchAndApplyRepeated``, the
    ``Pipeline`` around ``FullMapFusion``) stay nested by necessity: they are
    fixed-point / dependency-resolving machinery and cannot collapse into a
    flat pass list. ``FullMapFusion`` in particular *must* be wrapped in a
    ``Pipeline`` so its ``FindSingleUseData`` dependency is resolved.
    """
    s: List[Tuple[str, ppl.Pass]] = []

    # Stage 0 pre_simplify: unique loop iterators first so no later pattern
    # match / fusion is blocked by name reuse, then stable explicit-CF form.
    s += [('pre_simplify', SSALoopIterators()), ('pre_simplify', SimplifyPass())]

    # Stage 1b split_tasklets: atomic single-op tasklets so fission isolates each.
    s += [('split_tasklets', SplitTasklets())]

    # Stage 1c prepare_fission: clean copy/WCR shapes (no LoopToReduce yet --
    # accumulators are still fused pre-fission), push conditionals into maps,
    # then SimplifyPass inlines nested SDFGs / flattens structure.
    s += [('prepare_fission', p) for p in _cleanup(loop_to_reduce=False)]
    s += [('prepare_fission', PatternMatchAndApplyRepeated([MoveIfIntoMap()])), ('prepare_fission', SimplifyPass())]

    # Stage 2 maximal_fission: fission every independent computation into its
    # own map. First point LoopToReduce is sound (accumulators just isolated).
    s += [('maximal_fission', PatternMatchAndApplyRepeated([MapExpansion(), MapFission()])),
          ('maximal_fission', SimplifyPass())]
    s += [('maximal_fission', p) for p in _cleanup(loop_to_reduce=True)]

    # Stage 3 reorder_offsets: every map range -> 0-based / unit-step.
    s += [('reorder_offsets', NormalizeLoopsAndMaps()), ('reorder_offsets', SimplifyPass())]

    # Stage 4 perfect_loop_nesting: parent-nest duplication can expose new
    # isolated accumulator loops, so the cleanup re-runs with LoopToReduce.
    s += [('perfect_loop_nesting', PatternMatchAndApplyRepeated([PerfLoopNesting()])),
          ('perfect_loop_nesting', SimplifyPass())]
    s += [('perfect_loop_nesting', p) for p in _cleanup(loop_to_reduce=True)]

    # Stage 5 normalize: load-bearing order -- unique iterators, IV canon, LICM,
    # then LoopToReduce (canonical run, after IV canonicalization).
    s += [('normalize', SSALoopIterators()), ('normalize', SimplifyInductionVariables()),
          ('normalize', LoopInvariantCodeMotion()), ('normalize', LoopToReduce()), ('normalize', SimplifyPass())]

    # Stage 6 loop_to_map: canonical sequential loops -> parallel maps.
    s += [('loop_to_map', PatternMatchAndApplyRepeated([LoopToMap()]))]

    # Stage 7 maximal_fusion: simplify, fuse maps maximally, recompose tasklets.
    s += [('maximal_fusion', SimplifyPass()), ('maximal_fusion', ppl.Pipeline([FullMapFusion()])),
          ('maximal_fusion', PatternMatchAndApplyRepeated([TaskletFusion(),
                                                           TrivialTaskletElimination()]))]

    # Stage 8 hoist_if: terminal, no-op until the hoisting transformation lands.
    return s


#: A stage factory returns that stage's fresh passes, in order.
StageFactory = Callable[[], List[ppl.Pass]]


def _stage_factory(label: str) -> StageFactory:
    """Return a factory yielding fresh passes for the named stage."""
    return lambda: [p for lbl, p in _build_stages() if lbl == label]


#: Backward-compatible grouped view of :func:`_build_stages`: ``(label,
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
