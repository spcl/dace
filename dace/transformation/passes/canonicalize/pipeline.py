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

from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.tasklet_fusion import TaskletFusion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting

from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap

#: A stage is an ordered list of passes applied directly. ``Pipeline`` and
#: ``PatternMatchAndApplyRepeated`` are unhashable, so they cannot be composed
#: as nodes of an outer :class:`~dace.transformation.pass_pipeline.Pipeline`.
StageFactory = Callable[[], List[ppl.Pass]]


def _pre_simplify() -> List[ppl.Pass]:
    """Stage 0: preparation.

    Make every loop iterator name unique up front (``SSALoopIterators``) so no
    later pattern match or fusion is blocked by incidental name reuse, then
    bring the SDFG into a stable explicit-control-flow form (``SimplifyPass``).
    """
    return [SSALoopIterators(), SimplifyPass()]


def _split_tasklets() -> List[ppl.Pass]:
    """Stage 1b: split compound tasklets into single-op tasklets."""
    return [SplitTasklets()]


def _prepare_fission() -> List[ppl.Pass]:
    """Stage 1c: push conditionals into maps and flatten structure so the
    next stage can fission maximally. ``SimplifyPass`` inlines nested SDFGs
    (and fuses states / prunes dead structure) -- the structure-simplifying
    passes fission depends on."""
    return [PatternMatchAndApplyRepeated([MoveIfIntoMap()]), SimplifyPass()]


def _maximal_fission() -> List[ppl.Pass]:
    """Stage 2: fission every independent computation into its own map.

    The loop-level fission step is a TODO (its code lives outside this
    subpackage).
    """
    return [PatternMatchAndApplyRepeated([MapExpansion(), MapFission()]), SimplifyPass()]


def _reorder_offsets() -> List[ppl.Pass]:
    """Stage 3: normalize every map range to 0-based / unit-step; then
    simplify structure before the next stage."""
    return [NormalizeLoopsAndMaps(), SimplifyPass()]


def _perfect_loop_nesting() -> List[ppl.Pass]:
    """Stage 4: make nests perfectly nested, then inline produced nested SDFGs."""
    return [PatternMatchAndApplyRepeated([PerfLoopNesting()]), SimplifyPass()]


def _normalize() -> List[ppl.Pass]:
    """Stage 5: canonicalize iterators/IVs/invariants, then lift reductions.

    Order is load-bearing: unique iterators, then induction-variable
    canonicalization, then loop-invariant code motion, then ``LoopToReduce``
    (strictly after maximal fission).
    """
    return [SSALoopIterators(), SimplifyInductionVariables(), LoopInvariantCodeMotion(), LoopToReduce(), SimplifyPass()]


def _loop_to_map() -> List[ppl.Pass]:
    """Stage 6: convert canonical sequential loops into parallel maps."""
    return [PatternMatchAndApplyRepeated([LoopToMap()])]


def _maximal_fusion() -> List[ppl.Pass]:
    """Stage 7: simplify, fuse maps maximally, recompose split tasklets."""
    return [
        SimplifyPass(),
        ppl.Pipeline([FullMapFusion()]),
        PatternMatchAndApplyRepeated([TaskletFusion(), TrivialTaskletElimination()]),
    ]


def _hoist_if() -> List[ppl.Pass]:
    """Stage 8: hoist loop-invariant conditionals above their maps.

    No-op for now; the hoisting transformation is not yet wired.
    """
    return []


#: Ordered pipeline stages, applied once top to bottom. Stage 1 (``Untile``) and
#: the Stage-2 loop-fission step are intentionally absent until their code lands
#: (it lives outside this subpackage).
CANONICALIZE_STAGES: List[Tuple[str, StageFactory]] = [
    ('pre_simplify', _pre_simplify),
    ('split_tasklets', _split_tasklets),
    ('prepare_fission', _prepare_fission),
    ('maximal_fission', _maximal_fission),
    ('reorder_offsets', _reorder_offsets),
    ('perfect_loop_nesting', _perfect_loop_nesting),
    ('normalize', _normalize),
    ('loop_to_map', _loop_to_map),
    ('maximal_fusion', _maximal_fusion),
    ('hoist_if', _hoist_if),
]


@properties.make_properties
@transformation.explicit_cf_compatible
class CanonicalizationPipeline(ppl.Pass):
    """Rewrite an SDFG into its canonical form.

    Stages (:data:`CANONICALIZE_STAGES`) are applied once, imperatively, in
    order, as ``auto_optimize`` does. A single
    :class:`~dace.transformation.pass_pipeline.Pipeline` cannot be used because
    it forbids duplicate pass types and the recipe reuses ``SimplifyPass`` and
    ``PatternMatchAndApplyRepeated`` across stages. Stages that need iteration
    iterate internally; the pipeline itself does not re-run.

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
        :returns: The number of stages applied.
        """
        for _name, factory in CANONICALIZE_STAGES:
            for unit in factory():
                unit.apply_pass(sdfg, {})
                if self.validate_all:
                    sdfg.validate()
        if self.validate:
            sdfg.validate()
        return len(CANONICALIZE_STAGES)


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
