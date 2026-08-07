# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Placement invariants of the between-phase structural cleanup.

Breaking these is silent: a matcher that no longer sees its shape reports no
matches, and the value-preservation corpus stays green while parallelism drops.
"""
import pytest

from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.prune_connectors import PruneConnectors
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
from dace.transformation.passes.canonicalize.induction_variable_substitution import LoopCarriedRotationSubstitution
from dace.transformation.passes.canonicalize.pipeline import _build_stages, _structural_cleanup
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.simplify import SimplifyPass

#: Exact composition of the shared cleanup helper, in order.
CLEANUP_COMPOSITION = ['StateFusionExtended', 'EmptyStateElimination', 'DeadStateElimination']

#: The fold pass whose absence costs tsvc s252 its rematerialized producer.
SCALAR_SLICE_FOLD = 'CleanAccessNodeToScalarSliceToTaskletPattern'


def _names(passes):
    """:returns: one name per entry; a fixpoint contributes its member transformation names."""
    out = []
    for p in passes:
        if isinstance(p, PatternMatchAndApplyRepeated):
            out.extend(type(t).__name__ for t in p.transformations)
        else:
            out.append(type(p).__name__)
    return out


def _flat(target: str = 'cpu'):
    """:returns: the recipe as ``(stage_label, name)`` pairs, fixpoints flattened to their members."""
    return [(label, name) for label, p in _build_stages(target=target) for name in _names([p])]


def _indices(flat, name: str):
    """:returns: every index in ``flat`` at which ``name`` runs."""
    return [i for i, (_lbl, n) in enumerate(flat) if n == name]


def test_cleanup_helper_composition_is_state_machine_only():
    """A dataflow rewrite in this list would fire at every site by accident of cleanup placement."""
    assert _names([p for _lbl, p in _structural_cleanup('probe')]) == CLEANUP_COMPOSITION


def test_cleanup_helper_carries_the_owning_stage_label():
    """Attribution per stage, so stage-by-stage debugging stays honest."""
    assert [lbl for lbl, _p in _structural_cleanup('probe')] == ['probe'] * len(CLEANUP_COMPOSITION)


@pytest.mark.parametrize('name', [SimplifyPass.__name__, ConstantPropagation.__name__])
def test_cleanup_helper_never_carries_a_whole_pipeline_pass(name):
    """Neither is neutral, so neither may run at a dozen sites."""
    assert name not in _names([p for _lbl, p in _structural_cleanup('probe')])


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_multistate_inline_runs_exactly_once_at_the_lowering_stage(target):
    """Map->loop lowering is the only stage that mints control-flow-bearing nestings."""
    flat = _flat(target)
    at = _indices(flat, InlineMultistateSDFG.__name__)
    assert len(at) == 1, f'InlineMultistateSDFG runs {len(at)} times; the recipe allows exactly one'
    lower = _indices(flat, 'MapToForLoop')
    assert lower, 'MapToForLoop left the recipe -- re-home this constraint with it'
    assert flat[at[0]][0] == 'lower'
    assert lower[0] < at[0], 'the multistate inline must follow the lowering that mints the nestings'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_every_loop_to_map_reaches_map_fusion_through_an_inline(target):
    """An un-inlined map body reports a whole-array memlet, so fusion refuses on the bounding box."""
    flat = _flat(target)
    l2m = _indices(flat, LoopToMap.__name__)
    fusion = sorted(_indices(flat, MapFusionVertical.__name__) + _indices(flat, MapFusionHorizontal.__name__))
    inlines = _indices(flat, InlineSDFG.__name__)
    assert l2m and fusion and inlines
    for i in l2m:
        after = [f for f in fusion if f > i]
        if not after:
            continue
        nxt = after[0]
        assert any(i < j < nxt for j in inlines), \
            f'LoopToMap at {flat[i][0]}:{i} reaches map fusion at {flat[nxt][0]}:{nxt} with no InlineSDFG between'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_every_cleanup_past_the_first_loop_to_map_inlines_first(target):
    """The skew and wavefront matchers read map-body subsets too, not just fusion."""
    flat = _flat(target)
    l2m = _indices(flat, LoopToMap.__name__)
    assert l2m, 'LoopToMap left the recipe -- re-home this constraint with it'
    inlines = _indices(flat, InlineSDFG.__name__)
    for i in _indices(flat, DeadStateElimination.__name__):
        if i < l2m[0]:
            continue
        assert any(i - 6 < j < i for j in inlines), f'cleanup at {flat[i][0]}:{i} runs with no preceding inline'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_prune_connectors_leads_every_inline(target):
    """A dead NestedSDFG connector is a hard ``InlineSDFG`` refusal, so pruning shares the fixpoint."""
    for _label, p in _build_stages(target=target):
        if not isinstance(p, PatternMatchAndApplyRepeated):
            continue
        names = [type(t).__name__ for t in p.transformations]
        if InlineSDFG.__name__ not in names and InlineMultistateSDFG.__name__ not in names:
            continue
        assert names[0] == PruneConnectors.__name__, f'inliner fixpoint {names} does not start with PruneConnectors'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_cleanup_sites_all_use_the_shared_helper(target):
    """No site may grow a private variant of the cleanup."""
    flat = _flat(target)
    empty = _indices(flat, EmptyStateElimination.__name__)
    dead = _indices(flat, DeadStateElimination.__name__)
    assert dead, 'DeadStateElimination left the recipe'
    for i in dead:
        assert i - 1 in empty, f'cleanup at {flat[i][0]}:{i} is not preceded by EmptyStateElimination'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_scalar_slice_folds_precede_every_matcher_that_reads_subscripts(target):
    """Behind the synthetic transient the producer reads a scalar with no index to shift (s252)."""
    flat = _flat(target)
    folds = _indices(flat, SCALAR_SLICE_FOLD)
    assert folds, 'the scalar-slice fold left the recipe -- re-home this constraint with it'
    for consumer in (LoopCarriedRotationSubstitution.__name__, 'LoopToReduce', 'LoopToScan'):
        at = _indices(flat, consumer)
        assert at, f'{consumer} left the recipe -- re-home this constraint with it'
        assert any(f < at[0] for f in folds), f'{consumer} runs before any scalar-slice fold'


def test_scalar_slice_folds_are_not_part_of_the_cleanup_helper():
    """Placed, not sprayed: two sites is a decision, twelve is a side effect."""
    assert SCALAR_SLICE_FOLD not in _names([p for _lbl, p in _structural_cleanup('probe')])
