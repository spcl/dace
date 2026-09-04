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
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.simplify import SimplifyPass

#: Exact composition of the shared cleanup helper, in order: a symbol phase, then a structural one.
#: The symbol phase is what keeps two names for one address from reaching a consumer that compares
#: subsets syntactically (``AugAssignToWCR`` answering "different slots" turned an indirect
#: accumulate into a scatter that aborted at run time), and it closes with a second ``SymbolDedup``
#: because the prune deletes assignments and so mints merges no earlier dedup could see.
CLEANUP_COMPOSITION = [
    'SymbolDedup', 'SymbolPropagation', 'ConstantPropagation', 'RemoveUnusedSymbols', 'SymbolDedup',
    'StateFusionExtended', 'EmptyStateElimination', 'DeadStateElimination', 'RedundantOrderingEdgeElimination'
]

#: The symbol phase, in order. Split out because the structural phase's own placement invariants
#: are stated against indices within the block, and those shift when this grows.
SYMBOL_PHASE = CLEANUP_COMPOSITION[:CLEANUP_COMPOSITION.index('StateFusionExtended')]

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
    """Symbols and the state machine only. A dataflow rewrite in this list would fire at every site
    by accident of cleanup placement, and ``RedundantOrderingEdgeElimination`` -- the one member
    that works inside a state -- is last for that reason: fusing two states is what turns an
    ordering edge that was load-bearing into one the merged dataflow already implies."""
    assert _names([p for _lbl, p in _structural_cleanup('probe')]) == CLEANUP_COMPOSITION


def test_cleanup_helper_carries_the_owning_stage_label():
    """Attribution per stage, so stage-by-stage debugging stays honest."""
    assert [lbl for lbl, _p in _structural_cleanup('probe')] == ['probe'] * len(CLEANUP_COMPOSITION)


def test_cleanup_helper_never_carries_a_whole_pipeline_pass():
    """``SimplifyPass`` is a pipeline, not a tidy-up, and running one at a dozen sites hides which
    stage actually changed the graph.

    ``ConstantPropagation`` is the deliberate exception and is asserted as a MEMBER below: it is
    there to re-fold the chains a dedup exposes, in the symbol phase, and only its own targeted
    rewrite runs."""
    assert SimplifyPass.__name__ not in _names([p for _lbl, p in _structural_cleanup('probe')])


def test_symbol_phase_leads_the_helper_in_its_measured_order():
    """Order is the claim, not membership: propagation and folding rewrite the assignments the
    first dedup merged, which is what exposes the equal-RHS pairs the closing dedup merges, and the
    closing dedup runs AFTER the prune because the prune is itself a merge-minting deletion."""
    names = _names([p for _lbl, p in _structural_cleanup('probe')])
    assert names[:len(SYMBOL_PHASE)] == SYMBOL_PHASE
    assert names[len(SYMBOL_PHASE) - 1] == 'SymbolDedup', 'the closing dedup must follow the prune'


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
    """The skew and wavefront matchers read map-body subsets too, not just fusion.

    Stated against the cleanup BLOCK rather than a fixed lookback window: the inline has to run in
    the work this cleanup closes, which is everything since the previous cleanup, and a window of
    n entries silently stops meaning that the moment the helper's own length changes.
    """
    flat = _flat(target)
    l2m = _indices(flat, LoopToMap.__name__)
    assert l2m, 'LoopToMap left the recipe -- re-home this constraint with it'
    inlines = _indices(flat, InlineSDFG.__name__)
    dse_offset = CLEANUP_COMPOSITION.index(DeadStateElimination.__name__)
    previous_block_end = l2m[0]
    for i in _indices(flat, DeadStateElimination.__name__):
        if i < l2m[0]:
            previous_block_end = i
            continue
        block_start = i - dse_offset
        assert flat[block_start][1] == CLEANUP_COMPOSITION[0], \
            f'the cleanup ending at {flat[i][0]}:{i} is not the shared helper: {flat[block_start]}'
        assert any(previous_block_end < j < block_start for j in inlines), \
            f'cleanup at {flat[i][0]}:{i} closes work that was never inlined'
        previous_block_end = i


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
