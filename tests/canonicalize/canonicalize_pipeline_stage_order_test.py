# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering invariants of the canonicalization recipe.

These are the constraints that make the recipe's passes able to fire at all.
They are easy to break by inserting a pass in the wrong place -- and breaking
them is silent, because a pass that can never match simply reports no matches
and the value-preservation corpus stays green.
"""
import pytest

from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
from dace.transformation.passes.canonicalize import pipeline as canon_pipeline
from dace.transformation.passes.canonicalize.pipeline import (CANONICALIZE_STAGES, _assert_self_contained,
                                                              _build_stages)
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.symbol_propagation import SymbolPropagation


def _flat(target: str = 'cpu'):
    """The recipe as ``(stage_label, pass_type_name, inner_transformation_names)``."""
    out = []
    for label, p in _build_stages(target=target):
        inner = []
        if isinstance(p, PatternMatchAndApplyRepeated):
            inner = [type(t).__name__ for t in p.transformations]
        elif isinstance(p, canon_pipeline.IvSubstitutionFissionFixpoint):
            inner = [type(u).__name__ for u in p.round_units()]
        out.append((label, type(p).__name__, inner))
    return out


def _first_index(flat, name: str) -> int:
    for i, (_lbl, cls, inner) in enumerate(flat):
        if cls == name or name in inner:
            return i
    return -1


def _last_index(flat, name: str) -> int:
    for i in range(len(flat) - 1, -1, -1):
        _lbl, cls, inner = flat[i]
        if cls == name or name in inner:
            return i
    return -1


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_stride_permutation_precedes_any_map_collapse(target):
    """``MinimizeStridePermutation`` only walks chains of single-parameter maps.

    ``_collect_perfect_nest`` breaks out on a multi-parameter map and
    ``_reorder_nest`` needs at least two levels, so once ``MapCollapse`` has
    folded ``map i: { map j }`` into ``map[i, j]`` the permuter can no longer
    see it. Collapsing first would make the pass a guaranteed no-op on exactly
    the fully-parallel nests it exists to reorder.
    """
    flat = _flat(target)
    permute = _first_index(flat, MinimizeStridePermutation.__name__)
    collapse = _first_index(flat, MapCollapse.__name__)
    assert permute != -1 and collapse != -1
    assert permute < collapse, 'MapCollapse before MinimizeStridePermutation makes the permuter dead'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_map_collapse_precedes_map_fusion(target):
    """Collapsing before fusing is what keeps differently-parallel nests apart.

    An N-dimensional map no longer matches a sibling 1-D map for horizontal
    fusion, so a parallel ``map[i, j]`` beside a carried ``map i: { loop j }``
    survives. Fusing while both are still 1-D re-merges them into a single
    mixed-parallelism map.
    """
    flat = _flat(target)
    collapse = _first_index(flat, MapCollapse.__name__)
    vertical = _first_index(flat, MapFusionVertical.__name__)
    horizontal = _first_index(flat, MapFusionHorizontal.__name__)
    assert collapse != -1 and vertical != -1 and horizontal != -1
    assert collapse < vertical, 'fusion before collapse re-merges differently-parallel nests'
    assert collapse < horizontal, 'fusion before collapse re-merges differently-parallel nests'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_coalesce_runs_after_parallelize_and_before_fusion(target):
    """The phase is only meaningful once DOALL loops have become maps."""
    labels = [lbl for lbl, _cls, _inner in _flat(target)]
    assert 'coalesce' in labels
    assert labels.index('parallelize') < labels.index('coalesce')
    assert labels.index('coalesce') < labels.index('fuse')


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_trivial_loop_scaffold_is_removed_before_the_matchers(target):
    """``MoveIfIntoLoop`` wraps bare siblings in single-iteration loops.

    Every matcher after fission expects a loop body to be one ``SDFGState``:
    ``LoopStridePermutation`` would absorb the wrapper as a real nest level,
    and the copy/memset lift plus the ``LoopTo*`` family refuse outright on a
    body that is a ``LoopRegion``. The scaffold must therefore be spliced out
    before them, not merely before ``LoopToMap``.
    """
    flat = _flat(target)
    untrivialize = _first_index(flat, TrivialLoopElimination.__name__)
    assert untrivialize != -1
    for blocked in (LoopStridePermutation.__name__, 'FuseConsecutiveLoops', 'LoopToReduce', 'LoopToScan'):
        idx = _first_index(flat, blocked)
        # Presence, not ``if idx != -1``: a matcher dropped from the recipe must fail here so the
        # constraint is re-homed with it, never silently skipped into a passing no-op assertion.
        assert idx != -1, f'{blocked} left the recipe -- move this ordering constraint to its new pipeline'
        assert untrivialize < idx, f'{blocked} runs with the trivial-loop scaffold still in place'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_negative_stride_is_normalized_after_maps_become_loops(target):
    """``NormalizeNegativeStride`` only rewrites ``LoopRegion`` s.

    Running it solely before ``MapToForLoop`` leaves a negative-stride *map*
    unnormalized: it becomes a negative-stride loop only after the pass's one
    chance to see it. Downstream matchers do rely on the positive-stride
    invariant, so a normalization must follow the lowering.
    """
    flat = _flat(target)
    lower = _first_index(flat, 'MapToForLoop')
    assert lower != -1
    after = [i for i, (_l, cls, inner) in enumerate(flat) if NormalizeNegativeStride.__name__ in (cls, *inner)]
    assert any(i > lower for i in after), 'no NormalizeNegativeStride after maps are lowered to loops'


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_unused_symbols_are_pruned_after_the_terminal_propagation(target):
    """The terminal SymbolPropagation/ConstantPropagation strip symbols from
    interstate edges but leave the stale entries in ``sdfg.symbols`` -- nothing
    after them prunes ``sdfg.symbols`` unless RemoveUnusedSymbols runs last.
    """
    flat = _flat(target)
    prune = _last_index(flat, RemoveUnusedSymbols.__name__)
    symprop = _last_index(flat, SymbolPropagation.__name__)
    constprop = _last_index(flat, ConstantPropagation.__name__)
    assert prune != -1 and symprop != -1 and constprop != -1
    assert prune > symprop, 'RemoveUnusedSymbols must run after the terminal SymbolPropagation'
    assert prune > constprop, 'RemoveUnusedSymbols must run after the terminal ConstantPropagation'


#: Semantic lift -> the rewrite that would blind it by destroying the shape it matches on
#: (``LowerITEToFpFactor`` folds the ``1 if i == j else 0`` identity tasklet ``LiftInv`` needs into
#: arithmetic; ``SplitTasklets`` breaks up the multi-statement body ``LoopToSymm`` matches). Both
#: rewrites now live in the VECTORIZER, not in canonicalization -- see
#: ``tests/passes/vectorization/vectorize_stage_order_test.py`` for the ordering half.
SEMANTIC_LIFT_BEFORE = (('LiftInv', 'LowerITEToFpFactor'), ('LoopToSymm', 'SplitTasklets'))


def test_stage_grouping_preserves_recipe_order():
    """``CANONICALIZE_STAGES`` must be the flat recipe, only chunked.

    Several labels appear in more than one place (``clean`` most of all). Grouping by label
    alone would gather the separated occurrences at the first one and reorder the recipe --
    breaking every constraint the other tests here assert, all of which hold between passes
    that share a label but sit on opposite sides of another stage.
    """
    flat = [(lbl, type(p).__name__) for lbl, p in _build_stages()]
    grouped = [(lbl, type(p).__name__) for lbl, factory in CANONICALIZE_STAGES for p in factory()]
    assert grouped == flat


def test_semantic_lifts_are_present_and_their_rewrites_are_not():
    """The ``LiftInv`` / ``LoopToSymm`` ordering constraints, asserted as PRESENCE.

    Written as ``if earlier in names and later in names: assert index(...)`` these two pairs
    became silent no-ops the moment ``LowerITEToFpFactor`` / ``SplitTasklets`` left the recipe
    for the vectorizer -- two assertions asserting nothing, in a green test. So assert the two
    halves separately: the lift must still be HERE (else the constraint has no subject), and its
    rewrite must still be ELSEWHERE (else the ordering assertion has to come back).
    """
    names = [type(p).__name__ for _lbl, p in _build_stages()]
    for earlier, later in SEMANTIC_LIFT_BEFORE:
        assert earlier in names, (f'{earlier} left the recipe -- move the "{earlier} before {later}" '
                                  f'constraint to whichever pipeline runs it now')
        assert later not in names, (f'{later} is back in the recipe -- restore the strict '
                                    f'"index({earlier}) < index({later})" assertion here')


def test_guard_hoists_are_target_gated():
    """CPU hoists a guard as far as it goes; GPU only clears a whole chain."""
    for target, expected in (('cpu', False), ('gpu', True)):
        stages = _build_stages(target=target)
        hoists = [p for lbl, p in stages if lbl == 'hoist_guards']
        assert hoists, 'no hoist_guards stage'
        flags = [p.require_full_hoist for p in hoists if hasattr(p, 'require_full_hoist')]
        assert flags, 'no hoist exposes require_full_hoist'
        assert all(f is expected for f in flags), f'{target} hoists should use require_full_hoist={expected}'


_KNOBS = ('break_anti_dependence', 'interchange_carry_with_map', 'scatter_to_guarded_maps',
          'privatize_scatter_reductions', 'reconstruct_wavefront_nest', 'normalize_loop_and_map_origin',
          'assume_parallel_guards', 'lift', 'lift_copy', 'semantic_lifting')


@pytest.mark.parametrize('target', ('cpu', 'gpu'))
@pytest.mark.parametrize('knob', _KNOBS)
def test_every_stage_unit_is_self_contained(target: str, knob: str):
    """Every unit the recipe emits satisfies the invariant ``apply_pass`` enforces on it.

    Stages are applied with an EMPTY ``pipeline_results`` dict, so a bare dependency-bearing pass
    would silently lose its inputs; :func:`_assert_self_contained` turns that into an error at
    apply time. Checking it here instead means a mis-wired stage is caught by building the recipe,
    not by a corpus kernel getting far enough into ``canonicalize`` to reach the offending unit --
    which is what happened to ``ReorderStateForLoopFusion``: it was wired bare, and only a corpus run found it.
    """
    for flipped in (False, True):
        for _label, unit in _build_stages(target=target, **{knob: flipped}):
            _assert_self_contained(unit)


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_simplify_never_runs_after_the_reduce_stage(target):
    """``SimplifyPass`` is confined to ``clean`` and ``reduce``; nothing past them may lean on it.

    A terminal simplify makes every later stage's precondition invisible: it re-inlines, re-fuses
    states and re-folds copy chains, so a stage that only matches on already-normalized input still
    looks like it works. The stages after ``reduce`` -- the terminal LoopToMap, the terminal fuse,
    the remat and symbol-cleanup band -- have to hold on un-simplified input instead, and the
    reclaimers they DO need are wired explicitly (see
    ``tests/passes/canonicalize/terminal_cleanup_band_test.py``). Asserting the confinement here is what
    keeps a re-added simplify from silently restoring the crutch.
    """
    # A simplify OWNED by a composite stage counts: confinement is about where simplification
    # happens, and a wrapper past the reduce stage would restore the crutch just as effectively.
    labels = [lbl for lbl, cls, inner in _flat(target) if SimplifyPass.__name__ in (cls, *inner)]
    assert labels == ['clean', 'reduce', 'reduce'], labels


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_the_reclaimers_replace_the_terminal_simplify(target):
    """What the recipe kept of the terminal simplify: the two reclaimers, and only those.

    They must sit AFTER the last ``SimplifyPass`` (otherwise the simplify would do their work and
    they would be dead weight) and BEFORE the terminal ``LoopToMap`` (which is what the reclaimed
    graph feeds).
    """
    stages = _build_stages(target=target)
    reclaim = [
        i for i, (_lbl, p) in enumerate(stages)
        if isinstance(p, ppl.FixedPointPipeline) and not isinstance(p, SimplifyPass)
        and p._pass_names == {DeadDataflowElimination.__name__, ArrayElimination.__name__}
    ]
    assert len(reclaim) == 1, f'expected exactly one reclaim stage, found {reclaim}'
    flat = _flat(target)
    assert reclaim[0] > _last_index(flat, SimplifyPass.__name__), 'the reclaimers must follow the last simplify'
    assert reclaim[0] < _last_index(flat, 'LoopToMap'), 'the reclaimers must precede the terminal LoopToMap'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
