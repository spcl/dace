# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering invariants of the canonicalization recipe.

These are the constraints that make the recipe's passes able to fire at all.
They are easy to break by inserting a pass in the wrong place -- and breaking
them is silent, because a pass that can never match simply reports no matches
and the value-preservation corpus stays green.
"""
import pytest

from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES, _build_stages
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated


def _flat(target: str = 'cpu'):
    """The recipe as ``(stage_label, pass_type_name, inner_transformation_names)``."""
    out = []
    for label, p in _build_stages(target=target):
        inner = []
        if isinstance(p, PatternMatchAndApplyRepeated):
            inner = [type(t).__name__ for t in getattr(p, 'transformations', [])]
        out.append((label, type(p).__name__, inner))
    return out


def _first_index(flat, name: str) -> int:
    for i, (_lbl, cls, inner) in enumerate(flat):
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
        if idx != -1:
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


def test_stage_grouping_preserves_recipe_order():
    """``CANONICALIZE_STAGES`` must be the flat recipe, only chunked.

    Several labels appear in more than one place. Grouping by label alone would
    gather the separated occurrences at the first one and reorder the recipe --
    which breaks documented constraints, e.g. ``LiftInv`` must run before
    ``LowerITEToFpFactor`` rewrites the identity tasklet it matches on, though
    both are ``clean`` passes on opposite sides of the ``lift_inv`` stage.
    """
    flat = [(lbl, type(p).__name__) for lbl, p in _build_stages()]
    grouped = [(lbl, type(p).__name__) for lbl, factory in CANONICALIZE_STAGES for p in factory()]
    assert grouped == flat

    names = [cls for _lbl, cls in grouped]
    for earlier, later in (('LiftInv', 'LowerITEToFpFactor'), ('LoopToSymm', 'SplitTasklets')):
        if earlier in names and later in names:
            assert names.index(earlier) < names.index(later), f'{earlier} must precede {later} in the grouped view'


def test_guard_hoists_are_target_gated():
    """CPU hoists a guard as far as it goes; GPU only clears a whole chain."""
    for target, expected in (('cpu', False), ('gpu', True)):
        stages = _build_stages(target=target)
        hoists = [p for lbl, p in stages if lbl == 'hoist_guards']
        assert hoists, 'no hoist_guards stage'
        flags = [p.require_full_hoist for p in hoists if hasattr(p, 'require_full_hoist')]
        assert flags, 'no hoist exposes require_full_hoist'
        assert all(f is expected for f in flags), f'{target} hoists should use require_full_hoist={expected}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
