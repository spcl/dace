# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Closing the guard on a near-boundary index-set split (tsvc_2_5 ``ext_peel_multi_back``).

A back-conflict loop whose two tail iterations write conflicting elements is split at
``x = LEN_1D - 2`` (carving the last iterations into a disjoint-write remainder that maps).
:meth:`BestEffortLoopPeeling._split_loop_at` regroups the loop's own iterations only while
``start <= x <= end``; the end side ``x <= end`` is ``LEN_1D - 2 <= LEN_1D - 1`` (trivially true),
and the start side ``start <= x`` is ``0 <= LEN_1D - 2``.

Pure symbol-nonnegativity cannot decide ``LEN_1D - 2 >= 0``, so historically the split was emitted
under a runtime guard ``if (0 <= LEN_1D - 2) { map } else { sequential loop }``. But the same
large-trip-count assumption the modulo splits already lean on -- a symbolic loop being peeled/split
by ``k <= peel_limit`` iterations runs MORE than ``peel_limit`` times, so its trip count
``LEN_1D > peel_limit`` -- proves ``LEN_1D - 2 = trip - 2 >= 0`` statically. The guard is then
always-true: the pinned sequential fallback is dropped and the loop is UNCONDITIONALLY a map.

The two guard-preserving companions (a free ``K`` split point, an out-of-range one) live in
``strengthen2_parallelization_prep_test.py``; this file pins the newly-CLOSED case.
"""
import numpy as np

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling

LEN_1D = dace.symbol('LEN_1D', nonnegative=True)


@dace.program
def back_peel_multi(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] * 2.0
        if i == LEN_1D - 1:
            a[LEN_1D - 2] = a[LEN_1D - 2] + 1.0
        elif i == LEN_1D - 2:
            a[LEN_1D - 3] = a[LEN_1D - 3] + 1.0


def _reference(a, b, n):
    for i in range(n):
        a[i] = b[i] * 2.0
        if i == n - 1:
            a[n - 2] = a[n - 2] + 1.0
        elif i == n - 2:
            a[n - 3] = a[n - 3] + 1.0
    return a


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _guarded_fallback_loops(sdfg) -> int:
    """LoopRegions that are the sequential fallback of an ``if cond: map else: seq`` guard
    (a Map-less branch of a ConditionalBlock whose sibling holds a Map)."""
    n = 0
    for cfr in sdfg.all_control_flow_regions():
        if not isinstance(cfr, ConditionalBlock):
            continue
        has_map = [any(isinstance(x, nd.MapEntry) for x, _ in br.all_nodes_recursive()) for _, br in cfr.branches]
        if not any(has_map):
            continue
        for hm, (_, br) in zip(has_map, cfr.branches):
            if not hm:
                n += sum(1 for s in br.all_control_flow_regions(recursive=True) if isinstance(s, LoopRegion))
    return n


def test_split_range_relations_are_discharged_under_the_large_trip_assumption():
    """The start-side membership ``0 <= LEN_1D - 2`` must be proven statically, so the index-set
    split emits NO runtime guard (empty relation set)."""
    sdfg = back_peel_multi.to_sdfg(simplify=True)
    loops = _loops(sdfg)
    assert len(loops) == 1
    loop = loops[0]
    peel = BestEffortLoopPeeling(peel_limit=4)
    x = peel._best_split_for(loop, sdfg)
    assert x is not None, 'expected an index-set split point for the back-conflict loop'
    # The split point is the second-to-last iteration: x == end - 1 (compared against the loop's
    # own bound symbols so the assumption-tagged copies cancel).
    end = loop_analysis.get_loop_end(loop)
    assert dace.symbolic.simplify(x - (end - 1)) == 0, f'split point should be end - 1, got {x}'
    relations = peel._split_range_relations(loop, x)
    assert relations == frozenset(), f'both membership sides must be provable, got guard {relations}'


def test_back_peel_split_drops_the_pinned_sequential_fallback():
    """After the split the loop is UNCONDITIONALLY parallelized: no guarded sequential fallback
    remains (g: 1 -> 0) and a Map is present."""
    sdfg = back_peel_multi.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 1
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()
    assert _guarded_fallback_loops(sdfg) == 0, 'the pinned sequential fallback must be dropped'
    # No pinned-sequential loop should survive (the split has no fallback branch at all).
    assert not any(r.pinned_sequential for r in _loops(sdfg))


def test_back_peel_split_is_bit_exact_against_the_sequential_reference():
    """The unconditionally-split form reproduces the sequential result exactly."""
    sdfg = back_peel_multi.to_sdfg(simplify=True)
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    n = 128
    rng = np.random.default_rng(1234)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n)

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), LEN_1D=n)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12), 'split form must match the sequential meaning'


if __name__ == '__main__':
    test_split_range_relations_are_discharged_under_the_large_trip_assumption()
    test_back_peel_split_drops_the_pinned_sequential_fallback()
    test_back_peel_split_is_bit_exact_against_the_sequential_reference()
    print('OK')
