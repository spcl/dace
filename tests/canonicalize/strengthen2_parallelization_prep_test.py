# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Soundness of the ``BestEffortLoopPeeling`` index-set split at an OUT-OF-RANGE split point.

``_split_loop_at`` regroups ``loop`` into [start, x-1] + {x} + [x+1, end]. That is a pure
regrouping of the same iterations only while ``start <= x <= end``. A split point is merely
LOOP-INVARIANT, though -- an ``if i == K`` guard with a free symbol ``K`` is a perfectly ordinary
candidate whose position relative to the loop bounds is undecidable at transform time. When ``K``
lands OUTSIDE the range at runtime the split must degenerate to the original range, not invent
iterations the loop never ran.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling

N = dace.symbol('N', nonnegative=True)
K = dace.symbol('K', nonnegative=True)


@dace.program
def guarded_special_case(a: dace.float64[N], b: dace.float64[N]):
    # `i == K` is an index-set-split candidate; the body's `a[N - 1]` broadcast read against the
    # `a[i]` write is what keeps the whole loop off LoopToMap, so the split gets applied.
    for i in range(1, N):
        if i == K:
            a[i] = a[i] + a[N - 1]
        else:
            a[i] = b[i]


def _reference(a, b, n, k):
    """The sequential meaning: the loop starts at 1, so `i` NEVER equals a `k` below 1."""
    for i in range(1, n):
        if i == k:
            a[i] = a[i] + a[n - 1]
        else:
            a[i] = b[i]
    return a


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def test_split_point_below_loop_start_does_not_invent_an_iteration():
    """``K == 0`` is below the loop's start of 1, so the guard never fires and ``a[0]`` is
    untouched. The split must not run the carved-out ``{K}`` iteration at ``i = 0``."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 1

    n, k = 16, 0
    rng = np.random.default_rng(0)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref), f'a[0] must stay {ref[0]!r} (the loop starts at 1), got {got[0]!r}'


def test_split_point_inside_the_range_is_still_value_preserving():
    """The in-range companion: ``K == 5`` genuinely selects an iteration, and the split must
    reproduce the sequential result exactly."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)

    n, k = 16, 5
    rng = np.random.default_rng(1)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref)


def test_split_point_past_loop_end_does_not_run_past_the_end():
    """``K >= N`` is past the loop's end, so the guard never fires. Neither the carved-out ``{K}``
    iteration nor the ``[start, K-1]`` segment may run beyond the original last iteration."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)

    n, k = 16, 16
    rng = np.random.default_rng(2)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref)
