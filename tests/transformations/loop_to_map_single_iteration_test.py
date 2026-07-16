# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop that provably runs at most once carries no cross-iteration dependence, so ``LoopToMap``
maps it without dependence analysis (``loop_analysis.loop_provably_at_most_one_iteration``). This is
what lets the single-iteration ``Max``/``Min``-clamped middle segment a range split leaves behind
become a Map -- the dependence analysis is confounded by the clamp, but there is nothing to prove.
The negative cases pin soundness: a genuine multi-trip loop must NOT be admitted this way."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis.loop_analysis import loop_provably_at_most_one_iteration
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol('N', nonnegative=True)


def _mk(init, cond, upd):
    lp = LoopRegion('l', cond, 'i', init, upd)
    lp.add_state('s', is_start_block=True)
    return lp


@pytest.mark.parametrize('init,cond,upd,expected', [
    ('i = 0', 'i < N', 'i = i + 1', False),      # 0..N-1 -- genuinely multi-trip
    ('i = 1', 'i <= N', 'i = i + 1', False),     # 1..N   -- genuinely multi-trip
    ('i = 5', 'i < 6', 'i = i + 1', True),       # single iteration 5..5
    ('i = 5', 'i < 5', 'i = i + 1', True),       # empty 5..4 (zero iterations)
    ('i = 0', 'i < N', 'i = i + 2', False),      # unit-stride only -- stride 2 not admitted
])
def test_at_most_one_iteration_is_sound(init, cond, upd, expected):
    assert loop_provably_at_most_one_iteration(_mk(init, cond, upd)) is expected


def test_clamped_min_max_singleton_is_at_most_one():
    """The clamp shape a range split emits: ``max(x, 0) .. min(x, N-1)``. Whatever ``x`` is, this runs
    at most once (in-range -> 1 iteration; out-of-range -> 0), and must be provable."""
    lp = _mk('i = max(int_floor(N, 2), 0)', 'i < (min(int_floor(N, 2), N - 1) + 1)', 'i = i + 1')
    assert loop_provably_at_most_one_iteration(lp) is True


def test_single_iteration_loop_maps_and_is_bit_exact():
    """A hand-built single-iteration loop with an otherwise dependence-analysis-confounding clamped
    bound maps under LoopToMap and stays bit-exact."""
    M = dace.symbol('M', nonnegative=True)

    @dace.program
    def k(a: dace.float64[M], b: dace.float64[M]):
        # The middle-singleton shape: one iteration at the clamped index, writing b[j] = a[j] + 1.
        for j in range(max(M // 2, 0), min(M // 2, M - 1) + 1):
            b[j] = a[j] + 1.0

    a = np.random.default_rng(0).random(8)
    ref = a.copy()
    j = min(max(8 // 2, 0), 7)
    b_ref = np.zeros(8)
    b_ref[j] = a[j] + 1.0

    sdfg = k.to_sdfg(simplify=True)
    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied >= 1, 'the single-iteration loop must map'
    b = np.zeros(8)
    sdfg(a=a.copy(), b=b, M=8)
    assert np.array_equal(b, b_ref), f'got {b}, want {b_ref}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
