# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`."""
import copy
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = dace.symbol('N')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


# -----------------------------------------------------------------------------
# Case A -- inner is ``range(0, K)`` and body accesses via ``i + ii``.
# -----------------------------------------------------------------------------

def test_case_a_combined_access_K4_collapses_to_single_loop():
    """``for i in range(0, N, 4): for ii in range(4): a[i+ii] = b[i+ii]`` -- the
    canonical two-level tile that an unrolling pass would otherwise expand into
    4 straight-line copies. UntileLoops produces a single ``for k in range(N)``
    whose body sees the same memlets with ``i + ii`` substituted by ``k``."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[i + ii] = b[i + ii]

    n = 16
    rng = np.random.default_rng(0)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = b.copy()

    sdfg = tiled.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 2
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops_after = _loops(sdfg)
    assert len(loops_after) == 1, f'expected 1 collapsed loop, got {len(loops_after)}'
    assert loops_after[0].loop_variable.startswith('_untile_k_')

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a), f'value mismatch: got {a}, expected {ref_a}'


def test_case_a_with_arithmetic_combination_collapses():
    """Case A still applies when memlets use a richer affine combination of
    ``i + ii`` (e.g. ``2*(i+ii)`` indexes into a larger array) as long as every
    appearance of ``i`` co-occurs with ``ii`` and vice-versa."""

    @dace.program
    def tiled(a: dace.float64[2 * N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[2 * (i + ii)] = b[i + ii]

    n = 12
    rng = np.random.default_rng(1)
    a = np.zeros(2 * n)
    b = rng.standard_normal(n)
    ref_a = a.copy()
    for k in range(n):
        ref_a[2 * k] = b[k]

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a)


# -----------------------------------------------------------------------------
# Case B -- inner is ``range(i, i+K)`` and body accesses via ``ii``.
# -----------------------------------------------------------------------------

def test_case_b_absolute_inner_collapses_to_single_loop():
    """``for i in range(0, N, 4): for ii in range(i, i+4): a[ii] = b[ii]``."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(i, i + 4):
                a[ii] = b[ii]

    n = 8
    rng = np.random.default_rng(2)
    a = np.zeros(n)
    b = rng.standard_normal(n)
    ref_a = b.copy()

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, ref_a)


# -----------------------------------------------------------------------------
# Refusal contracts.
# -----------------------------------------------------------------------------

def test_refuses_when_outer_stride_is_not_concrete_int():
    """``for i in range(0, N, K)`` with ``K`` symbolic -- can't collapse without
    a literal stride to match against the inner trip count."""
    K_sym = dace.symbol('K_step')

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, K_sym):
            for ii in range(K_sym):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_outer_stride_is_one():
    """``for i in range(0, N, 1)`` is already untiled; the pass declines so it
    doesn't endlessly rename loops that are already in canonical form."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N):
            for ii in range(1):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_inner_trip_does_not_match_outer_stride():
    """Inner trip ``range(0, 3)`` doesn't match outer stride ``4``; the loop
    nest doesn't represent a complete tiling and UntileLoops refuses."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(3):  # mismatched trip
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_body_uses_bare_outer_iterator():
    """Case A but the body uses ``a[i]`` (bare ``i``, no ``ii``) -- collapsing
    to ``k`` would lose the per-tile granularity. Refuse."""

    @dace.program
    def tiled(a: dace.float64[N]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[i] = 1.0  # bare ``i`` without ``ii``

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_body_uses_bare_inner_iterator():
    """Case A but the body uses ``a[ii]`` -- bare inner iterator means the
    access is tile-relative, not combined; refuse to keep the rewrite sound."""

    @dace.program
    def tiled(a: dace.float64[4]):
        for i in range(0, N, 4):
            for ii in range(4):
                a[ii] = 1.0  # bare ``ii`` without ``i``

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_outer_start_is_not_zero():
    """``for i in range(P, N, 4)`` with ``P != 0`` -- the collapsed iterator
    needs a shifted range that's harder to verify safely; v1 sticks to start=0."""

    @dace.program
    def tiled(a: dace.float64[N + 8]):
        for i in range(8, N, 4):
            for ii in range(4):
                a[i + ii] = 1.0

    sdfg = tiled.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_outer_body_is_not_a_perfect_two_level_nest():
    """A bare tasklet beside the inner loop -- imperfect nest -- is refused."""

    @dace.program
    def imperfect(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 4):
            b[i] = 0.0  # bare body block beside the inner loop
            for ii in range(4):
                a[i + ii] = 1.0

    sdfg = imperfect.to_sdfg(simplify=True)
    res = UntileLoops().apply_pass(sdfg, {})
    assert res is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
