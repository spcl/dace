# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Adversarial soundness tests for ``LoopStridePermutation``.

The pass bubbles a unit-stride loop to innermost and certifies the move via the
``LoopToMap`` DOALL oracle. The oracle checks parallelism at the loop's CURRENT
(innermost, post-swap) position -- "parallelizable innermost" -- which is a
WEAKER property than "freely interchangeable DOALL". A dependence with a mixed
direction between the moved axis and a bubbled-past axis (the classic ``(<, >)``
interchange-preventing dependence) is parallelizable innermost yet its
interchange is illegal.

These tests pin down that the pass does NOT mis-schedule such shapes: it refuses
them (leaving the nest untouched), because the ``LoopToMap`` oracle only trusts a
cross-iteration disjointness proof drawn from dimensions indexed purely by the
iteration variable -- so a read/write pair that is non-aliasing only through a
bubbled-past axis's dimension is conservatively refused, which is exactly the
interchange-illegal set. ``_bounds_independent`` likewise rejects a triangular
bound on a bubbled-past iterator.

Each scenario compares the post-pass SDFG bit-exact against a numpy sequential
reference that mirrors the exact loop order. Each ``@dace.program`` is compiled
exactly once (no same-name double build) to avoid the shared-cache build race.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation

N = dace.symbol('N', nonnegative=True)


def _order(sdfg):
    order = []

    def walk(region):
        for blk in region.nodes():
            if isinstance(blk, LoopRegion):
                order.append(blk.loop_variable)
                walk(blk)

    walk(sdfg)
    return order


# ---------------------------------------------------------------------------
#  Rectangular mixed-direction dependence (i:<, j:>): unit-stride i is OUTER,
#  write a[j, i] reads a[j+1, i-1]. Moving i innermost reverses the anti-
#  dependence -> the interchange is ILLEGAL. i is "parallelizable innermost"
#  (for fixed j the write row j and read row j+1 never alias) but NOT freely
#  interchangeable. Must be refused.
# ---------------------------------------------------------------------------
@dace.program
def mixed_dir_2d(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(N - 1):
            a[j, i] = a[j + 1, i - 1] + b[j, i]


def _ref_mixed_dir_2d(a, b, n):
    for i in range(1, n):
        for j in range(n - 1):
            a[j, i] = a[j + 1, i - 1] + b[j, i]


def test_reject_mixed_direction_interchange_2d():
    n = 9
    rng = np.random.default_rng(0)
    a0 = rng.standard_normal((n, n))
    bb = rng.standard_normal((n, n))
    a_ref = a0.copy()
    _ref_mixed_dir_2d(a_ref, bb, n)

    sdfg = mixed_dir_2d.to_sdfg(simplify=True)
    before = _order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "an interchange-illegal (<, >) dependence must not be committed"
    assert _order(sdfg) == before

    a_got = a0.copy()
    sdfg(a=a_got, b=bb.copy(), N=n)
    assert np.array_equal(a_ref, a_got), "refused nest must compute the original result bit-exact"


# ---------------------------------------------------------------------------
#  3-level mixed-direction (i:<, j:>): unit-stride i outermost must bubble past
#  j and k. The i-past-j swap is illegal (read a[k, j+1, i-1]). The DOALL oracle
#  on the FINAL innermost position must not wrongly certify it.
# ---------------------------------------------------------------------------
@dace.program
def mixed_dir_3d(a: dace.float64[N, N, N], b: dace.float64[N, N, N]):
    for i in range(1, N):
        for j in range(N - 1):
            for k in range(N):
                a[k, j, i] = a[k, j + 1, i - 1] + b[k, j, i]


def _ref_mixed_dir_3d(a, b, n):
    for i in range(1, n):
        for j in range(n - 1):
            for k in range(n):
                a[k, j, i] = a[k, j + 1, i - 1] + b[k, j, i]


def test_reject_mixed_direction_interchange_3d():
    n = 7
    rng = np.random.default_rng(1)
    a0 = rng.standard_normal((n, n, n))
    bb = rng.standard_normal((n, n, n))
    a_ref = a0.copy()
    _ref_mixed_dir_3d(a_ref, bb, n)

    sdfg = mixed_dir_3d.to_sdfg(simplify=True)
    before = _order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "a 3-level illegal i-past-j swap must not be certified by the innermost oracle"
    assert _order(sdfg) == before

    a_got = a0.copy()
    sdfg(a=a_got, b=bb.copy(), N=n)
    assert np.array_equal(a_ref, a_got)


# ---------------------------------------------------------------------------
#  Triangular between the moved axis and a non-adjacent bubbled-past axis:
#  unit-stride i is DOALL (recurrence lives in j), but the innermost loop k has
#  a bound k in range(i, N) that references i. Bubbling i past k would change the
#  iteration SET -> _bounds_independent must reject the i-k swap.
# ---------------------------------------------------------------------------
@dace.program
def triangular_i_k(a: dace.float64[N, N, N], b: dace.float64[N, N, N]):
    for i in range(N):
        for j in range(1, N):
            for k in range(i, N):
                a[k, j, i] = a[k, j - 1, i] + b[k, j, i]


def _ref_triangular_i_k(a, b, n):
    for i in range(n):
        for j in range(1, n):
            for k in range(i, n):
                a[k, j, i] = a[k, j - 1, i] + b[k, j, i]


def test_reject_triangular_bound_on_bubbled_axis():
    n = 8
    rng = np.random.default_rng(2)
    a0 = rng.standard_normal((n, n, n))
    bb = rng.standard_normal((n, n, n))
    a_ref = a0.copy()
    _ref_triangular_i_k(a_ref, bb, n)

    sdfg = triangular_i_k.to_sdfg(simplify=True)
    before = _order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "a bound depending on a bubbled-past iterator must reject the swap"
    assert _order(sdfg) == before

    a_got = a0.copy()
    sdfg(a=a_got, b=bb.copy(), N=n)
    assert np.array_equal(a_ref, a_got)


# ---------------------------------------------------------------------------
#  Positive control: a genuinely DOALL unit-stride axis IS interchanged, and the
#  result is bit-exact (no reduction reordering, no data movement change).
# ---------------------------------------------------------------------------
@dace.program
def legal_doall_3d(a: dace.float64[N, N, N], b: dace.float64[N, N, N]):
    for i in range(N):
        for j in range(1, N):
            for k in range(1, N):
                a[k, j, i] = a[k - 1, j, i] + a[k, j - 1, i] + b[k, j, i]


def _ref_legal_doall_3d(a, b, n):
    for i in range(n):
        for j in range(1, n):
            for k in range(1, n):
                a[k, j, i] = a[k - 1, j, i] + a[k, j - 1, i] + b[k, j, i]


def test_legal_doall_interchange_is_bit_exact():
    n = 8
    rng = np.random.default_rng(3)
    a0 = rng.standard_normal((n, n, n))
    bb = rng.standard_normal((n, n, n))
    a_ref = a0.copy()
    _ref_legal_doall_3d(a_ref, bb, n)

    sdfg = legal_doall_3d.to_sdfg(simplify=True)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert applied == 1
    assert _order(sdfg)[-1] == 'i', "unit-stride i must be innermost"

    a_got = a0.copy()
    sdfg(a=a_got, b=bb.copy(), N=n)
    assert np.array_equal(a_ref, a_got), "a legal DOALL interchange must be bit-exact vs the sequential reference"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
