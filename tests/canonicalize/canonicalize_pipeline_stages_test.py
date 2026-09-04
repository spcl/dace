# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Additional end-to-end unit tests for the full ``canonicalize`` pipeline,
    targeting stage paths not covered by ``canonicalize_pipeline_test.py``:
    accumulator reduction, perfect-loop-nesting fission, a partially-shared
    transient (partial fission), a conditional with an ``else`` branch,
    indirect scatter, and a guarded multi-statement stencil (MoveIfIntoMap
    -> fission -> fuse -> conditional recombination).

    Every test asserts the canonicalized SDFG validates and is numerically
    identical to a deep-copied pre-canonicalization run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def accumulator(a: dace.float64[N], s: dace.float64[1]):
    acc = np.float64(0.0)
    for i in range(N):
        acc += a[i]
    s[0] = acc


@dace.program
def perfect_nest(a: dace.float64[N, M], b: dace.float64[N, M], c: dace.float64[N, M]):
    for i in dace.map[0:N]:
        for j in dace.map[0:M]:
            b[i, j] = a[i, j] + 1.0
        for j in dace.map[0:M]:
            c[i, j] = a[i, j] * 2.0


@dace.program
def shared_transient(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N], cc: dace.float64[N]):
    for i in dace.map[0:N]:
        t = a[i] * 2.0
        b[i] = t + 1.0
        d[i] = cc[i] * 3.0


@dace.program
def cond_else(a: dace.float64[N], b: dace.float64[N], act: dace.int32[1]):
    if act[0] > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] + 1.0
    else:
        for i in dace.map[0:N]:
            b[i] = a[i] - 1.0


@dace.program
def scatter(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], cc: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[idx[i]] = a[i] * 2.0
        e[idx[i]] = cc[i] + 1.0


@dace.program
def guarded_two_stencils(a: dace.float64[N], b: dace.float64[N], cc: dace.float64[N], d: dace.float64[N],
                         act: dace.int32[1]):
    if act[0] > 0:
        for i in dace.map[1:N - 1]:
            b[i] = a[i - 1] + a[i] + a[i + 1]
            d[i] = cc[i - 1] + cc[i] + cc[i + 1]


def test_canonicalize_accumulator_reduction():
    n = 25
    a = np.random.rand(n)
    ref = np.zeros(1)
    copy.deepcopy(accumulator.to_sdfg(simplify=True))(a=a.copy(), s=ref, N=n)

    sdfg = accumulator.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros(1)
    sdfg(a=a.copy(), s=out, N=n)
    assert np.allclose(out, ref) and np.allclose(out, a.sum())


def test_canonicalize_perfect_loop_nesting():
    n, m = 14, 10
    a = np.random.rand(n, m)
    ref_b, ref_c = np.zeros((n, m)), np.zeros((n, m))
    copy.deepcopy(perfect_nest.to_sdfg(simplify=True))(a=a.copy(), b=ref_b, c=ref_c, N=n, M=m)

    sdfg = perfect_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_c = np.zeros((n, m)), np.zeros((n, m))
    sdfg(a=a.copy(), b=out_b, c=out_c, N=n, M=m)
    assert np.allclose(out_b, ref_b) and np.allclose(out_c, ref_c)
    assert np.allclose(out_b, a + 1.0) and np.allclose(out_c, a * 2.0)


def test_canonicalize_partially_shared_transient():
    n = 20
    a, cc = np.random.rand(n), np.random.rand(n)
    ref_b, ref_d = np.zeros(n), np.zeros(n)
    copy.deepcopy(shared_transient.to_sdfg(simplify=True))(a=a.copy(), b=ref_b, d=ref_d, cc=cc.copy(), N=n)

    sdfg = shared_transient.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_d = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, d=out_d, cc=cc.copy(), N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d)
    assert np.allclose(out_b, a * 2.0 + 1.0) and np.allclose(out_d, cc * 3.0)


@pytest.mark.parametrize('av', [1, 0])
def test_canonicalize_conditional_with_else(av):
    n = 18
    a = np.random.rand(n)
    ref = np.zeros(n)
    copy.deepcopy(cond_else.to_sdfg(simplify=True))(a=a.copy(), b=ref, act=np.array([av], np.int32), N=n)

    sdfg = cond_else.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, act=np.array([av], np.int32), N=n)
    assert np.allclose(out, ref), f"mismatch act={av}"
    assert np.allclose(out, a + 1.0 if av > 0 else a - 1.0)


def test_canonicalize_indirect_scatter():
    n = 22
    a, cc = np.random.rand(n), np.random.rand(n)
    idx = np.random.permutation(n).astype(np.int32)
    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(scatter.to_sdfg(simplify=True))(a=a.copy(), idx=idx.copy(), b=ref_b, cc=cc.copy(), e=ref_e, N=n)

    sdfg = scatter.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, cc=cc.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    exp_b, exp_e = np.zeros(n), np.zeros(n)
    exp_b[idx] = a * 2.0
    exp_e[idx] = cc + 1.0
    assert np.allclose(out_b, exp_b) and np.allclose(out_e, exp_e)


@pytest.mark.parametrize('av', [1, 0])
def test_canonicalize_guarded_two_stencils(av):
    """Guard + two independent stencils: MoveIfIntoMap -> fission -> fuse ->
    conditional recombination, value-preserving for guard taken/not-taken."""
    n = 24
    a, cc = np.random.rand(n), np.random.rand(n)
    ref_b, ref_d = np.full(n, 5.0), np.full(n, 5.0)
    copy.deepcopy(guarded_two_stencils.to_sdfg(simplify=True))(a=a.copy(),
                                                               b=ref_b,
                                                               cc=cc.copy(),
                                                               d=ref_d,
                                                               act=np.array([av], np.int32),
                                                               N=n)

    sdfg = guarded_two_stencils.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_d = np.full(n, 5.0), np.full(n, 5.0)
    sdfg(a=a.copy(), b=out_b, cc=cc.copy(), d=out_d, act=np.array([av], np.int32), N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d), f"mismatch act={av}"
    if av > 0:
        exp_b, exp_d = np.full(n, 5.0), np.full(n, 5.0)
        exp_b[1:n - 1] = a[0:n - 2] + a[1:n - 1] + a[2:n]
        exp_d[1:n - 1] = cc[0:n - 2] + cc[1:n - 1] + cc[2:n]
        assert np.allclose(out_b, exp_b) and np.allclose(out_d, exp_d)
    else:
        assert np.allclose(out_b, 5.0) and np.allclose(out_d, 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
