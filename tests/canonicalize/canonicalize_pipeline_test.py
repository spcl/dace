# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" End-to-end unit tests for the full ``canonicalize`` pipeline.

    Each kernel exercises a different stage path (lower-to-loops, fission,
    normalize, parallelize, fuse, conditional recombination, indirection)
    and asserts the canonicalized SDFG validates and is numerically
    identical to a deep-copied pre-canonicalization run. Includes an
    idempotence check (canonicalizing twice stays valid and value-
    preserving). Kernels use the dace Python frontend.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def elemwise(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + 1.0


@dace.program
def stencil1d(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[1:N - 1]:
        b[i] = a[i - 1] + a[i] + a[i + 1]


@dace.program
def two_independent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
        d[i] = c[i] * 3.0


@dace.program
def jacobi2d(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[1:N - 1, 1:M - 1]:
        b[i, j] = 0.25 * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


@dace.program
def gather(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i]]
        e[i] = c[idx[i]]


@dace.program
def strided(a: dace.float64[40], b: dace.float64[40]):
    for i in dace.map[3:31:4]:
        b[i] = a[i] + 5.0


@dace.program
def guarded(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    if active[0] > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0


def test_canonicalize_elementwise():
    n = 24
    a = np.random.rand(n)
    ref = np.zeros(n)
    copy.deepcopy(elemwise.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n)

    sdfg = elemwise.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, ref) and np.allclose(out, a * 2.0 + 1.0)


def test_canonicalize_stencil_1d():
    n = 32
    a = np.random.rand(n)
    ref = np.zeros(n)
    copy.deepcopy(stencil1d.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n)

    sdfg = stencil1d.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros(n)
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, ref)
    exp = np.zeros(n)
    exp[1:n - 1] = a[0:n - 2] + a[1:n - 1] + a[2:n]
    assert np.allclose(out, exp)


def test_canonicalize_two_independent_fission_fuse_roundtrip():
    n = 20
    a, c = np.random.rand(n), np.random.rand(n)
    ref_b, ref_d = np.zeros(n), np.zeros(n)
    copy.deepcopy(two_independent.to_sdfg(simplify=True))(a=a.copy(), b=ref_b, c=c.copy(), d=ref_d, N=n)

    sdfg = two_independent.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_d = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, c=c.copy(), d=out_d, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d)
    assert np.allclose(out_b, a + 1.0) and np.allclose(out_d, c * 3.0)


def test_canonicalize_jacobi_2d():
    n, m = 16, 12
    a = np.random.rand(n, m)
    ref = np.zeros((n, m))
    copy.deepcopy(jacobi2d.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)

    sdfg = jacobi2d.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros((n, m))
    sdfg(a=a.copy(), b=out, N=n, M=m)
    assert np.allclose(out, ref)
    exp = np.zeros((n, m))
    exp[1:n - 1, 1:m - 1] = 0.25 * (a[0:n - 2, 1:m - 1] + a[2:n, 1:m - 1] + a[1:n - 1, 0:m - 2] + a[1:n - 1, 2:m])
    assert np.allclose(out, exp)


def test_canonicalize_indirect_gather():
    n = 28
    a, c = np.random.rand(n), np.random.rand(n)
    idx = np.random.randint(0, n, size=n).astype(np.int32)
    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(gather.to_sdfg(simplify=True))(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = gather.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    assert np.allclose(out_b, a[idx]) and np.allclose(out_e, c[idx])


def test_canonicalize_strided_map_normalized():
    a = np.random.rand(40)
    ref = np.zeros(40)
    copy.deepcopy(strided.to_sdfg(simplify=True))(a=a.copy(), b=ref)

    sdfg = strided.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.zeros(40)
    sdfg(a=a.copy(), b=out)
    assert np.allclose(out, ref)
    exp = np.zeros(40)
    exp[3:31:4] = a[3:31:4] + 5.0
    assert np.allclose(out, exp)


@pytest.mark.parametrize('av', [1, 0])
def test_canonicalize_guarded_conditional(av):
    n = 18
    a = np.random.rand(n)
    ref = np.full(n, 7.0)
    copy.deepcopy(guarded.to_sdfg(simplify=True))(a=a.copy(), b=ref, active=np.array([av], np.int32), N=n)

    sdfg = guarded.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    out = np.full(n, 7.0)
    sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n)
    assert np.allclose(out, ref), f"mismatch active={av}"
    assert np.allclose(out, a * 2.0 if av > 0 else np.full(n, 7.0))


def test_canonicalize_is_idempotent():
    """Canonicalizing twice stays valid and value-preserving (fixed-point /
    deterministic-key sanity for the pipeline)."""
    n = 22
    a, c = np.random.rand(n), np.random.rand(n)
    ref_b, ref_d = np.zeros(n), np.zeros(n)
    copy.deepcopy(two_independent.to_sdfg(simplify=True))(a=a.copy(), b=ref_b, c=c.copy(), d=ref_d, N=n)

    sdfg = two_independent.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    canonicalize(sdfg, validate=True)  # second application must stay valid
    out_b, out_d = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, c=c.copy(), d=out_d, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
