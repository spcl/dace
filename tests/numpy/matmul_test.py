# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``numpy.matmul`` function spelling, which delegates to the ``@`` operator."""
import dace
from dace.frontend.python.common import DaceSyntaxError
import numpy as np
import pytest

M, K, N, B1, B2 = 5, 3, 4, 2, 3

rng = np.random.default_rng(42)


@dace.program
def matmul_2d_2d(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = np.matmul(A, B)


def test_matmul_2d_2d():
    A, B = rng.random((M, K)), rng.random((K, N))
    C = np.zeros((M, N))
    matmul_2d_2d(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


SM, SK, SN = (dace.symbol(s) for s in ('SM', 'SK', 'SN'))


@dace.program
def matmul_2d_2d_symbolic(A: dace.float64[SM, SK], B: dace.float64[SK, SN], C: dace.float64[SM, SN]):
    C[:] = np.matmul(A, B)


def test_matmul_2d_2d_symbolic():
    A, B = rng.random((M, K)), rng.random((K, N))
    C = np.zeros((M, N))
    matmul_2d_2d_symbolic(A, B, C, SM=M, SK=K, SN=N)
    assert np.allclose(C, np.matmul(A, B))


@dace.program
def matmul_2d_1d(A: dace.float64[M, K], x: dace.float64[K], y: dace.float64[M]):
    y[:] = np.matmul(A, x)


def test_matmul_2d_1d():
    A, x = rng.random((M, K)), rng.random(K)
    y = np.zeros(M)
    matmul_2d_1d(A, x, y)
    assert np.allclose(y, np.matmul(A, x))


@dace.program
def matmul_1d_2d(x: dace.float64[M], A: dace.float64[M, N], y: dace.float64[N]):
    y[:] = np.matmul(x, A)


def test_matmul_1d_2d():
    x, A = rng.random(M), rng.random((M, N))
    y = np.zeros(N)
    matmul_1d_2d(x, A, y)
    assert np.allclose(y, np.matmul(x, A))


@dace.program
def matmul_1d_1d(x: dace.float64[K], y: dace.float64[K], out: dace.float64[1]):
    out[:] = np.matmul(x, y)


def test_matmul_1d_1d():
    x, y = rng.random(K), rng.random(K)
    out = np.zeros(1)
    matmul_1d_1d(x, y, out)
    # NumPy demotes the 1-D x 1-D result to a scalar; DaCe keeps it as a length-1 array.
    assert np.allclose(out[0], np.matmul(x, y))


@dace.program
def matmul_3d_3d(A: dace.float64[B1, M, K], B: dace.float64[B1, K, N], C: dace.float64[B1, M, N]):
    C[:] = np.matmul(A, B)


def test_matmul_3d_3d():
    A, B = rng.random((B1, M, K)), rng.random((B1, K, N))
    C = np.zeros((B1, M, N))
    matmul_3d_3d(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


@dace.program
def matmul_3d_2d(A: dace.float64[B1, M, K], B: dace.float64[K, N], C: dace.float64[B1, M, N]):
    C[:] = np.matmul(A, B)


def test_matmul_3d_2d():
    A, B = rng.random((B1, M, K)), rng.random((K, N))
    C = np.zeros((B1, M, N))
    matmul_3d_2d(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


@dace.program
def matmul_2d_3d(A: dace.float64[M, K], B: dace.float64[B1, K, N], C: dace.float64[B1, M, N]):
    C[:] = np.matmul(A, B)


def test_matmul_2d_3d():
    A, B = rng.random((M, K)), rng.random((B1, K, N))
    C = np.zeros((B1, M, N))
    matmul_2d_3d(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


@dace.program
def matmul_4d_4d(A: dace.float64[B1, B2, M, K], B: dace.float64[B1, B2, K, N], C: dace.float64[B1, B2, M, N]):
    C[:] = np.matmul(A, B)


def test_matmul_4d_4d():
    A, B = rng.random((B1, B2, M, K)), rng.random((B1, B2, K, N))
    C = np.zeros((B1, B2, M, N))
    matmul_4d_4d(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


@dace.program
def matmul_in_map(A: dace.float64[B1, M, K], B: dace.float64[B1, K, N], C: dace.float64[B1, M, N]):
    for i in dace.map[0:B1]:
        C[i] = np.matmul(A[i], B[i])


def test_matmul_in_map():
    A, B = rng.random((B1, M, K)), rng.random((B1, K, N))
    C = np.zeros((B1, M, N))
    matmul_in_map(A, B, C)
    assert np.allclose(C, np.matmul(A, B))


def test_matmul_3d_1d_refused():
    """A 1-D operand against a rank >= 3 one needs NumPy's promote-batch-demote rule, unimplemented."""

    @dace.program
    def matmul_3d_1d(A: dace.float64[B1, M, K], x: dace.float64[K], C: dace.float64[B1, M]):
        C[:] = np.matmul(A, x)

    with pytest.raises(DaceSyntaxError, match='numpy.matmul of a 3-D and a 1-D operand is not supported'):
        matmul_3d_1d.to_sdfg()


def test_matmul_broadcast_batch():
    """NumPy broadcasts a length-1 batch dimension, and so does the pure lowering now.

    Only the fixed-stride BLAS expansions cannot express it; they refuse for themselves.
    """

    @dace.program
    def matmul_bcast(A: dace.float64[1, M, K], B: dace.float64[B1, K, N], C: dace.float64[B1, M, N]):
        C[:] = np.matmul(A, B)

    a = np.random.rand(1, M, K)
    b = np.random.rand(B1, K, N)
    c = np.zeros((B1, M, N))
    matmul_bcast(A=a, B=b, C=c)
    assert np.allclose(c, a @ b), f'max|diff| = {np.max(np.abs(c - a @ b))}'


def test_matmul_1d_3d_refused():

    @dace.program
    def matmul_1d_3d(x: dace.float64[K], B: dace.float64[B1, K, N], C: dace.float64[B1, N]):
        C[:] = np.matmul(x, B)

    with pytest.raises(DaceSyntaxError, match='numpy.matmul of a 1-D and a 3-D operand is not supported'):
        matmul_1d_3d.to_sdfg()


if __name__ == '__main__':
    test_matmul_2d_2d()
    test_matmul_2d_2d_symbolic()
    test_matmul_2d_1d()
    test_matmul_1d_2d()
    test_matmul_1d_1d()
    test_matmul_3d_3d()
    test_matmul_3d_2d()
    test_matmul_2d_3d()
    test_matmul_4d_4d()
    test_matmul_in_map()
    test_matmul_3d_1d_refused()
    test_matmul_broadcast_batch()
    test_matmul_1d_3d_refused()
