"""Strict numpy slice/transpose semantics for unit dims: ``X[:, 0:1]`` keeps the axis (``(N, 1)``)
while ``X[:, 0]`` squeezes it (``(N,)``), and a ``(N, 1)`` array transposes to ``(1, N)`` -- the
DaCe frontend used to squeeze the unit dim and reject ``(N, 1).T`` as "not a matrix"."""
import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def col_transpose_matmul(x: dace.float64[N, 1], a: dace.float64[N, M]):
    return x.T @ a  # (1, N) @ (N, M) -> (1, M)


def test_column_vector_transpose_matmul():
    n, m = 5, 4
    rng = np.random.default_rng(0)
    x, a = rng.random((n, 1)), rng.random((n, m))
    got = np.asarray(col_transpose_matmul(x.copy(), a.copy()))
    ref = x.T @ a
    assert got.shape == ref.shape == (1, m)
    assert np.allclose(got.reshape(ref.shape), ref)


@dace.program
def slice_keeps_axis(x: dace.float64[N, 3]):
    col = x[:, 1:2]  # a length-1 slice keeps the axis: (N, 1)
    return col.T @ x  # (1, N) @ (N, 3) -> (1, 3); only valid if col stayed 2D


def test_length_one_slice_keeps_axis():
    n = 6
    rng = np.random.default_rng(1)
    x = rng.random((n, 3))
    got = np.asarray(slice_keeps_axis(x.copy()))
    ref = x[:, 1:2].T @ x
    assert got.shape == ref.shape == (1, 3)
    assert np.allclose(got.reshape(ref.shape), ref)


@dace.program
def slice_outer_product(x: dace.float64[N, 3]):
    col = x[:, 1:2]  # (N, 1)
    return col @ col.T  # (N, 1) @ (1, N) -> (N, N) outer product


def test_length_one_slice_outer_product():
    n = 6
    rng = np.random.default_rng(3)
    x = rng.random((n, 3))
    got = np.asarray(slice_outer_product(x.copy()))
    ref = x[:, 1:2] @ x[:, 1:2].T
    assert got.shape == ref.shape == (n, n)
    assert np.allclose(got, ref)


@dace.program
def index_squeezes_axis(x: dace.float64[N, 3]):
    col = x[:, 1]  # an integer index squeezes the axis: (N,)
    return np.sum(col)


def test_integer_index_squeezes_axis():
    n = 6
    rng = np.random.default_rng(2)
    x = rng.random((n, 3))
    got = np.asarray(index_squeezes_axis(x.copy()))
    assert np.allclose(got, np.sum(x[:, 1]))


if __name__ == "__main__":
    test_column_vector_transpose_matmul()
    test_length_one_slice_keeps_axis()
    test_length_one_slice_outer_product()
    test_integer_index_squeezes_axis()
    print("OK")
