# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A 2D array with a unit dim (``(N, 1)`` / ``(1, N)``) must transpose to the swapped shape; the
frontend used to squeeze it and reject ``(N, 1).T`` as "not a matrix". An integer index, by
contrast, squeezes its axis (``x[:, 1]`` is ``(N,)``) per numpy."""

import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def col_outer_product(x: dace.float64[N, 1]):
    return x @ x.T  # (N, 1) @ (1, N) -> (N, N) outer product


def test_column_vector_outer_product():
    n = 6
    rng = np.random.default_rng(1)
    x = rng.random((n, 1))
    got = np.asarray(col_outer_product(x.copy()))
    ref = x @ x.T
    assert got.shape == ref.shape == (n, n)
    assert np.allclose(got, ref)


@dace.program
def row_vector_transpose(x: dace.float64[1, N]):
    return x.T  # (1, N) -> (N, 1)


def test_row_vector_transpose():
    n = 7
    rng = np.random.default_rng(2)
    x = rng.random((1, n))
    got = np.asarray(row_vector_transpose(x.copy()))
    assert got.shape == (n, 1)
    assert np.allclose(got.reshape(n, 1), x.T)


if __name__ == "__main__":
    test_column_vector_outer_product()
    test_row_vector_transpose()
    print("OK")
