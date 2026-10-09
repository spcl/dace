# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A 2D array with a unit dim (``(N, 1)`` / ``(1, N)``) must transpose to the swapped shape; the
frontend used to squeeze it and reject ``(N, 1).T`` as "not a matrix". An integer index, by
contrast, squeezes its axis (``x[:, 1]`` is ``(N,)``) per numpy."""

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
def transposed_column_difference(pos: dace.float64[N, 3], dx: dace.float64[N, N]):
    dx[:] = pos[:, 0:1].T - pos[:, 0:1]


def test_a_length1_slice_keeps_its_axis():
    """``pos[:, 0:1]`` is ``(N, 1)`` as in numpy: squeezing it to ``(N,)`` makes ``.T`` a no-op and the difference a
    row-wise broadcast, silently wrong."""
    sdfg = transposed_column_difference.to_sdfg(simplify=False)
    views = [desc for desc in sdfg.arrays.values() if isinstance(desc, dace.data.View)]
    assert views and all(len(view.shape) == 2 for view in views), [str(view.shape) for view in views]
    pos = np.random.default_rng(0).random((6, 3))
    dx = np.zeros((6, 6))
    transposed_column_difference(pos=pos, dx=dx, N=6)
    assert np.allclose(dx, pos[:, 0:1].T - pos[:, 0:1])


if __name__ == "__main__":
    test_column_vector_transpose_matmul()
    test_a_length1_slice_keeps_its_axis()
