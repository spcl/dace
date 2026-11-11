# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A 2D array with a unit dim, including a length-1 slice such as ``x[:, 0:1]``, transposes to the swapped shape as
in numpy instead of being squeezed and rejected as "not a matrix"."""

import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")


def test_unit_dim_transposes():

    @dace.program
    def col_transpose_matmul(x: dace.float64[N, 1], a: dace.float64[N, M]):
        return x.T @ a

    @dace.program
    def col_outer_product(x: dace.float64[N, 1]):
        return x @ x.T

    @dace.program
    def row_vector_transpose(x: dace.float64[1, N]):
        return x.T

    rng = np.random.default_rng(0)
    x, a, r = rng.random((5, 1)), rng.random((5, 4)), rng.random((1, 7))
    for got, ref in (
        (col_transpose_matmul(x.copy(), a.copy()), x.T @ a),
        (col_outer_product(x.copy()), x @ x.T),
        (row_vector_transpose(r.copy()), r.T),
    ):
        assert np.asarray(got).shape == ref.shape and np.allclose(got, ref)


def test_a_length1_slice_keeps_its_axis():
    """``pos[:, 0:1]`` is ``(N, 1)``: squeezed to ``(N,)``, ``.T`` is a no-op and the difference is silently wrong."""

    @dace.program
    def transposed_column_difference(pos: dace.float64[N, 3], dx: dace.float64[N, N]):
        dx[:] = pos[:, 0:1].T - pos[:, 0:1]

    sdfg = transposed_column_difference.to_sdfg(simplify=False)
    views = [desc for desc in sdfg.arrays.values() if isinstance(desc, dace.data.View)]
    assert views and all(len(view.shape) == 2 for view in views)
    pos = np.random.default_rng(0).random((6, 3))
    dx = np.zeros((6, 6))
    sdfg(pos=pos, dx=dx, N=6)
    assert np.allclose(dx, pos[:, 0:1].T - pos[:, 0:1])


def test_an_all_slice_unit_subscript_stays_2d():
    """``a[1:2, 1:2]`` is 2-D in numpy; its ranges look like scalar indices, so it is the case where tracking which
    dims came from a slice (``slice_dims``) is the only thing keeping it off the index path."""

    @dace.program
    def all_slice(a: dace.float64[3, 3], out: dace.float64[1, 1]):
        out[:] = a[1:2, 1:2]

    a = np.arange(9.0).reshape(3, 3).copy()
    out = np.zeros((1, 1))
    all_slice(a, out)
    assert np.allclose(out, a[1:2, 1:2])


if __name__ == "__main__":
    test_unit_dim_transposes()
    test_a_length1_slice_keeps_its_axis()
    test_an_all_slice_unit_subscript_stays_2d()
