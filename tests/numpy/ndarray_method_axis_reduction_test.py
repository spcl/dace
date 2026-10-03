# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``a.sum(axis=...)`` and its sibling methods reduce the axes asked for, like the ``np.sum`` spelling."""
import numpy as np
import pytest

import dace

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def row_sums(a: dace.float64[N, M], out: dace.float64[N]):
    out[:] = a.sum(axis=1)


@dace.program
def row_means(a: dace.float64[N, M], out: dace.float64[N]):
    out[:] = a.mean(axis=1)


@dace.program
def column_products(a: dace.float64[N, M], out: dace.float64[M]):
    out[:] = a.prod(axis=0)


@dace.program
def row_sums_kept(a: dace.float64[N, M], out: dace.float64[N, 1]):
    out[:] = a.sum(axis=1, keepdims=True)


@dace.program
def rows_with_a_large_entry(a: dace.float64[N, M], out: dace.bool_[N]):
    out[:] = (a > 0.9).any(axis=1)


@dace.program
def total(a: dace.float64[N, M], out: dace.float64[1]):
    out[0] = a.sum()


@pytest.mark.parametrize('program, reference, out_shape', [
    (row_sums, lambda a: a.sum(axis=1), (5, )),
    (row_means, lambda a: a.mean(axis=1), (5, )),
    (column_products, lambda a: a.prod(axis=0), (7, )),
    (row_sums_kept, lambda a: a.sum(axis=1, keepdims=True), (5, 1)),
    (rows_with_a_large_entry, lambda a: (a > 0.9).any(axis=1), (5, )),
    (total, lambda a: np.array([a.sum()]), (1, )),
])
def test_a_method_reduction_matches_numpy(program, reference, out_shape):
    a = np.random.default_rng(0).random((5, 7))
    want = reference(a)
    out = np.zeros(out_shape, dtype=want.dtype)
    program(a=a, out=out, N=5, M=7)
    np.testing.assert_allclose(out, want)
