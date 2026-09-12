# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for issues/03-numpy-empty-dtype.md and issues/04-arrayview-copy.md.

Bug 3: numpy.empty's replacement declared dtype as a required positional argument, unlike
numpy.zeros/numpy.ones/numpy.empty_like and unlike numpy.empty itself, whose dtype defaults to
float64.

Bug 4: .copy() on a sliced array (an ArrayView) raised a bare NotImplementedError, because the
transient-creation dispatch it landed on keys on the exact data descriptor type and does not
recognize ArrayView, a concrete subclass of both Array and View.
"""
import numpy as np

import dace


def test_numpy_empty_default_dtype_matches_numpy():

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty(10)
        y[:] = 1.0
        x[:] = y

    x = np.zeros(10)
    f(x)
    assert np.array_equal(x, np.ones(10))

    sdfg = f.to_sdfg(simplify=False)
    transients = [d.dtype for d in sdfg.arrays.values() if d.transient]
    assert any(dt == dace.float64 for dt in transients)


def test_numpy_empty_explicit_dtype_still_honoured():

    @dace.program
    def f(x: dace.int32[10]):
        y = np.empty(10, dtype=np.int32)
        y[:] = 3
        x[:] = y

    x = np.zeros(10, dtype=np.int32)
    f(x)
    assert np.array_equal(x, np.full(10, 3, dtype=np.int32))

    sdfg = f.to_sdfg(simplify=False)
    transients = [d.dtype for d in sdfg.arrays.values() if d.transient]
    assert any(dt == dace.int32 for dt in transients)


def test_numpy_empty_like_unaffected():

    @dace.program
    def f(x: dace.float64[10]):
        y = np.empty_like(x)
        y[:] = 2.0
        x[:] = y

    x = np.zeros(10)
    f(x)
    assert np.array_equal(x, np.full(10, 2.0))


def test_copy_of_column_slice_matches_numpy():

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        out[:] = path[:, 1].copy()

    rng = np.random.default_rng(0)
    path = rng.random((10, 10))
    out = np.zeros(10)
    f(path, out)
    assert np.allclose(out, path[:, 1])


def test_copy_of_row_slice_matches_numpy():

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        out[:] = path[1, :].copy()

    rng = np.random.default_rng(0)
    path = rng.random((10, 10))
    out = np.zeros(10)
    f(path, out)
    assert np.allclose(out, path[1, :])


def test_numpy_copy_function_on_slice_matches_numpy():

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10]):
        out[:] = np.copy(path[:, 1])

    rng = np.random.default_rng(0)
    path = rng.random((10, 10))
    out = np.zeros(10)
    f(path, out)
    assert np.allclose(out, path[:, 1])


def test_copy_of_whole_array_still_works():
    """Control: copying a plain (non-view) array must remain unaffected."""

    @dace.program
    def f(path: dace.float64[10, 10], out: dace.float64[10, 10]):
        out[:] = path.copy()

    rng = np.random.default_rng(0)
    path = rng.random((10, 10))
    out = np.zeros((10, 10))
    f(path, out)
    assert np.allclose(out, path)


if __name__ == '__main__':
    test_numpy_empty_default_dtype_matches_numpy()
    test_numpy_empty_explicit_dtype_still_honoured()
    test_numpy_empty_like_unaffected()
    test_copy_of_column_slice_matches_numpy()
    test_copy_of_row_slice_matches_numpy()
    test_numpy_copy_function_on_slice_matches_numpy()
    test_copy_of_whole_array_still_works()
