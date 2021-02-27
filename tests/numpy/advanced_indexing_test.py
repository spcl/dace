# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" 
Tests for numpy advanced indexing syntax. See also:
https://numpy.org/devdocs/reference/arrays.indexing.html
"""
import dace
import numpy as np
import pytest

N = dace.symbol('N')
M = dace.symbol('M')


def test_flat():
    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A.flat

    A = np.random.rand(20, 30)
    res = indexing_test(A)
    assert np.allclose(A.flat, res)


@pytest.mark.skip
def test_flat_noncontiguous():
    @dace.program
    def indexing_test(A):
        return A.flat

    A = np.random.rand(20, 30).transpose()
    res = indexing_test(A)
    assert np.allclose(A.flat, res)


def test_ellipsis():
    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        return A[1:5, ..., 0]

    A = np.random.rand(5, 5, 5, 5, 5)
    res = indexing_test(A)
    assert np.allclose(A[1:5, ..., 0], res)


@pytest.mark.skip
def test_aug_implicit():
    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        A[:, 1:5][:, 0:2] += 5

    A = np.random.rand(5, 5, 5, 5, 5)
    regression = np.copy(A)
    regression[:, 1:5][:, 0:2] += 5
    indexing_test(A)
    assert np.allclose(A, regression)


@pytest.mark.skip
def test_ellipsis_aug():
    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        A[1:5, ..., 0] += 5

    A = np.random.rand(5, 5, 5, 5, 5)
    regression = np.copy(A)
    regression[1:5, ..., 0] += 5
    indexing_test(A)
    assert np.allclose(A, regression)


def test_newaxis():
    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A[:, np.newaxis, None, :]

    A = np.random.rand(20, 30)
    res = indexing_test(A)
    assert res.shape == (20, 1, 1, 30)
    assert np.allclose(A[:, np.newaxis, None, :], res)


def test_index_intarr_1d():
    @dace.program
    def indexing_test(A: dace.float64[N], indices: dace.int32[M]):
        return A[indices]

    A = np.random.rand(20)
    indices = [1, 10, 15]
    res = indexing_test(A, indices, M=3)
    assert np.allclose(A[indices], res)


def test_index_intarr_nd():
    @dace.program
    def indexing_test(A: dace.float64[4, 3], rows: dace.int64[2, 2],
                      columns: dace.int64[2, 2]):
        return A[rows, columns]

    A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                 dtype=np.float64)
    rows = np.array([[0, 0], [3, 3]], dtype=np.intp)
    columns = np.array([[0, 2], [0, 2]], dtype=np.intp)
    expected = A[rows, columns]
    res = indexing_test(A, rows, columns)
    assert np.allclose(expected, res)


def test_index_boolarr_rhs():
    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A[A > 15]

    A = np.ndarray((20, 30), dtype=np.float64)
    for i in range(20):
        A[i, :] = np.arange(0, 30)
    regression = A[A > 15]

    # Right-hand side boolean array indexing is unsupported
    with pytest.raises(IndexError):
        res = indexing_test(A)
        assert np.allclose(regression, res)


if __name__ == '__main__':
    test_flat()
    # test_flat_noncontiguous() # Skip due to broken strided copy
    test_ellipsis()
    # test_aug_implicit() # Skip due to duplicate make_slice
    # test_ellipsis_aug() # Skip due to duplicate make_slice
    test_newaxis()
    test_index_intarr_1d()
    test_index_intarr_nd()
    test_index_boolarr_rhs()
