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


if __name__ == '__main__':
    test_flat()
    # test_flat_noncontiguous() # Skip due to broken strided copy
    test_ellipsis()
