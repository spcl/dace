# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests variants of the numpy split array manipulation.
"""
import dace
import numpy as np
from common import compare_numpy_output
import pytest

M = 9
N = 20
K = 30


@compare_numpy_output()
def test_split():
    arr = np.arange(M)
    a, b, c = np.split(arr, 3)
    return a + b + c


def test_uneven_split_fail():
    with pytest.raises(ValueError):

        @dace.program
        def tester():
            arr = np.arange(N)
            a, b, c = np.split(arr, 3)
            return a + b + c

        tester()


def test_symbolic_split_fail():
    with pytest.raises(ValueError):
        n = dace.symbol('n')

        @dace.program
        def tester():
            arr = np.arange(N)
            a, b, c = np.split(arr, n)
            return a + b + c

        tester()


def test_array_split_fail():
    with pytest.raises(ValueError):

        @dace.program
        def tester():
            arr = np.arange(N)
            split = np.arange(N)
            a, b, c = np.split(arr, split)
            return a + b + c

        tester()


@compare_numpy_output()
def test_array_split():
    arr = np.arange(N)
    a, b, c = np.array_split(arr, 3)
    return a, b, c


@compare_numpy_output()
def test_array_split_multidim():
    arr = np.ones((N, N))
    a, b, c = np.array_split(arr, 3, axis=1)
    return a, b, c


@compare_numpy_output()
def test_split_sequence():
    arr = np.arange(N)
    a, b = np.split(arr, [3])
    return a, b


@compare_numpy_output()
def test_split_sequence_2():
    arr = np.arange(M)
    a, b, c = np.split(arr, [3, 6])
    return a + b + c


def test_split_sequence_symbolic():
    n = dace.symbol('n')

    @dace.program
    def tester(arr: dace.float64[3 * n]):
        a, b, c = np.split(arr, [n, n + 2])
        return a, b, c

    nval = K // 3
    a = np.random.rand(K)
    ra, rb, rc = tester(a)
    assert ra.shape[0] == nval
    assert rb.shape[0] == 2
    assert rc.shape[0] == K - nval - 2
    ref = np.split(a, [nval, nval + 2])
    assert len(ref) == 3
    assert np.allclose(ra, ref[0])
    assert np.allclose(rb, ref[1])
    assert np.allclose(rc, ref[2])


@compare_numpy_output()
def test_vsplit():
    arr = np.ones((N, M))
    a, b = np.vsplit(arr, 2)
    return a, b


@compare_numpy_output()
def test_hsplit():
    arr = np.ones((M, N))
    a, b = np.hsplit(arr, 2)
    return a, b


@compare_numpy_output()
def test_dsplit_4d():
    arr = np.ones([N, M, K, K], dtype=np.float32)
    a, b, c = np.dsplit(arr, 3)
    return a, b, c


if __name__ == "__main__":
    test_split()
    test_uneven_split_fail()
    test_symbolic_split_fail()
    test_array_split_fail()
    test_array_split()
    test_array_split_multidim()
    test_split_sequence()
    test_split_sequence_2()
    test_split_sequence_symbolic()
    test_vsplit()
    test_hsplit()
    test_dsplit_4d()
