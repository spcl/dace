# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output
import pytest

M = 10
N = 20
K = 30


@compare_numpy_output()
def test_concatenate():
    a = np.zeros([N, N], dtype=np.float32)
    b = np.ones([N, 1], dtype=np.float32)
    return np.concatenate((a, b), axis=-1)


@compare_numpy_output()
def test_concatenate_four():
    a = np.zeros([N, N], dtype=np.float32)
    b = np.ones([N, 1], dtype=np.float32)
    c = np.full([N, M], 2.0, dtype=np.float32)
    return np.concatenate((a, b, c, a), axis=-1)


@compare_numpy_output()
def test_concatenate_out():
    a = np.zeros([N, N], dtype=np.float32)
    b = np.ones([M, N], dtype=np.float32)
    c = np.full([N + M, N], -1, dtype=np.float32)
    np.concatenate([a, b], out=c)
    return c + 1


def test_concatenate_symbolic():
    n = dace.symbol('n')
    m = dace.symbol('m')
    k = dace.symbol('k')

    @dace.program
    def tester(a: dace.float64[k, m], b: dace.float64[k, n]):
        return np.concatenate((a, b), axis=1)

    aa = np.random.rand(10, 4)
    bb = np.random.rand(10, 5)
    cc = tester(aa, bb)
    assert tuple(cc.shape) == (10, 9)
    assert np.allclose(np.concatenate((aa, bb), axis=1), cc)


def test_concatenate_fail():
    with pytest.raises(ValueError):

        @dace.program
        def tester(a: dace.float64[K, M], b: dace.float64[N, K]):
            return np.concatenate((a, b), axis=1)

        aa = np.random.rand(K, M)
        bb = np.random.rand(N, K)
        tester(aa, bb)


@compare_numpy_output()
def test_concatenate_flatten():
    a = np.zeros([1, 2, 3], dtype=np.float32)
    b = np.ones([4, 5, 6], dtype=np.float32)
    return np.concatenate([a, b], axis=None)


@compare_numpy_output()
def test_stack():
    a = np.zeros([N, M, K], dtype=np.float32)
    b = np.ones([N, M, K], dtype=np.float32)
    return np.stack((a, b), axis=-1)


@compare_numpy_output()
def test_vstack():
    a = np.zeros([N, M], dtype=np.float32)
    b = np.ones([N, M], dtype=np.float32)
    return np.vstack((a, b))


@compare_numpy_output()
def test_vstack_1d():
    a = np.zeros([N], dtype=np.float32)
    b = np.ones([N], dtype=np.float32)
    return np.vstack((a, b))


@compare_numpy_output()
def test_hstack():
    a = np.zeros([N, M], dtype=np.float32)
    b = np.ones([N, M], dtype=np.float32)
    return np.hstack((a, b))


@compare_numpy_output()
def test_hstack_1d():
    a = np.zeros([N], dtype=np.float32)
    b = np.ones([N], dtype=np.float32)
    return np.hstack((a, b))


@compare_numpy_output()
def test_dstack():
    a = np.zeros([N, M, K], dtype=np.float32)
    b = np.ones([N, M, K], dtype=np.float32)
    return np.dstack((a, b))


@compare_numpy_output()
def test_dstack_4d():
    a = np.zeros([N, M, K, K], dtype=np.float32)
    b = np.ones([N, M, K, K], dtype=np.float32)
    return np.dstack((a, b))


if __name__ == "__main__":
    test_concatenate()
    test_concatenate_four()
    test_concatenate_out()
    test_concatenate_symbolic()
    test_concatenate_fail()
    test_concatenate_flatten()
    test_stack()
    test_vstack()
    test_vstack_1d()
    test_hstack()
    test_hstack_1d()
    test_dstack()
    test_dstack_4d()
