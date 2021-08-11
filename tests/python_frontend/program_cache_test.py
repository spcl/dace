# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_cache_same_args():
    """ 
    Tests that two subsequent calls to a program does not trigger
    recompilation.
    """
    @dace.program
    def test(x):
        return x * x

    test(5)
    assert len(test._cache.cache) == 1
    test(5)
    assert len(test._cache.cache) == 1


def test_cache_different_args():
    """ 
    Tests that two subsequent calls to a program with different shapes does
    trigger recompilation.
    """
    @dace.program
    def test(x):
        return x * x

    a = np.random.rand(2)
    b = np.random.rand(3)
    ra = test(a)
    assert len(test._cache.cache) == 1
    rb = test(b)
    assert len(test._cache.cache) == 2

    assert np.allclose(a * a, ra)
    assert np.allclose(b * b, rb)


def test_cache_return_values():
    @dace.program
    def test(x):
        return x * x

    a = test(5)
    b = test(6)

    assert a == 25 and b == 36


def test_cache_argument_names():
    @dace.program
    def test(C: dace.float32[20], A: dace.float64[30]):
        A *= 5
        C *= 2

    sdfg = test.to_sdfg()
    a = np.random.rand(20).astype(np.float32)
    c = np.random.rand(30)
    rega = a * 2
    regc = c * 5
    sdfg(a, c)

    assert np.allclose(a, rega) and np.allclose(c, regc)


if __name__ == '__main__':
    test_cache_same_args()
    test_cache_different_args()
    test_cache_return_values()
    test_cache_argument_names()
