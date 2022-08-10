# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace

N, M = dace.symbol('N'), dace.symbol('M')


def test_len():

    @dace.program
    def tester(A: dace.float64[N, M]):
        return len(A)

    a = np.random.rand(20, 30)
    assert np.allclose(tester(a), len(a))

def test_len_constant():

    @dace.program
    def tester(A: dace.float64[N, M]):
        b = np.array([1., 2., 3.])
        return len(b)

    a = np.random.rand(20, 30)
    assert np.allclose(tester(a), 3)

def test_sum():

    @dace.program
    def tester(A: dace.float64[N, M]):
        return sum(A)

    a = np.random.rand(20, 30)
    assert np.allclose(tester(a), sum(a))


if __name__ == '__main__':
    test_len()
    test_len_constant()
    test_sum()
