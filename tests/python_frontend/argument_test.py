# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import numpy as np

N = dace.symbol('N')


@dace.program
def imgcpy(img1: dace.float64[N, N], img2: dace.float64[N, N], coefficient: dace.float64):
    img1[:, :] = img2[:, :] * coefficient


def test_extra_args():
    with pytest.raises(TypeError):
        imgcpy([[1, 2], [3, 4]], [[4, 3], [2, 1]], 0.0, 1.0)


def test_missing_arguments_regression():

    def nester(a, b, T):
        for i, j in dace.map[0:20, 0:20]:
            start = 0
            end = min(T, 6)

            elem: dace.float64 = 0
            for ii in range(start, end):
                if ii % 2 == 0:
                    elem += b[ii]

            a[j, i] = elem

    @dace.program
    def tester(x: dace.float64[20, 20]):
        gdx = np.ones((10, ), dace.float64)
        for T in range(2):
            nester(x, gdx, T)

    tester.to_sdfg().compile()


def test_missing_arguments_2_regression():

    @dace.program
    def tester(x: dace.float64[20]):
        x[:] = 0

    with pytest.raises(KeyError):
        tester()


def test_nested_call_with_too_small_argument():

    @dace.program
    def callee(a: dace.float64[10]):
        a += 1

    @dace.program
    def caller(a: dace.float64[5, 10]):
        for i in range(5):
            callee(a[:, i])

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match='declared with 10 elements'):
        caller.to_sdfg(simplify=False)


def test_nested_call_with_reshaped_argument():

    @dace.program
    def callee(a: dace.float64[20]):
        a += 1

    @dace.program
    def caller(a: dace.float64[5, 4]):
        callee(a.reshape((20, )))

    A = np.random.rand(5, 4)
    expected = A + 1
    caller(A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_extra_args()
    test_missing_arguments_regression()
    test_missing_arguments_2_regression()
    test_nested_call_with_too_small_argument()
    test_nested_call_with_reshaped_argument()
