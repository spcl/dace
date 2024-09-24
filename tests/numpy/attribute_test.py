# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def doublefor_jit(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] *= 5


@dace.program
def doublefor_aot(A: dace.float64[N, N]):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] *= A.shape[0]


def test_attribute_in_ranged_loop():
    a = np.random.rand(20, 20)
    regression = a * 5
    doublefor_jit(a)
    assert np.allclose(a, regression)


def test_attribute_in_ranged_loop_symbolic():
    a = np.random.rand(20, 20)
    regression = a * 20
    doublefor_aot(a)
    assert np.allclose(a, regression)


def test_attribute_new_state():

    N, F_in, F_out, heads = 2, 3, 4, 5

    @dace.program
    def fn(a: dace.float64[N, F_in], b: dace.float64[N, heads, F_out], c: dace.float64[heads * F_out, F_in]):
        tmp = a.T @ np.reshape(b, (N, heads * F_out))
        c[:] = tmp.T

    rng = np.random.default_rng(42)

    a = rng.random((N, F_in))
    b = rng.random((N, heads, F_out))
    c_expected = np.zeros((heads * F_out, F_in))
    c = np.zeros((heads * F_out, F_in))

    fn.f(a, b, c_expected)
    fn(a, b, c)
    assert np.allclose(c, c_expected)


if __name__ == '__main__':
    test_attribute_in_ranged_loop()
    test_attribute_in_ranged_loop_symbolic()
    test_attribute_new_state()
