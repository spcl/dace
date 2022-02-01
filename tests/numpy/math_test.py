# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from common import compare_numpy_output
import math
from numpy import exp, sin, cos, sqrt, log, conj, real, imag
import pytest

M, N = 24, 24


@compare_numpy_output()
def test_exponent(A: dace.complex64[M, N]):
    return exp(A)


@compare_numpy_output()
def test_sine(A: dace.complex64[M, N]):
    return sin(A)


@compare_numpy_output()
def test_cosine(A: dace.complex64[M, N]):
    return cos(A)


@compare_numpy_output(non_zero=True, positive=True)
def test_square_root(A: dace.complex64[M, N]):
    return sqrt(A)


@compare_numpy_output(non_zero=True, positive=True)
def test_logarithm(A: dace.complex64[M, N]):
    return log(A)


@compare_numpy_output()
def test_conjugate(A: dace.complex64[M, N]):
    return conj(A)


@compare_numpy_output()
def test_real_part(A: dace.complex64[M, N]):
    return real(A)


@compare_numpy_output()
def test_imag_part(A: dace.complex64[M, N]):
    return imag(A)


@dace.program
def exponent_m(A: dace.complex64[M, N]):
    B = np.ndarray([M, N], dace.complex64)
    for i in dace.map[0:M]:
        B[i] = exp(A[i])
    return B


def test_exponent_m():
    A = np.random.rand(M, N).astype(np.float32) + 1j * np.random.rand(M, N).astype(np.float32)
    B = exponent_m(A)
    assert np.allclose(B, np.exp(A))


@dace.program
def exponent_t(A: dace.complex64[M, N]):
    B = np.ndarray([M, N], dace.complex64)
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = exp(A[i, j])
    return B


def test_exponent_t():
    A = np.random.rand(M, N).astype(np.float32) + 1j * np.random.rand(M, N).astype(np.float32)
    B = exponent_t(A)
    assert np.allclose(B, np.exp(A))


class TestMathFuncs:
    @pytest.mark.parametrize("mathfunc", [abs, np.abs, np.sqrt, math.floor, math.ceil])
    @pytest.mark.parametrize("arg", [0.7, np.random.randn(5, 5)])
    def test_func(self, mathfunc, arg):
        @dace.program
        def func(arg):
            return mathfunc(arg)

        res = func(arg)
        assert np.allclose(mathfunc(arg), res, equal_nan=True)

    @pytest.mark.parametrize("mathfunc", [min, max])
    def test_func2_scalar(self, mathfunc):
        @dace.program
        def func(arg1, arg2):
            return mathfunc(arg1, arg2)

        res = func(0.7, 0.5)
        assert np.allclose(mathfunc(0.7, 0.5), res)

    @pytest.mark.parametrize("mathfunc", [np.minimum, np.maximum])
    def test_func2_arr(self, mathfunc):
        @dace.program
        def func(arg1, arg2):
            return mathfunc(arg1, arg2)

        arg1 = np.random.randn(5, 5)
        arg2 = np.random.randn(5, 5)
        res = func(arg1, arg2)
        assert np.allclose(mathfunc(arg1, arg2), res)


if __name__ == '__main__':
    test_exponent()
    test_sine()
    test_cosine()
    test_square_root()
    test_logarithm()
    test_conjugate()
    test_real_part()
    test_imag_part()
    test_exponent_m()
    test_exponent_t()
    TestMathFuncs().test_func(np.abs, 0.7)
    TestMathFuncs().test_func(abs, 0.7)
    TestMathFuncs().test_func(math.floor, 0.7)
    TestMathFuncs().test_func(math.ceil, 0.7)
