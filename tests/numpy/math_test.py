# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from common import compare_numpy_output
import math
from numpy import exp, sin, cos, sqrt, log, log10, conj, real, imag
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


@compare_numpy_output(non_zero=True, positive=True)
def test_log10(A: dace.complex64[M, N]):
    return log10(A)


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

    @pytest.mark.parametrize("mathfunc", [abs, np.abs, np.sqrt])
    @pytest.mark.parametrize("arg", [0.7, np.random.randn(5, 5)])
    def test_func(self, mathfunc, arg):

        @dace.program
        def func(arg):
            return mathfunc(arg)

        res = func(arg)
        assert np.allclose(mathfunc(arg), res, equal_nan=True)

    @pytest.mark.parametrize("mathfunc", [math.floor, math.ceil])
    def test_func_scalar(self, mathfunc):
        self.test_func(mathfunc, 0.7)

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


@compare_numpy_output(check_dtype=True)
def test_abs_complex(A: dace.complex64[M, N]):
    # abs() of a complex value is its (real-valued) magnitude; the result dtype must be real.
    return abs(A)


@compare_numpy_output(check_dtype=True)
def test_np_abs_complex(A: dace.complex128[M, N]):
    return np.abs(A)


def test_abs_complex_comparison():
    # The original bug: abs(z) stayed typed complex, so abs(z) < 1.0 failed to compile
    # (no operator< on std::complex). This program must both build and match numpy.
    @dace.program
    def func(A: dace.complex128[M, N]):
        return abs(A) < 1.0

    A = (np.random.rand(M, N) + 1j * np.random.rand(M, N)).astype(np.complex128)
    assert np.array_equal(func(A), np.abs(A) < 1.0)


@compare_numpy_output(check_dtype=True)
def test_pow_int_exponent_complex(A: dace.complex128[M, N]):
    # base ** int must stay an exact integer power (not routed through exp/log via an
    # up-cast exponent), preserving both the complex result dtype and bit-exactness.
    return A**2


@compare_numpy_output(check_dtype=True)
def test_pow_int_exponent_float(A: dace.float64[M, N]):
    return A**3


def test_pow_int_exponent_bit_exact():
    # Guard the bit-exactness the fix restores: squaring must equal an explicit multiply.
    @dace.program
    def func(A: dace.complex128[M, N]):
        return A**2

    A = (np.random.rand(M, N) + 1j * np.random.rand(M, N)).astype(np.complex128)
    assert np.array_equal(func(A), A * A)


def test_scalarret_cond_1():

    @dace.program
    def func(arg: dace.float64):
        n = math.floor(1.0 + arg)
        if n > 1.0:
            return 0.0
        else:
            return 1.0

    res = func(3.7)
    assert res == 0.0


def test_scalarret_cond_2():

    @dace.program
    def func():
        n = math.floor(1.0 + 2.0)
        if n > 1.0:
            return 0.0
        else:
            return 1.0

    res = func()
    assert res == 0.0


def test_scalarret_cond_3():

    @dace.program
    def func():
        cval = 2.5
        n = math.floor(1.0 + cval)
        if n > 1.0:
            return 0.0
        else:
            return 1.0

    res = func()
    assert res == 0.0


if __name__ == '__main__':
    test_exponent()
    test_sine()
    test_cosine()
    test_square_root()
    test_logarithm()
    test_log10()
    test_conjugate()
    test_real_part()
    test_imag_part()
    test_exponent_m()
    test_exponent_t()
    TestMathFuncs().test_func(np.abs, 0.7)
    TestMathFuncs().test_func(abs, 0.7)
    TestMathFuncs().test_func_scalar(math.floor)
    test_abs_complex()
    test_np_abs_complex()
    test_abs_complex_comparison()
    test_pow_int_exponent_complex()
    test_pow_int_exponent_float()
    test_pow_int_exponent_bit_exact()
    TestMathFuncs().test_func_scalar(math.ceil)
    test_scalarret_cond_1()
    test_scalarret_cond_2()
    test_scalarret_cond_3()
