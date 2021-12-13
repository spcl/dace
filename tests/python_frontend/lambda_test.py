# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests lambda functions. """
import dace
import numpy as np
import pytest


def test_inline_lambda_tasklet():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a >> A[i]
                b << B[i]
                c << C[i]
                f = lambda a, b: a + b
                a = f(b, c)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


def test_inline_lambda_tasklet_2():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a >> A[i]
                b << B[i]
                c << C[i]
                a = (lambda a, b: a + b)(b, c)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


@pytest.mark.skip
def test_inline_lambda_scalar():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        f = lambda a, b: a + b
        for i in dace.map[0:20]:
            A[i] = f(B[i], C[i])

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


@pytest.mark.skip
def test_inline_lambda_scalar_2():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        for i in dace.map[0:20]:
            A[i] = (lambda a, b: a + b)(B[i], C[i])

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


@pytest.mark.skip
def test_inline_lambda_array():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        f = lambda a, b: a + b
        A[:] = f(B, C)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


def test_lambda_global():
    f = lambda a, b: a + b

    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        A[:] = f(B, C)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(A, B + C)


def test_lambda_call_jit():
    @dace.program
    def lamb(A, B, C, f):
        A[:] = f(B, C)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    f = lambda a, b: a + b
    lamb(A, B, C, f)
    assert np.allclose(A, B + C)


@pytest.mark.skip
def test_lambda_nested_call():
    @dace.program
    def lamb2(A, B, C, f):
        A[:] = f(B, C)

    @dace.program
    def lamb1(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        f = lambda a, b: a + b
        lamb2(A, B, C, f)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb1(A, B, C)
    assert np.allclose(A, B + C)


@pytest.mark.skip
def test_lambda_nested_call_inline():
    @dace.program
    def lamb2(A, B, C, f):
        A[:] = f(B, C)

    @dace.program
    def lamb1(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        lamb2(A, B, C, lambda a, b: a + b)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb1(A, B, C)
    assert np.allclose(A, B + C)


def test_lambda_with_closure():
    c = 4
    f = lambda a, b: a + b * c

    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        C[:] = f(A, B)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C)
    assert np.allclose(C, A + c * B)


def test_lambda_not_parsable():
    @dace.program
    def lamb(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20], f1,
             f2):
        B[:] = f1(A)
        C[:] = f2(A)

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    lamb(A, B, C, lambda a: 2 * a, lambda a: 4 * a)
    assert np.allclose(B, 2 * A)
    assert np.allclose(C, 4 * A)


if __name__ == '__main__':
    test_inline_lambda_tasklet()
    test_inline_lambda_tasklet_2()
    # test_inline_lambda_scalar()
    # test_inline_lambda_scalar_2()
    # test_inline_lambda_array()
    test_lambda_global()
    test_lambda_call_jit()
    # test_lambda_nested_call()
    # test_lambda_nested_call_inline()
    test_lambda_with_closure()
    test_lambda_not_parsable()
