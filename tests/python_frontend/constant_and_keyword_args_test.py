# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests constants, optional, and keyword arguments. """
import dace
import numpy as np
import pytest


def test_kwargs():
    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20]):
        A[:] = kw + 1

    A = np.random.rand(20)
    kw = np.random.rand(20)
    kwarg(A, kw=kw)
    assert np.allclose(A, kw + 1)


def test_kwargs_jit():
    @dace.program
    def kwarg(A, kw):
        A[:] = kw + 1

    A = np.random.rand(20)
    kw = np.random.rand(20)
    kwarg(A, kw=kw)
    assert np.allclose(A, kw + 1)


def test_kwargs_with_default():
    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20] = np.ones([20])):
        A[:] = kw + 1

    # Call without argument
    A = np.random.rand(20)
    kwarg(A)
    assert np.allclose(A, 2.0)

    # Call with argument
    kw = np.random.rand(20)
    kwarg(A, kw)
    assert np.allclose(A, kw + 1)


def test_var_args_aot():
    # This test is supposed to be unsupported
    with pytest.raises(SyntaxError):

        @dace.program
        def arg_aot(*args: dace.float64[20]):
            return args[0] + args[1]

        arg_aot.compile()


def test_var_args_empty():
    # This test is supposed to be unsupported
    with pytest.raises(SyntaxError):

        @dace.program
        def arg_aot(*args):
            return np.zeros([20])

        arg_aot.compile()


def test_var_kwargs_aot():
    # This test is supposed to be unsupported
    with pytest.raises(SyntaxError):

        @dace.program
        def kwarg_aot(**kwargs: dace.float64[20]):
            return kwargs['A'] + kwargs['B']

        kwarg_aot.compile()


def test_none_arrays():
    @dace.program
    def myprog(A: dace.float64[20], B: dace.float64[20]):
        result = np.zeros([20], dtype=dace.float64)
        if B is None:
            if A is not None:
                result[:] = A
            else:
                result[:] = 1
        else:
            result[:] = B
        return result

    # Tests
    A = np.random.rand(20)
    B = np.random.rand(20)
    assert np.allclose(myprog(A, B), B)
    assert np.allclose(myprog(A, None), A)
    assert np.allclose(myprog(None, None), 1)


def test_none_arrays_jit():
    @dace.program
    def myprog_jit(A, B):
        if B is None:
            if A is not None:
                return A
            else:
                return np.ones([20], np.float64)
        else:
            return B

    # Tests
    A = np.random.rand(20)
    B = np.random.rand(20)
    assert np.allclose(myprog_jit(A, B), B)
    assert np.allclose(myprog_jit(A, None), A)
    assert np.allclose(myprog_jit(None, None), 1)


def test_optional_argument_jit():
    @dace.program
    def linear(x, w, bias):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return x @ w.T + bias
        else:
            return x @ w.T

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w, None), linear(x, w, None))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


def test_optional_argument_jit_kwarg():
    @dace.program
    def linear(x, w, bias=None):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return np.dot(x, w.T) + bias
        else:
            return np.dot(x, w.T)

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w), linear(x, w))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


def test_optional_argument():
    @dace.program
    def linear(x: dace.float64[13, 14],
               w: dace.float64[10, 14],
               bias: dace.float64[10] = None):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return np.dot(x, w.T) + bias
        else:
            return np.dot(x, w.T)

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w), linear(x, w))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


if __name__ == '__main__':
    test_kwargs()
    test_kwargs_jit()
    test_kwargs_with_default()
    test_var_args_aot()
    test_var_args_empty()
    test_var_kwargs_aot()
    test_none_arrays()
    test_none_arrays_jit()
    test_optional_argument_jit()
    test_optional_argument_jit_kwarg()
    test_optional_argument()
