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


def test_var_args_jit():
    @dace.program
    def arg_jit(*args):
        return args[0] + args[1]

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = arg_jit(A, B)
    assert np.allclose(C, A + B)


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


def test_var_kwargs_jit():
    @dace.program
    def kwarg_jit(**kwargs):
        return kwargs['A'] + kwargs['B']

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = kwarg_jit(A=A, B=B)
    assert np.allclose(C, A + B)


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


def test_constant_argument_simple():
    @dace.program
    def const_prog(cst: dace.constant, B: dace.float64[20]):
        B[:] = cst

    # Test program
    A = np.random.rand(20)
    cst = 4.0
    const_prog(cst, A)
    assert np.allclose(A, cst)

    # Test code for folding
    code = const_prog.to_sdfg(cst).generate_code()[0].clean_code
    assert 'cst' not in code


def test_constant_argument_default():
    @dace.program
    def const_prog(B: dace.float64[20], cst: dace.constant = 7):
        B[:] = cst

    # Test program
    A = np.random.rand(20)
    const_prog(A)
    assert np.allclose(A, 7)

    # Forcefully clear cache to recompile
    const_prog.clear_cache()

    # Test program
    A = np.random.rand(20)
    const_prog(A, cst=4)
    assert np.allclose(A, 4)

    # Test code for folding
    code = const_prog.to_sdfg().generate_code()[0].clean_code
    assert 'cst' not in code


def test_constant_argument_object():
    """
    Tests nested functions with constant parameters passed in as arguments.
    """
    class MyConfiguration:
        def __init__(self, parameter):
            self.p = parameter * 2
            self.q = parameter * 4

        @staticmethod
        def get_random_number():
            return 4

    @dace.program
    def nested_func(cfg: dace.constant, A: dace.float64[20]):
        return A[cfg.p]

    @dace.program
    def constant_parameter(cfg: dace.constant, cfg2: dace.constant,
                           A: dace.float64[20]):
        A[cfg.q] = nested_func(cfg, A)
        A[MyConfiguration.get_random_number()] = nested_func(cfg2, A)

    cfg1 = MyConfiguration(3)
    cfg2 = MyConfiguration(4)
    A = np.random.rand(20)
    reg_A = np.copy(A)
    reg_A[12] = reg_A[6]
    reg_A[4] = reg_A[8]

    constant_parameter(cfg1, cfg2, A)
    assert np.allclose(A, reg_A)


if __name__ == '__main__':
    test_kwargs()
    test_kwargs_jit()
    test_kwargs_with_default()
    test_var_args_jit()
    test_var_args_aot()
    test_var_args_empty()
    test_var_kwargs_jit()
    test_var_kwargs_aot()
    test_none_arrays()
    test_none_arrays_jit()
    test_optional_argument_jit()
    test_optional_argument_jit_kwarg()
    test_optional_argument()
    test_constant_argument_simple()
    test_constant_argument_default()
    test_constant_argument_object()
