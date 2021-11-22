# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests assignments in conditions. """
import numpy as np
import dace
import pytest


def test_none_or_field_call():
    @dace.program
    def func2(A, B):
        A[...] = B

    @dace.program
    def func(A, B):
        if B is None:
            func2(A, 7.0)
        else:
            func2(A, B)

    B = np.random.randn(10)
    A = np.ones((10, ))
    func(A, B)
    assert np.allclose(A, B)
    func(A, None)
    assert np.allclose(A, 7.0)


def test_none_or_field_assignment_globalarr():
    globalarr = np.random.randn(10)

    @dace.program
    def func(A, B):
        if B is None:
            C = globalarr
        else:
            C = B
        A[...] = C

    B = np.random.randn(10)
    A = np.ones((10, ))
    func(A, B)
    assert np.allclose(A, B)
    func(A, None)
    assert np.allclose(A, globalarr)


def test_none_or_field_assignment_arr():
    @dace.program
    def func(A, B, arr):
        if B is None:
            C = arr
        else:
            C = B
        A[...] = C

    B = np.random.randn(10)
    A = np.ones((10, ))
    arr = np.random.randn(10)
    func(A, B, arr)
    assert np.allclose(A, B)
    func(A, None, arr)
    assert np.allclose(A, arr)


def test_none_arg():
    @dace.program
    def some_func(field, may_be_none):
        if may_be_none is None:
            field[...] = 1.0
        else:
            field[...] = 2.0

    field = np.zeros((10, ))
    some_func(field, None)
    assert np.allclose(field, 1.0)


@pytest.mark.skip
def test_maybe_none_scalar_arg():
    @dace.program
    def some_func(field, a_scalar):
        if a_scalar is not None:
            field[...] = a_scalar

    field = np.zeros((10, ))
    some_func(field, 3.0)
    assert np.allclose(field, 3.0)


def test_default_arg():
    @dace.program
    def func(arg2=None):
        if arg2 is None:
            return 1.0
        else:
            return 2.0

    res = func()
    assert res == 1.0


def test_kwarg_none():
    @dace.program
    def func(arg2):
        if arg2 is None:
            return 1.0
        else:
            return 2.0

    @dace.program
    def outer(arg2):
        return func(arg2=None)

    res = outer(1.0)
    assert res == 1.0


def test_conditional_print():
    @dace.program
    def inner(do_print: dace.constant = False):
        if do_print:
            print("PRINT!")

    @dace.program
    def func():
        inner(do_print=False)

    func()


if __name__ == '__main__':
    test_none_or_field_call()
    test_none_or_field_assignment_globalarr()
    test_none_or_field_assignment_arr()
    test_none_arg()
    # test_maybe_none_scalar_arg()
    test_default_arg()
    test_kwarg_none()
    test_conditional_print()
