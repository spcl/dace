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


@pytest.mark.skip('Needs Reference support')
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


@pytest.mark.skip('Needs Reference support')
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


@pytest.mark.skip('Reference scalars unsupported in Python frontend (fails without simplification)')
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
    def inner(do_print: dace.compiletime = False):
        if do_print:
            print("PRINT!")

    @dace.program
    def func():
        inner(do_print=False)

    func()


def test_conditional_expression_over_symbols():
    """``a if cond else b`` in a map body, where the condition is a runtime symbol relation.

    Preprocessing folds the conditional when the condition is known at parse time, so what this
    exercises is the case that reaches the program visitor and becomes a symbolic ``ITE``.
    """

    @dace.program
    def identity(out: dace.float64[6, 6]):
        for i, j in dace.map[0:6, 0:6]:
            out[i, j] = 1 if i == j else 0

    out = np.zeros((6, 6))
    identity(out=out)
    assert np.allclose(out, np.eye(6))


def test_conditional_expression_branches_are_values_not_constants():
    """Both branches are ordinary values, and the condition is not an equality."""

    @dace.program
    def banded(out: dace.float64[8, 8]):
        for i, j in dace.map[0:8, 0:8]:
            out[i, j] = 2.5 if i > j else -1.0

    out = np.zeros((8, 8))
    banded(out=out)
    rows, cols = np.indices((8, 8))
    assert np.allclose(out, np.where(rows > cols, 2.5, -1.0))


def test_conditional_expression_over_data_keeps_the_old_value():
    """``b[i] = a[i] if cond else b[i]`` -- the if-then form, where the false case is what was there.

    Its operands are DATA, so the choice is a dataflow one and cannot be a symbolic ``ITE``; it is
    built as the same tasklet ``numpy.where`` produces.
    """

    @dace.program
    def keep_old(a: dace.float64[8], b: dace.float64[8]):
        for i in dace.map[0:8]:
            b[i] = a[i] if a[i] > 0.5 else b[i]

    a = np.random.rand(8)
    original = np.random.rand(8)
    b = original.copy()
    keep_old(a=a, b=b)
    assert np.allclose(b, np.where(a > 0.5, a, original))


def test_conditional_expression_mixes_data_and_constants():
    """One branch data, the other a literal -- the mixed case the where builder also has to cover."""

    @dace.program
    def clamp(a: dace.float64[8], b: dace.float64[8]):
        for i in dace.map[0:8]:
            b[i] = a[i] if a[i] > 0.5 else 0.0

    a = np.random.rand(8)
    b = np.zeros(8)
    clamp(a=a, b=b)
    assert np.allclose(b, np.where(a > 0.5, a, 0.0))


def test_numpy_where_covers_the_data_case():
    """The alternative the refusal names has to actually work."""

    @dace.program
    def clamp(a: dace.float64[8], b: dace.float64[8]):
        b[:] = np.where(a > 0.0, a, 0.0)

    a = np.random.rand(8) - 0.5
    b = np.zeros(8)
    clamp(a=a, b=b)
    assert np.allclose(b, np.maximum(a, 0.0))


if __name__ == '__main__':
    test_none_or_field_call()
    # test_none_or_field_assignment_globalarr()
    # test_none_or_field_assignment_arr()
    test_none_arg()
    # test_maybe_none_scalar_arg()
    test_default_arg()
    test_kwarg_none()
    test_conditional_print()
