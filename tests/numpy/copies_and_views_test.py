# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace.frontend.python import parser


@dace.program
def set_by_view(A: dace.int64[10]):
    v = A
    v += 1


def test_set_by_view():
    val = np.arange(10)
    set_by_view(val)
    ref = np.arange(10)
    set_by_view.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_1(A: dace.int64[10]):
    v = A[:]
    v += 1


def test_set_by_view_1():
    val = np.arange(10)
    set_by_view_1(val)
    ref = np.arange(10)
    set_by_view_1.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_2(A: dace.int64[10]):
    v = A[1:-1]
    v += 1


def test_set_by_view_2():
    val = np.arange(10)
    set_by_view_2(val)
    ref = np.arange(10)
    set_by_view_2.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_3(A: dace.int64[10]):
    v = A[4:5]
    v += 1


def test_set_by_view_3():
    val = np.arange(10)
    set_by_view_3(val)
    ref = np.arange(10)
    set_by_view_3.f(ref)
    assert (np.allclose(val, ref))


@dace.program
def set_by_view_4(A: dace.float64[10]):
    B = A[1:-1]
    B[...] = 2.0
    B += 1.0


def test_set_by_view_4():
    A = np.ones((10, ), dtype=np.float64)

    set_by_view_4(A)

    assert np.all(A[1:-1] == 3.0)
    assert A[0] == 1.0
    assert A[-1] == 1.0


@dace.program
def inner(A: dace.float64[8]):
    tmp = 2 * A[1:]
    A[:-1] = tmp


@dace.program
def set_by_view_5(A: dace.float64[10]):
    inner(A[1:-1])


def test_set_by_view_5():
    A = np.ones((10, ), dtype=np.float64)

    set_by_view_5(A)

    assert np.all(A[1:-2] == 2.0)
    assert A[0] == 1.0
    assert np.all(A[-2:] == 1.0)


@dace.program
def is_a_copy(A: dace.int64[10]):
    v = A[4]
    v += 1


def test_is_a_copy():
    val = np.arange(10)
    is_a_copy(val)
    ref = np.arange(10)
    is_a_copy.f(ref)
    assert (np.allclose(val, ref))


if __name__ == '__main__':
    # test_set_by_view()
    # test_set_by_view_1()
    # test_set_by_view_2()
    # test_set_by_view_3()
    # test_set_by_view_4()
    test_set_by_view_5()
    test_is_a_copy()
