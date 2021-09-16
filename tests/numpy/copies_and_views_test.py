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
    test_set_by_view()
    test_set_by_view_1()
    test_set_by_view_2()
    test_set_by_view_3()
    test_is_a_copy()
    