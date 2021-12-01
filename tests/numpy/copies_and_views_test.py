# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace import data


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


def test_needs_view():
    @dace.program
    def nested(q, i, j):
        q[3 + j, 4 + i, 0:3] = q[3 - i + 1, 4 + j, 3:6]

    @dace.program
    def selfcopy(q: dace.float64[128, 128, 80]):
        for i in range(1, 4):
            for j in range(1, 4):
                nested(q, i, j)

    sdfg = selfcopy.to_sdfg()
    for s in sdfg.all_sdfgs_recursive():
        assert not any(
            isinstance(d, data.Array) and not isinstance(d, data.View)
            and d.transient and d.shape == (3,) for d in s.arrays.values())


def test_needs_copy():
    @dace.program
    def nested(q, i, j):
        q[3 + j, 4 + i, 0:3] = q[3 - i + 1, 4 + j, 1:4]

    @dace.program
    def selfcopy(q: dace.float64[128, 128, 80]):
        for i in range(1, 4):
            for j in range(1, 4):
                nested(q, i, j)

    sdfg = selfcopy.to_sdfg(strict=False)
    found_copy = False
    for s in sdfg.all_sdfgs_recursive():
        found_copy |= any(
            isinstance(d, data.Array)  and not isinstance(d, data.View)
            and d.transient and d.shape == (3,) for d in s.arrays.values())
    assert found_copy


if __name__ == '__main__':
    test_set_by_view()
    test_set_by_view_1()
    test_set_by_view_2()
    test_set_by_view_3()
    test_set_by_view_4()
    test_set_by_view_5()
    test_is_a_copy()
    test_needs_view()
    test_needs_copy()
