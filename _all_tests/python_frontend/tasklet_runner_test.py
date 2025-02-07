# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_simple():
    A = np.random.rand(4)
    B = np.random.rand(4)
    with dace.tasklet():
        a << A[0]
        b >> B[1]
        b = a

    assert np.allclose(B[1], A[0])


def test_locals():
    a = 1
    b = 'aa'
    c = 3
    A = np.random.rand(4)
    B = np.random.rand(4)
    with dace.tasklet():
        a << A[0]
        b >> B[1]
        b = a + c

    assert np.allclose(B[1], A[0] + c)


def test_wcr():
    A = np.random.rand(4)
    B = np.random.rand(4)
    C = np.copy(B)

    with dace.tasklet:
        a << A[0]
        b >> B(1, lambda a, b: a * b)[1]
        b = a

    assert np.allclose(B[1], A[0] * C[1])


def test_dynamic_input():
    A = np.random.rand(4, 5)
    B = np.random.rand(4)

    with dace.tasklet:
        a << A(-1)
        b >> B[1]
        b = a[0, 3]

    assert np.allclose(A[0, 3], B[1])


def test_nested_range():
    A = np.random.rand(4, 5)
    B = np.random.rand(4)

    with dace.tasklet:
        a << A(-1)[:, 1]
        b >> B[1]
        b = a[2]

    assert np.allclose(A[2, 1], B[1])


def test_dynamic_output():
    A = np.random.rand(4)
    B = np.random.rand(4)
    C = np.copy(B)

    def doit(A, B, i):
        with dace.tasklet:
            a << A(-1)
            b >> B(-1)[i]
            if a[i] > 0.5:
                b = 0

    A[2] = 0.7
    doit(A, B, 2)
    assert np.allclose(B[2], 0)

    A[3] = 0.2
    doit(A, B, 3)
    assert np.allclose(B[3], C[3])


def test_dynamic_output_wcr():
    A = np.random.rand(20)
    B = np.zeros([1], dtype=np.int32)

    # Count number of elements >= 0.5
    for i in dace.map[0:A.shape[0]]:
        with dace.tasklet:
            a << A[i]
            b >> B(-1, lambda a, b: a + b)
            if a >= 0.5:
                b = 1

    expected = np.where(A >= 0.5)[0].shape[0]
    assert B[0] == expected


if __name__ == '__main__':
    test_simple()
    test_locals()
    test_wcr()
    test_dynamic_input()
    test_nested_range()
    test_dynamic_output()
    test_dynamic_output_wcr()
