# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def control_flow_test(A, B, tol):
    if tol[0] < 4:
        while tol[0] < 4:

            @dace.map(_[0:W])
            def something(i):
                a << A[0, i]
                b >> B[0, i]
                t >> tol(1, lambda x, y: x + y)
                b = a
                t = a * a
    elif tol[0] <= 5:

        @dace.map(_[0:W])
        def something(i):
            a << A[0, i]
            b >> B[0, i]
            b = a
    elif tol[0] <= 6:

        @dace.map(_[0:W])
        def something(i):
            a << A[0, i]
            b >> B[0, i]
            b = a
    else:
        for i in range(W):

            @dace.map(_[0:W])
            def something(j):
                a << A[0, j]
                b >> B[0, j]
                b = a


@dace.program
def fictest(A: dace.int32[4]):
    for a in range(min(A[0], A[1])):
        with dace.tasklet:
            inp << A[2]
            out >> A[3]
            out = inp + a


def test_control_flow_basic():
    control_flow_test.compile(dace.float32[W, H], dace.float32[H, W],
                              dace.float32[1])


def test_function_in_condition():
    A = np.random.randint(0, 10, 4, dtype=np.int32)
    expected = A.copy()
    for a in range(min(A[0], A[1])):
        expected[3] = expected[2] + a

    fictest(A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_control_flow_basic()
    test_function_in_condition()
