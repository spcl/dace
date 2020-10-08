# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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


if __name__ == '__main__':
    control_flow_test.compile(dace.float32[W, H], dace.float32[H, W],
                              dace.float32[1])
