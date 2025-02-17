# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def myprogram(A, B, tol):
    # Tree
    if tol[0] < 4:

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

        @dace.map(_[0:W])
        def something(j):
            a << A[0, j]
            b >> B[0, j]
            b = a

    for i in range(3):

        @dace.map(_[0:W])
        def something(j):
            a << A[1, j]
            b >> B[1, j]
            b = a

    while tol[0] < 4:

        @dace.map(_[0:W])
        def something(i):
            a << A[0, i]
            b >> B[0, i]
            t >> tol(1, lambda x, y: x + y)
            b = a
            t = a * a


def test():
    myprogram.compile(dace.float32[W, H], dace.float32[H, W], dace.float32[1])


if __name__ == "__main__":
    test()
