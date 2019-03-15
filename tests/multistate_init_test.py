#!/usr/bin/env python
import dace
import numpy as np

W = dace.symbol()


@dace.program
def prog(A):
    number = dace.define_local([1], dace.float32)

    @dace.map(_[0:W])
    def bla(i):
        inp << A[i]
        out >> A[i]
        osum >> number(1, lambda x, y: x + y, 0)

        out = 2 * inp
        osum = inp

    @dace.map(_[0:W])
    def bla2(i):
        inp << A[i]
        out >> A[i]

        out = 2 * inp


if __name__ == '__main__':
    W.set(3)

    A = dace.ndarray([W])
    regression = dace.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    regression[:] = A[:]

    prog(A)

    diff = np.linalg.norm(4 * regression - A) / W.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
