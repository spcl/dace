#!/usr/bin/env python
import dace
import numpy as np

W = dace.symbol()

number = 42


@dace.external_function
def f(A, number):
    @dace.map(_[0:W])
    def bla(i):
        inp << A[i]
        out >> number[i]
        out = 2 * inp


@dace.program
def prog(A, B):
    no = dace.define_local([number], dace.float32)
    number = dace.define_local([W], dace.float32)

    f(A, number)

    @dace.map(_[0:W])
    def bla2(i):
        inp << number[i]
        out >> B[i]
        out = 2 * inp


if __name__ == '__main__':
    W.set(3)

    A = dace.ndarray([W])
    B = dace.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = dace.float32(0.0)

    prog(A, B)

    diff = np.linalg.norm(4 * A - B) / W.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
