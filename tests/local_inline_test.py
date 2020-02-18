#!/usr/bin/env python
import dace
import numpy as np

W = dace.symbol('W')


@dace.program
def bla(AA, BB):
    tmp = dace.define_local([W], AA.dtype)

    @dace.map(_[0:W])
    def compute(i):
        a << AA[i]
        b >> tmp[i]
        b = -a

    @dace.map(_[0:W])
    def compute2(i):
        a << tmp[i]
        b >> BB[i]
        b = a + 1


@dace.program
def prog(A: dace.float64[W], B: dace.float64[W], C: dace.float64[W]):
    bla(A, B)
    bla(B, C)


if __name__ == '__main__':
    W.set(3)

    A = dace.ndarray([W])
    B = dace.ndarray([W])
    C = dace.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = 0.0
    C[:] = 0.0

    prog(A, B, C)

    diff = np.linalg.norm((-(-A + 1) + 1) - C) / W.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
