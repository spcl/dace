# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')

number = 42


@dace.program
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


def test():
    W.set(3)

    A = dace.ndarray([W])
    B = dace.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = dace.float32(0.0)

    prog(A, B, W=W)

    diff = np.linalg.norm(4 * A - B) / W.get()
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
