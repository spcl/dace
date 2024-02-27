# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')


@dace.program
def multistate_init(A):
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


def test():
    W = 3

    A = dace.ndarray([W])
    regression = dace.ndarray([W])

    A[:] = np.mgrid[0:W]
    regression[:] = A[:]

    multistate_init(A, W=W)

    diff = np.linalg.norm(4 * regression - A) / W
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
