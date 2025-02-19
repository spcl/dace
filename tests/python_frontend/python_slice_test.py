# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_slice_constant():

    q = np.arange(1000).reshape(10, 10, 10).copy()
    ref = q.copy()

    direction = "x"

    @dace.program
    def slicem(A, i, j, kslice: dace.compiletime):
        if direction == "x":
            A[i, j, kslice] = A[9 - i, 9 - j, kslice]

    @dace.program
    def forloops(A, kslice: dace.compiletime):
        for i in range(3):
            for j in range(3):
                slicem(A, i, j, kslice)

    @dace.program
    def outer(A):
        forloops(A, slice(2, None))

    outer(q)
    outer.f(ref)
    assert (np.allclose(q, ref))


def test_slice():

    q = np.arange(1000).reshape(10, 10, 10).copy()
    ref = q.copy()

    @dace.program
    def forloops(A):
        for i in range(3):
            for j in range(3):
                A[i, j, slice(2, None)] = A[9 - i, 9 - j, slice(2, None)]

    forloops(q)
    forloops.f(ref)
    assert (np.allclose(q, ref))


if __name__ == '__main__':
    test_slice()
    test_slice_constant()
