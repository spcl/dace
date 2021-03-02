# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_multiassign():
    @dace.program
    def multiassign(A: dace.float64[20], B: dace.float64[1],
                    C: dace.float64[2]):
        tmp = C[0] = A[5]
        B[0] = tmp

    A = np.random.rand(20)
    B = np.random.rand(1)
    C = np.random.rand(2)
    multiassign(A, B, C)
    assert B == C[0] and C[0] == A[5]


def test_multiassign_mutable():
    @dace.program
    def mutable(D: dace.float64[2]):
        D[0] += 1
        return D[0]

    @dace.program
    def multiassign(B: dace.float64[1],
                    C: dace.float64[2]):
        tmp = C[1] = mutable(C)
        B[0] = tmp

    B = np.random.rand(1)
    C = np.random.rand(2)
    expected = C[0] + 1
    multiassign(B, C)
    assert B[0] == expected and C[1] == expected


if __name__ == '__main__':
    test_multiassign()
    test_multiassign_mutable()
