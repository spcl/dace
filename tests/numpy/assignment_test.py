# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def multiassign(A: dace.float64[20], B: dace.float64[1], C: dace.float64[2]):
    tmp = C[0] = A[5]
    B[0] = tmp


def test_multiassign():
    A = np.random.rand(20)
    B = np.random.rand(1)
    C = np.random.rand(2)
    multiassign(A, B, C)
    assert B == C[0] and C[0] == A[5]


if __name__ == '__main__':
    test_multiassign()
