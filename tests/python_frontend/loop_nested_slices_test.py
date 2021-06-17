# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


N = dc.symbol('N', dtype=dc.int32, integer=True, positive=True)


@dc.program
def nested_slice(A: dc.float64[N, N, N], B: dc.float64[N]):
    # B = np.ndarray(N, dtype=dc.float64)
    for i in range(N):
        tmp = A[2:-2, 2:-2, i:]
        B[i] = np.sum(tmp)
    # return B


def test_nested_slice():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    nested_slice(A, B)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert(np.allclose(B, B_ref))


if __name__ == "__main__":
    test_nested_slice()