# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def indirect_access(A: dace.float64[5], B: dace.float64[5], ub: dace.int64[1]):
    for i in range(ub[0]):
        A[i] = B[i] + 1


def test():
    sdfg = indirect_access.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ub = np.array([5], dtype=np.int64)
    sdfg.safe_call(A, B, ub)
    assert np.allclose(A, B + 1)

    # This should raise an exception, but not crash
    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ub = np.array([6400], dtype=np.int64)
    caught = False
    try:
        sdfg.safe_call(A, B, ub)
        caught = False
    except Exception as e:
        caught = True
    assert caught, "Exception not raised!"


if __name__ == '__main__':
    test()
