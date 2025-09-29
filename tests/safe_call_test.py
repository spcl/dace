# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def indirect_access(A: dace.float64[5], B: dace.float64[5], idx: dace.int64[5]):
    for i in range(5):
        A[idx[i]] = B[idx[i]] + 1

def test():
    sdfg = indirect_access.to_sdfg()

    A = np.zeros((5,), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    sdfg.safe_call(A, B, idx)
    assert np.allclose(A, B + 1)

    # This should raise an exception, but not crash
    A = np.zeros((5,), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    idx = np.array([400, 800, 1600, 3200, 6400], dtype=np.int64)
    caught = False
    try:
      sdfg.safe_call(A, B, idx)
      caught = False
    except Exception as e:
      print("Caught exception:", e)
      caught = True
      return
    assert caught, "Exception not raised!"


if __name__ == '__main__':
    test()
