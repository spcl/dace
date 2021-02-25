# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace


@dace.program
def foo123(a: dace.float32[2, 3], b: dace.float32[2, 3]):
    b[0, :] = a[0, :]


def test_subarray_assignment():
    A = np.full((2, 3), 3, dtype=np.float32)
    B = np.full((2, 3), 4, dtype=np.float32)

    foo123(A, B)

    assert np.allclose(B[0, :], A[0, :])


if __name__ == '__main__':
    test_subarray_assignment()
