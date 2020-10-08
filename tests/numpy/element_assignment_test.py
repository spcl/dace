# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace


@dace.program
def foo123(a: dace.float32[2], b: dace.float32[2]):
    b[0] = a[0]


if __name__ == '__main__':
    A = np.array([1, 2], dtype=np.float32)
    B = np.array([3, 4], dtype=np.float32)

    foo123(A, B)

    if A[0] != B[0]:
        exit(1)
