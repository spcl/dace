# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol("M")
K = dace.symbol("K")


@dace.program
def transpose_add(A: dace.float32[M, K], B: dace.float32[K, M]):
    for i, j in dace.map[0:M, 0:K]:
        B[j, i] = A[i, j] + 1


if __name__ == '__main__':
    K.set(24)
    M.set(25)

    A = np.random.rand(25, 24).astype(np.float32)
    B = np.random.rand(24, 25).astype(np.float32)

    transpose_add(A, B)

    diff = np.linalg.norm(A.transpose() - B + 1)
    print('Difference:', diff)
    exit(0 if diff < 1e-5 else 1)
