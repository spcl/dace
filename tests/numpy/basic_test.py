# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N, K = 24, 24, 24


@dace.program
def gemm(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N],
         alpha: dace.float32, beta: dace.float32):
    C[:] = alpha * A @ B + beta * C


if __name__ == '__main__':
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    origC = np.zeros([M, N], dtype=np.float32)
    origC[:] = C
    gemm(A, B, C, 1.0, 1.0)

    realC = 1.0 * (A @ B) + 1.0 * origC
    diff = np.linalg.norm(C - realC) / (M * N)
    print('Difference:', diff)
    exit(1 if diff >= 1e-5 else 0)
