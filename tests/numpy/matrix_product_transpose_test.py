# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N, K = 24, 12, 48


@dace.program
def matrix_product_transpose_test(A: dace.float32[K, M], B: dace.float32[N, K],
                                  C: dace.float32[M, N]):
    C[:] = np.transpose(A) @ np.transpose(B)


if __name__ == '__main__':
    A = np.random.rand(K, M).astype(np.float32)
    B = np.random.rand(N, K).astype(np.float32)
    C = np.zeros([M, N], dtype=np.float32)
    matrix_product_transpose_test(A, B, C)

    realC = np.transpose(A) @ np.transpose(B)
    rel_error = np.linalg.norm(C - realC) / np.linalg.norm(realC)
    print('Relative_error:', rel_error)
    exit(1 if rel_error >= 1e-5 else 0)
