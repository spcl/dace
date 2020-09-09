# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N = 24, 24


@dace.program
def transpose_test(A: dace.float32[M, N], B: dace.float32[M, N]):
    B[:] = np.transpose(A)


if __name__ == '__main__':
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros([M, N], dtype=np.float32)
    transpose_test(A, B)

    realB = np.transpose(A)
    rel_error = np.linalg.norm(B - realB) / np.linalg.norm(realB)
    print('Relative_error:', rel_error)
    exit(1 if rel_error >= 1e-5 else 0)
