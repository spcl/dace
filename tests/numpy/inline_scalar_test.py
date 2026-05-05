# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol("M")
K = dace.symbol("K")


@dace.program
def transpose_add(A: dace.float32[M, K], B: dace.float32[K, M]):
    for i, j in dace.map[0:M, 0:K]:
        B[j, i] = A[i, j] + 1


def test_inline_scalar():
    A = np.random.rand(25, 24).astype(np.float32)
    B = np.random.rand(24, 25).astype(np.float32)

    transpose_add(A, B)

    diff = np.linalg.norm(A.transpose() - B + 1)
    print('Difference:', diff)
    assert diff < 1e-5


if __name__ == '__main__':
    test_inline_scalar()
