# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N, K = (dace.symbol(name) for name in ['M', 'N', 'K'])


@dace.program
def copy3d(A: dace.float32[M, N, K], B: dace.float32[M, N, K]):
    for i in parrange(M):
        for j, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, j, k]
                b >> B[i, j, k]
                b = a


def test_copy3d():
    N = M = K = 24
    A = np.random.rand(M, N, K).astype(np.float32)
    B = np.random.rand(M, N, K).astype(np.float32)
    copy3d(A, B)

    diff = np.linalg.norm(B - A) / (M * N)
    print('Difference:', diff)
    assert diff < 1e-5


def test_map_python():
    A = np.random.rand(20, 20)
    B = np.random.rand(20, 20)
    for i, j in dace.map[0:20, 1:20]:
        B[i, j] = A[i, j]

    assert np.allclose(A[:, 1:], B[:, 1:])


if __name__ == '__main__':
    test_copy3d()
    test_map_python()
