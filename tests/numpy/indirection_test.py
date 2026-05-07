# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Declaration of symbolic variables
M, N = (dace.symbol(name) for name in ['M', 'N'])


@dace.program
def indirection(A: dace.float64[M], x: dace.int32[N]):

    A[:] = 1.0
    for j in range(1, N):
        A[x[j]] += A[x[j - 1]]


def test_indirection():
    M = 100
    N = 100

    x = np.ndarray((N, ), dtype=np.int32)
    for i in range(N):
        x[i] = N - 1 - i
    A = np.ndarray((M, ), dtype=np.float64)

    indirection(A, x)

    npA = np.ndarray((M, ), dtype=np.float64)
    npA[:] = 1.0
    for j in range(1, N):
        npA[x[j]] += npA[x[j - 1]]

    rel_norm = np.linalg.norm(npA - A) / np.linalg.norm(npA)

    print(rel_norm)
    assert rel_norm < 1e-12


if __name__ == '__main__':
    test_indirection()
