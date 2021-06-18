# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Declaration of symbolic variables
M, N = (dace.symbol(name) for name in ['M', 'N'])


@dace.program
def range_indirection(A: dace.float64[M, N], x: dace.int32[M]):

    A[:] = 1.0
    for j in range(1, M):
        A[x[j]] += A[x[j - 1]]


def test():
    M.set(100)
    N.set(100)

    x = np.ndarray((M.get(), ), dtype=np.int32)
    for i in range(M.get()):
        x[i] = M.get() - 1 - i
    A = np.ndarray((M.get(), N.get()), dtype=np.float64)

    range_indirection(A, x)

    npA = np.ndarray((M.get(), N.get()), dtype=np.float64)
    npA[:] = 1.0
    for j in range(1, M.get()):
        npA[x[j]] += npA[x[j - 1]]

    rel_norm = np.linalg.norm(npA - A) / np.linalg.norm(npA)

    print(rel_norm)
    assert rel_norm < 1e-12


if __name__ == '__main__':
    test()
