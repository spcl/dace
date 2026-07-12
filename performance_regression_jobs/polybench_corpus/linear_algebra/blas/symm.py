# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 20, N: 30}, {M: 60, N: 80}, {M: 200, N: 240}, {M: 1000, N: 1200}, {M: 2000, N: 2600}]

args = [([M, N], datatype), ([M, M], datatype), ([M, N], datatype), ([1], datatype), ([1], datatype)]

outputs = [(0, 'C')]


def init_array(C, A, B, alpha, beta, n, m):
    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(m):
        for j in range(n):
            C[i, j] = datatype((i + j) % 100) / m
            B[i, j] = datatype((n + i - j) % 100) / m
    for i in range(m):
        for j in range(i + 1):
            A[i, j] = datatype((i + j) % 100) / m
        for j in range(i + 1, m):
            A[i, j] = -999
            # regions of arrays that should not be used


@dace.program
def symm(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):
    # npbench formulation: symmetric-matrix multiply expressed with column slices + a
    # ``B[:i, j] @ A[i, :i]`` inner product (Dot library node). ``alpha``/``beta`` are 1-element
    # arrays in the corpus signature.
    temp2 = np.zeros((N, ), dtype=C.dtype)
    C *= beta[0]
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha[0] * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha[0] * B[i, :] * A[i, i] + alpha[0] * temp2


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, outputs, init_array, symm)
