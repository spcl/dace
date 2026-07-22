# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 20, N: 30}, {M: 60, N: 80}, {M: 200, N: 240}, {M: 1000, N: 1200}, {M: 2000, N: 2600}]

args = [([N, N], datatype), ([N, M], datatype), ([N, M], datatype), ([1], datatype), ([1], datatype)]

outputs = [(0, 'C')]


def init_array(C, A, B, alpha, beta, n, m):
    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        for j in range(m):
            A[i, j] = datatype((i * j + 1) % n) / n
            B[i, j] = datatype((i * j + 2) % m) / m
        for j in range(n):
            C[i, j] = datatype((i * j + 3) % n) / m


@dace.program
def syr2k(C: datatype[N, N], A: datatype[N, M], B: datatype[N, M], alpha: datatype[1], beta: datatype[1]):
    # npbench formulation: symmetric rank-2k update via slice-vectorized outer products.
    # ``alpha``/``beta`` are 1-element arrays in the corpus signature.
    for i in range(N):
        C[i, :i + 1] *= beta[0]
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha[0] * B[i, k] + B[:i + 1, k] * alpha[0] * A[i, k])


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, outputs, init_array, syr2k)
