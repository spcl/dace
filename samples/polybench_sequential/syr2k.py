# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 20,
    N: 30
}, {
    M: 60,
    N: 80
}, {
    M: 200,
    N: 240
}, {
    M: 1000,
    N: 1200
}, {
    M: 2000,
    N: 2600
}]

args = [([N, N], datatype), ([N, M], datatype), ([N, M], datatype),
        ([1], datatype), ([1], datatype)]

outputs = [(0, 'C')]


def init_array(C, A, B, alpha, beta):
    n = N.get()
    m = M.get()

    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        for j in range(m):
            A[i, j] = datatype((i * j + 1) % n) / n
            B[i, j] = datatype((i * j + 2) % m) / m
        for j in range(n):
            C[i, j] = datatype((i * j + 3) % n) / m


@dace.program(datatype[N, N], datatype[N, M], datatype[N, M], datatype[1],
              datatype[1])
def syr2k(C, A, B, alpha, beta):
    for i in range(N):
        for j in range(i+1):
            with dace.tasklet:
                ic << C[i, j]
                ib << beta
                oc >> C[i, j]
                oc = ic * ib
            # C[i, j] *= beta
        for k in range(M):
            for j in range(i + 1):
                with dace.tasklet:
                    ialpha << alpha
                    b_in << B[i, k]
                    a_in << A[j, k]
                    c_in << C[i, j]
                    c_out >> C[i, j]
                    c_out = c_in + ialpha * a_in * b_in
                # C[i, j] += A[j, k] * alpha * B[i, k]
                with dace.tasklet:
                    ialpha << alpha
                    b_in << B[j, k]
                    a_in << A[i, k]
                    c_in << C[i, j]
                    c_out >> C[i, j]
                    c_out = c_in + ialpha * a_in * b_in
                # C[i, j] += B[j, k] * alpha * A[i, k]


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, syr2k)
