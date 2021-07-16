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

args = [
    ([M, M], datatype),
    ([M, N], datatype),
    ([1], datatype),
]

outputs = [(1, 'B')]


def init_array(A, B, alpha):
    n = N.get()
    m = M.get()

    alpha[0] = datatype(1.5)

    for i in range(m):
        for j in range(i):
            A[i, j] = datatype((i + j) % m) / m
        A[i, i] = 1.0
        for j in range(n):
            B[i, j] = datatype((n + (i - j)) % n) / n


@dace.program(datatype[M, M], datatype[M, N], datatype[1])
def trmm(A, B, alpha):
    # for small inputs is faster without dace.taskelt
    for i in range(M):
        for j in range(N):
            for k in range(i+1, M):
                with dace.tasklet:
                    aki_in << A[k][i]
                    bkj_in << B[k][j]
                    bij_in << B[i][j]
                    bij_out >> B[i][j]
                    bij_out = bij_in + (aki_in * bkj_in)
                # B[i][j] += A[k][i] * B[k][j]
            with dace.tasklet:
                ialpha << alpha
                bij_in << B[i][j]
                bij_out >> B[i][j]
                bij_out = ialpha * bij_in
            # B[i][j] = alpha * B[i][j]

if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, trmm)
