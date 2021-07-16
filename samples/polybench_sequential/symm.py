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

args = [([M, N], datatype), ([M, M], datatype), ([M, N], datatype),
        ([1], datatype), ([1], datatype)]

outputs = [(0, 'C')]


def init_array(C, A, B, alpha, beta):
    n = N.get()
    m = M.get()

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

    print('aval', beta[0] * C[0, 0] + alpha[0] * B[0, 0] * A[0, 0])


@dace.program(datatype[M, N], datatype[M, M], datatype[M, N], datatype[1],
              datatype[1])
def symm(C, A, B, alpha, beta):
    temp2 = dace.define_local([M, N], datatype)

    for i in range(M):
        for j in range(N):
            # temp2 = dace.define_local_scalar(datatype)
            temp2[i, j] = 0
            for k in range(i):
                with dace.tasklet:
                    ialpha << alpha
                    ia << A[i, k]
                    ibi << B[i, j]
                    ic << C[k, j]
                    oc >> C[k, j]
                    oc = ic + ialpha * ibi * ia
                with dace.tasklet:
                    ia << A[i, k]
                    ibk << B[k, j]
                    it2 << temp2[i, j]
                    ot2 >> temp2[i, j]
                    ot2 = it2 + ibk * ia
                # C[k, j] += alpha * B[i, j] * A[i, k]
                # temp2 += B[k, j] * A[i, k]
            with dace.tasklet:
                ibeta << beta
                ib << B[i, j]
                iadiag << A[i, i]
                ialpha << alpha
                it2 << temp2[i, j]
                ic << C[i, j]
                oc >> C[i, j]
                oc = ibeta * ic + ialpha * ib * iadiag + ialpha * it2
            # C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, symm)
