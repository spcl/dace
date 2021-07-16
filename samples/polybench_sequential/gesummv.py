# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 30}, {N: 90}, {N: 250}, {N: 1300}, {N: 2800}]

args = [([N, N], datatype), ([N, N], datatype), ([N], datatype),
        ([N], datatype), ([N], datatype), ([1], datatype), ([1], datatype)]

outputs = [(4, 'y')]


def init_array(A, B, tmp, x, y, alpha, beta):
    n = N.get()

    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        x[i] = datatype(i % n) / n
        for j in range(n):
            A[i, j] = datatype((i * j + 1) % n) / n
            B[i, j] = datatype((i * j + 2) % n) / n


@dace.program(datatype[N, N], datatype[N, N], datatype[N], datatype[N],
              datatype[N], datatype[1], datatype[1])
def gesummv(A, B, tmp, x, y, alpha, beta):

    for i in range(N):
        tmp[i] = 0.0
        y[i] = 0.0
        for j in range(N):
            with dace.tasklet:
                ia << A[i, j]
                ix << x[j]
                it << tmp[i]
                ot >> tmp[i]
                ot = ia * ix + it
            with dace.tasklet:
                ib << B[i, j]
                ix << x[j]
                iy << y[i]
                oy >> y[i]
                oy = ib * ix + iy
        with dace.tasklet:
            iy << y[i]
            ialpha << alpha
            ibeta << beta
            it << tmp[i]
            oy >> y[i]
            oy = ialpha * it + ibeta * iy


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, gesummv)
