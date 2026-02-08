# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 30}, {N: 90}, {N: 250}, {N: 1300}, {N: 2800}]

args = [([N, N], datatype), ([N, N], datatype), ([N], datatype), ([N], datatype), ([N], datatype), ([1], datatype),
        ([1], datatype)]

outputs = [(4, 'y')]


def init_array(A, B, tmp, x, y, alpha, beta, n):
    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        x[i] = datatype(i % n) / n
        for j in range(n):
            A[i, j] = datatype((i * j + 1) % n) / n
            B[i, j] = datatype((i * j + 2) % n) / n


@dace.program
def gesummv(A: datatype[N, N], B: datatype[N, N], tmp: datatype[N], x: datatype[N], y: datatype[N], alpha: datatype[1],
            beta: datatype[1]):

    @dace.map
    def compute_ty(i: _[0:N], j: _[0:N]):
        ia << A[i, j]
        ib << B[i, j]
        ix << x[j]
        ot >> tmp(1, lambda a, b: a + b, 0)[i]
        oy >> y(1, lambda a, b: a + b, 0)[i]

        ot = ia * ix
        oy = ib * ix

    @dace.map
    def update_y(i: _[0:N]):
        iy << y[i]
        ialpha << alpha
        ibeta << beta
        it << tmp[i]
        oy >> y[i]
        oy = ialpha * it + ibeta * iy


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, gesummv)
