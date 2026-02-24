# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{M: 20, N: 30}, {M: 60, N: 80}, {M: 200, N: 240}, {M: 1000, N: 1200}, {M: 2000, N: 2600}]

args = [
    ([M, M], datatype),
    ([M, N], datatype),
    ([1], datatype),
]

outputs = [(1, 'B')]


def init_array(A, B, alpha, n, m):
    alpha[0] = datatype(1.5)

    for i in range(m):
        for j in range(i):
            A[i, j] = datatype((i + j) % m) / m
        A[i, i] = 1.0
        for j in range(n):
            B[i, j] = datatype((n + (i - j)) % n) / n


@dace.program
def trmm(A: datatype[M, M], B: datatype[M, N], alpha: datatype[1]):

    @dace.mapscope
    def compute(j: _[0:N]):

        @dace.mapscope
        def computecol(i: _[0:M]):
            tmp = dace.define_local_scalar(datatype)

            @dace.tasklet
            def reset_tmp():
                out >> tmp
                out = 0

            @dace.map
            def compute_elem(k: _[i + 1:M]):
                ia << A[k, i]
                ib << B[k, j]
                ob >> tmp(1, lambda a, b: a + b)
                ob = ia * ib

            @dace.tasklet
            def mult():
                ib << B[i, j]
                ialpha << alpha
                itmp << tmp
                ob >> B[i, j]
                ob = ialpha * (ib + itmp)


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, trmm)
