# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

N = dace.symbol('N')
M = dace.symbol('M')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 38,
    N: 42,
}, {
    M: 116,
    N: 124,
}, {
    M: 390,
    N: 410,
}, {
    M: 1900,
    N: 2100,
}, {
    M: 1800,
    N: 2200,
}]

args = [([M, N], datatype), ([N], datatype), ([N], datatype)]


def init_array(A, x, y, n, m):
    fn = datatype(n)

    for i in range(n):
        x[i] = 1 + (i / fn)
    for i in range(m):
        for j in range(n):
            A[i, j] = datatype((i + j) % n) / (5 * m)


@dace.program
def atax(A: datatype[M, N], x: datatype[N], y: datatype[N]):
    tmp = dace.define_local([M], dtype=datatype)

    @dace.map
    def reset_y(i: _[0:N]):
        out >> y[i]
        out = 0.0

    for i in range(M):

        @dace.map
        def compute_tmp(j: _[0:N]):
            inA << A[i, j]
            inx << x[j]
            out >> tmp(1, lambda a, b: a + b, 0)[i]
            out = inA * inx

        @dace.map
        def compute_y(j: _[0:N]):
            inA << A[i, j]
            intmp << tmp[i]
            outy >> y(1, lambda a, b: a + b)[j]
            outy = inA * intmp


if __name__ == '__main__':
    polybench.main(sizes, args, [(2, 'y')], init_array, atax)
