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

args = [([N, M], datatype), ([M], datatype), ([N], datatype), ([M], datatype), ([N], datatype)]


def init_array(A, s, q, p, r):
    n = N.get()
    m = M.get()

    for i in range(m):
        p[i] = datatype(i % m) / m
    for i in range(n):
        r[i] = datatype(i % n) / n
        for j in range(m):
            A[i, j] = datatype(i * (j + 1) % n) / n


@dace.program(datatype[N, M], datatype[M], datatype[N], datatype[M], datatype[N])
def bicg(A, s, q, p, r):
    @dace.map
    def reset_s(i: _[0:M]):
        out >> s[i]
        out = 0.0

    @dace.map
    def compute(i: _[0:N], j: _[0:M]):
        inA << A[i, j]
        inr << r[i]
        inp << p[j]
        outs >> s(1, lambda a, b: a + b)[j]
        outq >> q(1, lambda a, b: a + b)[i]
        outs = inr * inA
        outq = inA * inp


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 's'), (2, 'q')], init_array, bicg)
