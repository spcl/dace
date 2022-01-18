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

args = [([N, N], datatype), ([N, M], datatype), ([1], datatype), ([1], datatype)]

outputs = [(0, 'C')]


def init_array(C, A, alpha, beta):
    n = N.get()
    m = M.get()

    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(n):
        for j in range(m):
            A[i, j] = datatype((i * j + 1) % n) / n
        for j in range(n):
            C[i, j] = datatype((i * j + 2) % m) / m


@dace.program(datatype[N, N], datatype[N, M], datatype[1], datatype[1])
def syrk(C, A, alpha, beta):
    @dace.mapscope
    def mult_c_rows(i: _[0:N]):
        @dace.map
        def mult_c_cols(j: _[0:i + 1]):
            ic << C[i, j]
            ib << beta
            oc >> C[i, j]
            oc = ic * ib

    @dace.mapscope
    def compute(i: _[0:N], k: _[0:M]):
        @dace.map
        def compute_elem(j: _[0:i + 1]):
            ialpha << alpha
            ia << A[i, k]
            iat << A[j, k]
            oc >> C(1, lambda a, b: a + b)[i, j]
            oc = ialpha * ia * iat


if __name__ == '__main__':
    polybench.main(sizes, args, outputs, init_array, syrk)
