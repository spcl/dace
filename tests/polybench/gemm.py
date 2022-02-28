# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
try:
    import polybench
except ImportError:
    polybench = None

NI = dace.symbol('NI')
NJ = dace.symbol('NJ')
NK = dace.symbol('NK')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    NI: 20,
    NJ: 25,
    NK: 30
}, {
    NI: 60,
    NJ: 70,
    NK: 80
}, {
    NI: 200,
    NJ: 220,
    NK: 240
}, {
    NI: 1000,
    NJ: 1100,
    NK: 1200
}, {
    NI: 2000,
    NJ: 2300,
    NK: 2600
}]

args = [([NI, NJ], datatype), ([NI, NK], datatype), ([NK, NJ], datatype), ([1], datatype), ([1], datatype)]


def init_array(C, A, B, alpha, beta):
    ni = NI.get()
    nj = NJ.get()
    nk = NK.get()

    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(ni):
        for j in range(nj):
            C[i, j] = datatype((i * j + 1) % ni) / ni
    for i in range(ni):
        for j in range(nk):
            A[i, j] = datatype(i * (j + 1) % nk) / nk
    for i in range(nk):
        for j in range(nj):
            B[i, j] = datatype(i * (j + 2) % nj) / nj


@dace.program(datatype[NI, NJ], datatype[NI, NK], datatype[NK, NJ], datatype[1], datatype[1])
def gemm(C, A, B, alpha, beta):
    @dace.map
    def mult_c(i: _[0:NI], j: _[0:NJ]):
        inp << C[i, j]
        in_beta << beta
        out >> C[i, j]

        out = inp * in_beta

    @dace.map
    def comp(i: _[0:NI], k: _[0:NK], j: _[0:NJ]):
        in_a << A[i, k]
        in_b << B[k, j]
        in_alpha << alpha
        out >> C(1, lambda x, y: x + y)[i, j]
        out = in_alpha * in_a * in_b


if __name__ == '__main__':
    if polybench:
        polybench.main(sizes, args, [(0, 'C')], init_array, gemm)
    else:
        [k.set(v) for k, v in sizes[2].items()]
        init_array(*args)
        gemm(*args)
