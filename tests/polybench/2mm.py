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
NL = dace.symbol('NL')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    NI: 16,
    NJ: 18,
    NK: 22,
    NL: 24
}, {
    NI: 40,
    NJ: 50,
    NK: 70,
    NL: 80
}, {
    NI: 180,
    NJ: 190,
    NK: 210,
    NL: 220
}, {
    NI: 800,
    NJ: 900,
    NK: 1100,
    NL: 1200
}, {
    NI: 1600,
    NJ: 1800,
    NK: 2200,
    NL: 2400
}]

args = [([NI, NK], datatype), ([NK, NJ], datatype), ([NJ, NL], datatype), ([NI, NL], datatype), ([1], datatype),
        ([1], datatype)]


def init_array(A, B, C, D, alpha, beta, ni, nj, nk, nl):
    alpha[0] = datatype(1.5)
    beta[0] = datatype(1.2)

    for i in range(ni):
        for j in range(nk):
            A[i, j] = datatype((i * j + 1) % ni) / ni
    for i in range(nk):
        for j in range(nj):
            B[i, j] = datatype(i * (j + 1) % nj) / nj
    for i in range(nj):
        for j in range(nl):
            C[i, j] = datatype((i * (j + 3) + 1) % nl) / nl
    for i in range(ni):
        for j in range(nl):
            D[i, j] = datatype(i * (j + 2) % nk) / nk


@dace.program
def k2mm(A: datatype[NI, NK], B: datatype[NK, NJ], C: datatype[NJ, NL], D: datatype[NI, NL], alpha: datatype[1],
         beta: datatype[1]):
    tmp = dace.define_local([NI, NJ], dtype=datatype)

    @dace.map
    def zerotmp(i: _[0:NI], j: _[0:NJ]):
        out >> tmp[i, j]
        out = 0.0

    @dace.map
    def mult_tmp(i: _[0:NI], j: _[0:NJ], k: _[0:NK]):
        in_a << A[i, k]
        in_b << B[k, j]
        in_alpha << alpha
        out >> tmp(1, lambda x, y: x + y)[i, j]
        out = in_alpha * in_a * in_b

    @dace.map
    def mult_d(i: _[0:NI], j: _[0:NL]):
        inp << D[i, j]
        in_beta << beta
        out >> D[i, j]

        out = inp * in_beta

    @dace.map
    def comp_d(i: _[0:NI], j: _[0:NL], k: _[0:NJ]):
        in_a << tmp[i, k]
        in_b << C[k, j]
        out >> D(1, lambda x, y: x + y)[i, j]
        out = in_a * in_b


if __name__ == '__main__':
    if polybench:
        polybench.main(sizes, args, [(3, 'D')], init_array, k2mm)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        k2mm(*args)
