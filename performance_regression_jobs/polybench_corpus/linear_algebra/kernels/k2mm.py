# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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
    # npbench formulation: ``D = alpha * A @ B @ C + beta * D`` (chained MatMul library nodes).
    # ``alpha``/``beta`` are 1-element arrays in the corpus signature, so index the scalar out.
    D[:] = alpha[0] * A @ B @ C + beta[0] * D


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    if polybench:
        polybench.main(sizes, args, [(3, 'D')], init_array, k2mm)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        k2mm(*args)
