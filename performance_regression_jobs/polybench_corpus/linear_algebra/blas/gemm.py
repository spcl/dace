# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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


def init_array(C, A, B, alpha, beta, ni, nj, nk):
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


@dace.program
def gemm(C: datatype[NI, NJ], A: datatype[NI, NK], B: datatype[NK, NJ], alpha: datatype[1], beta: datatype[1]):
    # npbench formulation: ``C[:] = alpha * A @ B + beta * C`` (the ``@`` lowers to a
    # Gemm/MatMul library node instead of the scalar triple-loop). ``alpha``/``beta`` are
    # 1-element arrays in the corpus signature, so index the scalar out.
    C[:] = alpha[0] * A @ B + beta[0] * C


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    if polybench:
        polybench.main(sizes, args, [(0, 'C')], init_array, gemm)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        gemm(*args)
