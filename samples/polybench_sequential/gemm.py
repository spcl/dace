# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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

args = [([NI, NJ], datatype), ([NI, NK], datatype), ([NK, NJ], datatype),
        ([1], datatype), ([1], datatype)]


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


@dace.program(datatype[NI, NJ], datatype[NI, NK], datatype[NK, NJ],
              datatype[1], datatype[1])
def gemm(C, A, B, alpha, beta):
    for i in range(NI):
        for j in range(NJ):
            C[i, j] *= beta
        for k in range(NK):
            for j in range(NJ):
                with dace.tasklet:
                    in_a << A[i, k]
                    in_b << B[k, j]
                    in_alpha << alpha
                    c_in << C[i, j]
                    c_out >> C[i, j]
                    c_out = c_in + (in_alpha * in_a * in_b)
                # C[i, j] += alpha * A[i, k] * B[k, j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'C')], init_array, gemm)