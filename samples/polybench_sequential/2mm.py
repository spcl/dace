# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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

args = [([NI, NK], datatype), ([NK, NJ], datatype), ([NJ, NL], datatype),
        ([NI, NL], datatype), ([1], datatype), ([1], datatype)]


def init_array(A, B, C, D, alpha, beta):
    ni = NI.get()
    nj = NJ.get()
    nk = NK.get()
    nl = NL.get()

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
def k2mm(A: datatype[NI, NK], B: datatype[NK, NJ], C: datatype[NJ, NL],
         D: datatype[NI, NL], alpha: datatype[1], beta: datatype[1]):
        tmp = dace.define_local([NI, NJ], dtype=datatype)

        for i in range(NI):
            for j in range(NJ):
                with dace.tasklet:
                    temp_out >> tmp[i, j]
                    temp_out = 0.0
                for k in range(NK):
                    with dace.tasklet:
                        temp_in << tmp[i, j]
                        temp_out >> tmp[i, j]
                        alpha_in << alpha
                        A_in << A[i, k]
                        B_in << B[k, j]
                        temp_out = temp_in + alpha_in * A_in * B_in

        for i in range(NI):
            for j in range(NL):
                with dace.tasklet:
                    D_out >> D[i, j]
                    D_in << D[i, j]
                    beta_in << beta
                    D_out = D_in * beta_in
                for k in range(NJ):
                    with dace.tasklet:
                        D_out >> D[i, j]
                        D_in << D[i, j]
                        temp_in << tmp[i, k]
                        C_in << C[k, j]
                        D_out = D_in + temp_in * C_in

if __name__ == '__main__':
    if polybench:
        polybench.main(sizes, args, [(3, 'D')], init_array, k2mm)
    else:
        [k.set(v) for k, v in sizes[2].items()]
        init_array(*args)
        k2mm(*args)
