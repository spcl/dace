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
NM = dace.symbol('NM')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    NI: 16,
    NJ: 18,
    NK: 20,
    NL: 22,
    NM: 24
}, {
    NI: 40,
    NJ: 50,
    NK: 60,
    NL: 70,
    NM: 80
}, {
    NI: 180,
    NJ: 190,
    NK: 200,
    NL: 210,
    NM: 220
}, {
    NI: 800,
    NJ: 900,
    NK: 1000,
    NL: 1100,
    NM: 1200
}, {
    NI: 1600,
    NJ: 1800,
    NK: 2000,
    NL: 2200,
    NM: 2400
}]

args = [([NI, NK], datatype), ([NK, NJ], datatype), ([NJ, NM], datatype),
        ([NM, NL], datatype), ([NI, NL], datatype)]


def init_array(A, B, C, D, G):
    ni = NI.get()
    nj = NJ.get()
    nk = NK.get()
    nl = NL.get()
    nm = NM.get()

    for i in range(ni):
        for j in range(nk):
            A[i, j] = datatype((i * j + 1) % ni) / (5 * ni)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = datatype((i * (j + 1) + 2) % nj) / (5 * nj)
    for i in range(nj):
        for j in range(nm):
            C[i, j] = datatype(i * (j + 3) % nl) / (5 * nl)
    for i in range(nm):
        for j in range(nl):
            D[i, j] = datatype((i * (j + 2) + 2) % nk) / (5 * nk)


@dace.program(datatype[NI, NK], datatype[NK, NJ], datatype[NJ, NM],
              datatype[NM, NL], datatype[NI, NL])
def k3mm(A, B, C, D, G):
    E = dace.define_local([NI, NJ], dtype=datatype)
    F = dace.define_local([NJ, NL], dtype=datatype)

    for i in range(NI):
        for j in range(NJ):
            E[i, j] = 0
            for k in range(NK):
                with dace.tasklet:
                    a_in << A[i, k]
                    b_in << B[k, j]
                    e_in << E[i, j]
                    e_out >> E[i, j]
                    e_out = e_in + (a_in * b_in)
                # E[i, j] += A[i, k] * B[k, j]

    for i in range(NJ):
        for j in range(NL):
            F[i, j] = 0
            for k in range(NM):
                with dace.tasklet:
                    c_in << C[i, k]
                    d_in << D[k, j]
                    f_in << F[i, j]
                    f_out >> F[i, j]
                    f_out = f_in + (c_in * d_in)
                # F[i, j] += C[i, k] * D[k, j]
    for i in range(NI):
        for j in range(NL):
            G[i, j] = 0
            for k in range(NJ):
                with dace.tasklet:
                    e_in << E[i, k]
                    f_in << F[k, j]
                    g_in << G[i, j]
                    g_out >> G[i, j]
                    g_out = g_in + (e_in * f_in)
                # G[i, j] += E[i, k] * F[k, j]


if __name__ == '__main__':
    if polybench:
        polybench.main(sizes, args, [(4, 'G')], init_array, k3mm)
    else:
        [k.set(v) for k, v in sizes[2].items()]
        init_array(*args)
        k3mm(*args)
