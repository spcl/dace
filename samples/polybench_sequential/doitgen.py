# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

NQ = dace.symbol('NQ')
NR = dace.symbol('NR')
NP = dace.symbol('NP')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    NQ: 8,
    NR: 10,
    NP: 12
}, {
    NQ: 20,
    NR: 25,
    NP: 30
}, {
    NQ: 40,
    NR: 50,
    NP: 60
}, {
    NQ: 140,
    NR: 150,
    NP: 160
}, {
    NQ: 220,
    NR: 250,
    NP: 270
}]

args = [([NR, NQ, NP], datatype), ([NP, NP], datatype)]


def init_array(A, C4):
    nr = NR.get()
    nq = NQ.get()
    np = NP.get()

    for i in range(nr):
        for j in range(nq):
            for k in range(np):
                A[i, j, k] = datatype((i * j + k) % np) / np
    for i in range(np):
        for j in range(np):
            C4[i, j] = datatype((i * j) % np) / np


@dace.program
def doitgen(A: datatype[NR, NQ, NP],
            C: datatype[NP, NP]):
    for r in range(NR):
        for q in range(NQ):
            sum = dace.define_local([NP], dtype=datatype)
            for p in range(NP):
                sum[p] = 0
                for s in range(NP):
                    with dace.tasklet:
                        c_in << C[s, p]
                        a_in << A[r, q, s]
                        sum_in << sum[p]
                        sum_out >> sum[p]
                        sum_out = sum_in + (a_in * c_in)
            for p in range(NP):
                A[r, q, p] = sum[p]


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, doitgen)
