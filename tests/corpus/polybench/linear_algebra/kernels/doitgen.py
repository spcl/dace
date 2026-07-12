# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

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


def init_array(A, C4, nr, nq, np):
    for i in range(nr):
        for j in range(nq):
            for k in range(np):
                A[i, j, k] = datatype((i * j + k) % np) / np
    for i in range(np):
        for j in range(np):
            C4[i, j] = datatype((i * j) % np) / np


@dace.program
def doitgen(A: datatype[NR, NQ, NP], C4: datatype[NP, NP]):
    # npbench formulation: per-``r`` matmul ``A[r] @ C4`` (a Gemm/MatMul library node). This is
    # the (NQ, NP) @ (NP, NP) contraction npbench expresses as ``reshape(A[r], (NQ,1,NP)) @ C4``;
    # the plain 2-D matmul is equivalent (per-row ``A[r, q, :] @ C4``) and avoids the unsupported
    # 4-D matmul.
    for r in range(NR):
        A[r, :, :] = A[r] @ C4


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'A')], init_array, doitgen)
