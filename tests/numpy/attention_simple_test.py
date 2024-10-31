# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program
def dace_softmax(X_in: dace.float32[N], X_out: dace.float32[N]):

    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in)
    X_out[:] = np.exp(X_in - tmp_max)
    tmp_sum = dace.reduce(lambda a, b: a + b, X_out, identity=0)
    X_out[:] /= tmp_sum


@dace.program
def sdfg_transpose(A: dace.float32[M, K], B: dace.float32[K, M]):
    for i, j in dace.map[0:M, 0:K]:
        B[j, i] = A[i, j]


Qsize = dace.symbol('Qsize')
numHeads = dace.symbol('numHeads')
projQsize = dace.symbol('projQsize')
seqLenQ = dace.symbol('seqLenQ')
seqLenK = dace.symbol('seqLenK')
batchSize = dace.symbol('batchSize')


@dace.program
def attn_fwd(
    q: dace.float32[batchSize, Qsize, seqLenQ],
    k: dace.float32[batchSize, Qsize, seqLenK],
    v: dace.float32[batchSize, Qsize, seqLenK],
    wq: dace.float32[numHeads, projQsize, Qsize],
    wk: dace.float32[numHeads, projQsize, Qsize],
    wv: dace.float32[numHeads, projQsize, Qsize],
    wo: dace.float32[numHeads, Qsize, projQsize],
    out: dace.float32[batchSize, Qsize, seqLenQ],
):

    for b in dace.map[0:batchSize]:

        outs = dace.define_local([numHeads, Qsize, seqLenQ], dace.float32)

        for h in dace.map[0:numHeads]:

            q_bar = wq[h] @ q[b]  # projQsize x seqLenQ
            k_bar = wk[h] @ k[b]  # projQsize x seqLenK
            v_bar = wv[h] @ v[b]  # projQsize x seqLenK

            k_bar_t = dace.define_local([seqLenK, projQsize], dace.float32)
            sdfg_transpose(k_bar, k_bar_t)
            beta = k_bar_t @ q_bar  # seqLenK x seqLenQ

            alpha = dace.define_local([seqLenK, seqLenQ], dace.float32)
            for j in dace.map[0:seqLenK]:
                dace_softmax(beta[j], alpha[j])

            h_bar = v_bar @ alpha  # projQsize x seqLenQ
            outs[h] = wo[h] @ h_bar  # Qsize x seqLenQ

        out[b] = dace.reduce(lambda a, b: a + b, outs, axis=0, identity=0)


def test_attn_simple():
    print("=== Generating SDFG ===")
    sdfg = attn_fwd.to_sdfg()
    print("=== Compiling ===")
    sdfg.compile()


if __name__ == '__main__':
    test_attn_simple()
