# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program
def dace_sum(X_in: dace.float32[N], X_out: dace.float32[1]):
    dace.reduce(lambda a, b: a + b, X_in, X_out, identity=0)


@dace.program
def dace_max(X_in: dace.float32[N], X_out: dace.float32[1]):
    dace.reduce(lambda a, b: max(a, b), X_in, X_out)


@dace.program
def dace_softmax(X_in: dace.float32[N], X_out: dace.float32[N]):

    tmp_max = dace.define_local([1], dtype=dace.float32)
    tmp_sum = dace.define_local([1], dtype=dace.float32)

    dace_max(X_in, tmp_max)

    @dace.map
    def softmax_tasklet_sub(i: _[0:N]):
        x_in << X_in[i]
        x_max << tmp_max
        x_out >> X_out[i]

        x_out = exp(x_in - x_max)

    dace_sum(X_out, tmp_sum)

    @dace.map
    def softmax_tasklet_div(i: _[0:N]):
        x_in << X_out[i]
        x_sum << tmp_sum
        x_out >> X_out[i]

        x_out = x_in / x_sum


@dace.program
def sdfg_transpose(A: dace.float32[M, K], B: dace.float32[K, M]):
    for i, j in dace.map[0:M, 0:K]:
        B[j, i] = A[i, j]


# sdfg_transpose.to_sdfg()

Qsize = dace.symbol('Qsize')
numHeads = dace.symbol('numHeads')
projQsize = dace.symbol('projQsize')
seqLenQ = dace.symbol('seqLenQ')
seqLenK = dace.symbol('seqLenK')
batchSize = dace.symbol('batchSize')


@dace.program
def attn_fwd(q: dace.float32[batchSize, Qsize, seqLenQ], k: dace.float32[batchSize, Qsize, seqLenK],
             v: dace.float32[batchSize, Qsize, seqLenK], wq: dace.float32[numHeads, projQsize, Qsize],
             wk: dace.float32[numHeads, projQsize, Qsize], wv: dace.float32[numHeads, projQsize, Qsize],
             wo: dace.float32[numHeads, Qsize, projQsize], out: dace.float32[batchSize, Qsize, seqLenQ]):

    for b in dace.map[0:batchSize]:

        outs = dace.define_local([numHeads, Qsize, seqLenQ], dace.float32)

        for h in dace.map[0:numHeads]:

            # q_bar = dace.define_local([projQsize, seqLenQ], dace.float32)
            k_bar = dace.define_local([projQsize, seqLenK], dace.float32)
            v_bar = dace.define_local([projQsize, seqLenK], dace.float32)

            beta = dace.define_local([seqLenK, seqLenQ], dace.float32)
            alpha = dace.define_local([seqLenK, seqLenQ], dace.float32)

            h_bar = dace.define_local([projQsize, seqLenQ], dace.float32)

            # q_bar[:] = wq[h] @ q[b,:,:] # projQsize x seqLenQ
            q_bar = wq[h, :, :] @ q[b, :, :]  # projQsize x seqLenQ
            k_bar[:, :] = wk[h, :, :] @ k[b, :, :]  # projQsize x seqLenK
            v_bar[:, :] = wv[h, :, :] @ v[b, :, :]  # projQsize x seqLenK

            k_bar_t = dace.define_local([seqLenK, projQsize], dace.float32)
            sdfg_transpose(k_bar, k_bar_t)
            beta[:, :] = k_bar_t @ q_bar[:, :]  # seqLenK x seqLenQ

            for j in dace.map[0:seqLenK]:
                dace_softmax(beta[j], alpha[j])
            # alpha[:,:] = softmax(beta[:,:]) # rowwise softmax: seqLenK x seqLenQ

            h_bar[:, :] = v_bar[:, :] @ alpha[:, :]  # projQsize x seqLenQ
            outs[h, :, :] = wo[h, :, :] @ h_bar[:, :]  # Qsize x seqLenQ

        out[b, :, :] = dace.reduce(lambda a, b: a + b, outs, axis=0)


def test_attention():
    print("=== Generating SDFG ===")
    sdfg = attn_fwd.to_sdfg()
    print("=== Compiling ===")
    sdfg.compile()


if __name__ == '__main__':
    test_attention()
