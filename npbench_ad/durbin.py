import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

N = 32


@dc.program
def flip(A: dc.float64[M]):
    B = np.ndarray((M, ), dtype=np.float64)
    for i in dc.map[0:M]:
        B[i] = A[M - 1 - i]
    return B


@dc.program
def durbin(r: dc.float64[N], S: dc.float64[1]):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * flip(y[:k])
        y[k] = alpha

    S[0] = np.sum(y)


sdfg = durbin.to_sdfg()

sdfg.save("log_sdfgs/durbin_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["r"], outputs=["S"])

sdfg.save("log_sdfgs/durbin_backward.sdfg")
sdfg.compile()
