import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M = 32 
N = 32

@dc.program
def trmm(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N], S: dc.float64[1]):

    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha

    @dc.map(_[0:M, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i, j]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[M, M])
B = np.ones(shape=[M, N])
sdfg = trmm.to_sdfg(alpha=alpha, beta=beta, A=A)

sdfg.save("log_sdfgs/trmm_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/trmm_backward.sdfg")

