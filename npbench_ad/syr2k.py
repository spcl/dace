import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M =  32
@dc.program
def syr2k(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N],
           A: dc.float64[N, M], B: dc.float64[N, M], S: dc.float64[1]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << C[i, j]
        s = z



sdfg = syr2k.to_sdfg()

sdfg.save("log_sdfgs/syr2k_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/syr2k_backward.sdfg")

