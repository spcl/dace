import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M =  32
@dc.program
def syrk(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N],
           A: dc.float64[N, M], S: dc.float64[1]):

    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << C[i, j]
        s = z


alpha = 0.2
beta = 1.2

sdfg = syrk.to_sdfg()

sdfg.save("log_sdfgs/syrk_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/syrk_backward.sdfg")

