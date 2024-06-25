import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
NI = 32
NJ = 32
NK = 32
NL = 32
@dc.program
def gemm(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
           A: dc.float64[NI, NK], B: dc.float64[NK, NJ], D: dc.float64[NI, NJ], S: dc.float64[1]):

    D[:] = alpha * A @ B + beta * C
    @dc.map(_[0:NI, 0:NL])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << D[i, j]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[NI, NJ])
B = np.ones(shape=[NK, NJ])
C = np.ones(shape=[NJ, NL])
D = np.ones(shape=[NI, NL])
sdfg = gemm.to_sdfg(alpha=alpha, beta=beta, A=A, B=B, C=C, D=D)

sdfg.save("log_sdfgs/gemm_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gemm_backward.sdfg")

