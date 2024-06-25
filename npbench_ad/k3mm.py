import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
NI = 32
NJ = 32
NK = 32
NM = 32
NL = 32
@dc.program
def k3mm(A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NM],
           D: dc.float64[NM, NL], E: dc.float64[NI, NL], S: dc.float64[1]):

    E[:] = A @ B @ C @ D

    @dc.map(_[0:NI, 0:NL])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << E[i, j]
        s = z


A = np.ones(shape=[NI, NK])
B = np.ones(shape=[NK, NJ])
C = np.ones(shape=[NJ, NM])
D = np.ones(shape=[NM, NL])
E = np.ones(shape=[NI, NL])

sdfg = k3mm.to_sdfg(A=A, B=B, C=C, D=D)

sdfg.save("log_sdfgs/k3mm_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/k3mm_backward.sdfg")

