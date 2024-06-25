import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32

@dc.program
def bicg(A: dc.float64[N, M], B: dc.float64[M], D: dc.float64[N], p: dc.float64[M], r: dc.float64[N], S: dc.float64[1]):

    B[:], D[:] = r @ A, A @ p

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i]
        s = z


sdfg = bicg.to_sdfg()

sdfg.save("log_sdfgs/bicg_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/bicg_backward.sdfg")

