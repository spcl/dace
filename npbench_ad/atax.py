import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32

@dc.program
def atax(A: dc.float64[M, N], x: dc.float64[N], B: dc.float64[M], S: dc.float64[1]):

    B[:] = (A @ x) @ A

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i]
        s = z


sdfg = atax.to_sdfg()

sdfg.save("log_sdfgs/atax_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/atax_backward.sdfg")

