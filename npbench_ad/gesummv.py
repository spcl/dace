import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
@dc.program
def gesummv(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
           B: dc.float64[N, N], x: dc.float64[N], D: dc.float64[N], S: dc.float64[1]):

    D[:] = alpha * A @ x + beta * B @ x
    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << D[i]
        s = z


sdfg = gesummv.to_sdfg()

sdfg.save("log_sdfgs/gesummv_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gesummv_backward.sdfg")

