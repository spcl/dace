import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M = 32 
N = 32

@dc.program
def trisolv(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N], S: dc.float64[1]):

    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << x[i]
        s = z


alpha = 0.2
beta = 1.2

A = np.ones(shape=[M, M])
B = np.ones(shape=[M, N])
sdfg = trisolv.to_sdfg()

sdfg.save("log_sdfgs/trisolv_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["L"], outputs=["S"])

sdfg.save("log_sdfgs/trisolv_backward.sdfg")

