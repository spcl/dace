import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32

@dc.program
def go_fast(a: dc.float64[N, N], S: dc.float64[1]):

    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])

    D = a + trace

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << D[i, j]
        s = z
    return a + trace


sdfg = go_fast.to_sdfg()

sdfg.save("log_sdfgs/go_fast_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["a"], outputs=["S"])

sdfg.save("log_sdfgs/go_fast_backward.sdfg")

