import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32

@dc.program
def compute(array_1: dc.int64[M, N], array_2: dc.int64[M, N], a: dc.int64,
            b: dc.int64, B: dc.float64[M, N], c: dc.int64, S: dc.float64[1]):

    B[:] = np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c

    @dc.map(_[0:N, 0:M])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i, j]
        s = z


sdfg = compute.to_sdfg()

sdfg.save("log_sdfgs/compute_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["array_1"], outputs=["S"])

sdfg.save("log_sdfgs/compute_backward.sdfg")

