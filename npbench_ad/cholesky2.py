import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
k = 1


@dc.program
def triu(A: dc.float64[N, N]):
    B = np.zeros_like(A)
    for i in dc.map[0:N]:
        for j in dc.map[i + k:N]:
            B[i, j] = A[i, j]
    return B


@dc.program
def cholesky2(A: dc.float64[N, N], S: dc.float64[1]):

    A[:] = np.linalg.cholesky(A) + triu(A)

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = cholesky2.to_sdfg()

sdfg.save("log_sdfgs/cholesky2_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/cholesky2_backward.sdfg")

