import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
k = 1


@dc.program
def cholesky(A: dc.float64[N, N], S: dc.float64[1]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = cholesky.to_sdfg()

sdfg.save("log_sdfgs/cholesky_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/cholesky_backward.sdfg")

