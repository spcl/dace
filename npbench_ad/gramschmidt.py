import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

@dc.program
def gramschmidt(A: dc.float64[M, N], S:dc.float64[1]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i]
        s = z

sdfg = gramschmidt.to_sdfg()

sdfg.save("log_sdfgs/gramschmidt_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gramschmidt_backward.sdfg")

