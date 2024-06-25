import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
    
@dc.program
def ludcmp(A: dc.float64[N, N], b: dc.float64[N], S: dc.float64[1]):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(N):
        for j3 in range(i):
            A[i, j3] -= A[i, :j3] @ A[:j3, j3]
            A[i, j3] /= A[j3, j3]
        for j2 in range(i, N):
            A[i, j2] -= A[i, :i] @ A[:i, j2]
    for i1 in range(N):
        y[i1] = b[i1] - A[i1, :i1] @ y[:i1]
    for i2 in range(N - 1, -1, -1):
        x[i2] = (y[i2] - A[i2, i2 + 1:] @ x[i2 + 1:]) / A[i2, i2]
    
    S[0] = np.sum(x)



sdfg = ludcmp.to_sdfg()

sdfg.save("log_sdfgs/ludcmp_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/ludcmp_backward.sdfg")

