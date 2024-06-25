import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M =  32

@dc.program
def symm(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N],
           A: dc.float64[M, M], B: dc.float64[M, N], S: dc.float64[1]):

    temp2 = np.empty((N, ), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << C[i, j]
        s = z



sdfg = symm.to_sdfg()

sdfg.save("log_sdfgs/symm_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/symm_backward.sdfg")

