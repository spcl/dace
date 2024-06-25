import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
    
@dc.program
def lu(A: dc.float64[N, N], S: dc.float64[1]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j1 in range(i, N):
            A[i, j1] -= A[i, :i] @ A[:i, j1]
    
    S[0] = np.sum(A)



sdfg = lu.to_sdfg()

sdfg.save("log_sdfgs/lu_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/lu_backward.sdfg")

