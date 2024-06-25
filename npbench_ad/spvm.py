import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M, N, nnz = 32, 32, 5

@dc.program
def spvm(A_row: dc.uint32[M + 1], A_col: dc.uint32[nnz],
         A_val: dc.float64[nnz], x: dc.float64[N], S: dc.float64[1]):

    # y = np.empty(A_row.size - 1, A_val.dtype)
    y = np.empty(M, A_val.dtype)

    # for i in range(A_row.size - 1):
    for i in range(M):
        start = dc.define_local_scalar(dc.uint32)
        stop = dc.define_local_scalar(dc.uint32)
        start = A_row[i]
        stop = A_row[i + 1]
        # cols = A_col[A_row[i]:A_row[i + 1]]
        # vals = A_val[A_row[i]:A_row[i + 1]]
        cols = A_col[start:stop]
        vals = A_val[start:stop]
        y[i] = vals @ x[cols]

    S[0] = np.sum(y)

    return y


sdfg = spvm.to_sdfg()

sdfg.save("log_sdfgs/spvm_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A_row"], outputs=["S"])

sdfg.save("log_sdfgs/spvm_backward.sdfg")

