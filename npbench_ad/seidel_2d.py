import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M =  32
TSTEPS = 3

@dc.program
def seidel_2d(TSTEPS: dc.int64, A: dc.float64[N, N], S: dc.float64[1]):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                           A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z



sdfg = seidel_2d.to_sdfg()

sdfg.save("log_sdfgs/seidel_2d_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/seidel_2d_backward.sdfg")

