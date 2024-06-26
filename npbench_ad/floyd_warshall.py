import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32

@dc.program
def floyd_warshall(path: dc.float64[N, N], S: dc.float64[1]):

    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << path[i, j]
        s = z


sdfg = floyd_warshall.to_sdfg()

sdfg.save("log_sdfgs/floyd_warshall_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["path"], outputs=["S"])

sdfg.save("log_sdfgs/floyd_warshall_backward.sdfg")

