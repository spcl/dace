import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

@dc.program
def gemver(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
           u1: dc.float64[N], v1: dc.float64[N], u2: dc.float64[N],
           v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N],
           y: dc.float64[N], z: dc.float64[N], S:dc.float64[1]):

    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << w[i]
        s = z


A = np.ones(shape=[N, N])
u1 = np.ones(shape=[N])
u2 = np.ones(shape=[N])
v1 = np.ones(shape=[N])
v2 = np.ones(shape=[N])
x = np.ones(shape=[N])
y = np.ones(shape=[N])
w = np.ones(shape=[N])
z = np.ones(shape=[N])

sdfg = gemver.to_sdfg()

sdfg.save("log_sdfgs/gemver_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gemver_backward.sdfg")

