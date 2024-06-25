import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

@dc.program
def mvt(x1: dc.float64[N], x2: dc.float64[N], y_1: dc.float64[N],
           y_2: dc.float64[N], A: dc.float64[N, N], S:dc.float64[1]):

    x1 += A @ y_1
    x2 += y_2 @ A

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << x2[i]
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

sdfg = mvt.to_sdfg()

sdfg.save("log_sdfgs/mvt_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/mvt_backward.sdfg")

