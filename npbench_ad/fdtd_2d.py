import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
NX = 32
NY = 32
TMAX = 32

@dc.program
def fdtd_2d(ex: dc.float64[NX, NY], ey: dc.float64[NX, NY],
           hz: dc.float64[NX, NY], _fict_: dc.float64[TMAX], S: dc.float64[1]):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])

    @dc.map(_[0:NX, 0:NY])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << hz[i, j]
        s = z


sdfg = fdtd_2d.to_sdfg()

sdfg.save("log_sdfgs/fdtd_2d_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["ex"], outputs=["S"])

sdfg.save("log_sdfgs/fdtd_2d_backward.sdfg")

