import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32

NR, NM, slab_per_bc = 43, 43, 5


@dc.program
def contour(Ham: dc.complex128[slab_per_bc + 1, NR, NR], int_pts: dc.complex128[32], Y: dc.complex128[NR, NM],
            c: dc.int64, S: dc.float64[1]):

    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        # if NR == NM:
        #     X = np.linalg.inv(Tz)
        # else:
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X[:] = -X
        P0 += X
        P1 += z * X

    @dc.map(_[0:NR, 0:NM])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << P0[i, j]
        s = z


sdfg = contour.to_sdfg()

sdfg.save("log_sdfgs/contour_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["Y"], outputs=["S"])

sdfg.save("log_sdfgs/contour_backward.sdfg")
