import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
npt = 3

@dc.program
def azimint_naive(data: dc.float64[N], radius: dc.float64[N], S: dc.float64[1]):


    rmax = np.amax(radius)
    res = np.zeros((npt, ), dtype=np.float64)  # Fix in np.full
    for i in range(npt):

        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))

        on_values = 0
        tmp = np.float64(0)
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values

    @dc.map(_[0:npt])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << res[i]
        s = z


sdfg = azimint_naive.to_sdfg()

sdfg.save("log_sdfgs/azimint_naive_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/azimint_naive_backward.sdfg")

