import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

@dc.program
def covariance(float_n: dc.float64, data: dc.float64[N, M], S: dc.float64[1]):

    mean = np.mean(data, axis=0)
    # data -= mean
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    @dc.map(_[0:M, 0:M])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << cov[i, j]
        s = z


sdfg = covariance.to_sdfg()

sdfg.save("log_sdfgs/covariance_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/covariance_backward.sdfg")

