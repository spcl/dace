import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
N = 32
M = 32

@dc.program
def correlation(float_n: dc.float64, data: dc.float64[N, M], S: dc.float64[1]):

    mean = np.mean(data, axis=0)
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.mean(np.subtract(data, mean)**2, axis=0))
    stddev[stddev <= 0.1] = 1.0
    # data -= mean
    np.subtract(data, mean, out=data)
    # data /= np.sqrt(float_n) * stddev
    np.divide(data, np.sqrt(float_n) * stddev, out=data)
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        # corr[i, i+1:M] = np.transpose(data[:, i+1:M]) @ data[:, i]
        corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
        corr[i + 1:M, i] = corr[i, i + 1:M]

    @dc.map(_[0:M, 0:M])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << corr[i, j]
        s = z


sdfg = correlation.to_sdfg()

sdfg.save("log_sdfgs/correlation_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/correlation_backward.sdfg")

