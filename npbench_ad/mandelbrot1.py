import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

XN, YN, N = 32, 32, 32


@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dc.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64,
               ymax: dc.float64, maxiter: dc.int64, horizon: dc.float64, S: dc.float64[1]):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.ndarray((XN, ), dtype=np.float64)
    Y = np.ndarray((YN, ), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    # C = X + np.reshape(Y, (YN, 1)) * 1j
    C = np.ndarray((YN, XN), dtype=np.complex128)
    for i, j in dc.map[0:YN, 0:XN]:
        C[i, j] = X[j] + Y[i] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(maxiter):
        I = np.less(np.absolute(Z), horizon)
        N[I] = n
        # np.positive(n, out=N, where=I)
        # for j, k in dace.map[0:YN, 0:XN]:
        #     if I[j, k]:
        #         N[j, k] = n
        # Z[I] = Z[I]**2 + C[I]
        # np.add(np.power(Z, 2, where=I), C, out=Z, where=I)
        for j, k in dc.map[0:YN, 0:XN]:
            if I[j, k]:
                Z[j, k] = Z[j, k]**2 + C[j, k]
    N[N == maxiter - 1] = 0

    @dc.map(_[0:YN, 0:XN])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << N[i, j]
        s = z
    return Z, N



sdfg = mandelbrot.to_sdfg()

sdfg.save("log_sdfgs/mandelbrot_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["xmin"], outputs=["S"])

sdfg.save("log_sdfgs/mandelbrot_backward.sdfg")
