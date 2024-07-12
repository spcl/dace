import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

W = 32
H = 32


@dc.program
def deriche(alpha: dc.float64, imgIn: dc.float64[W, H], S: dc.float64[1]):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = k
    a5 = k
    a2 = k * np.exp(-alpha) * (alpha - 1.0)
    a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = k * np.exp(-alpha) * (alpha + 1.0)
    a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = -k * np.exp(-2.0 * alpha)
    a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0**(-alpha)
    b2 = -np.exp(-2.0 * alpha)

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] + b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] + b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut = (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] + b1 * y1[i - 1, :] + b2 * y1[i - 2, :])

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] + b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = (y1 + y2)

    @dc.map(_[0:W, 0:H])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << imgOut[i, j]
        s = z


sdfg = deriche.to_sdfg()

sdfg.save("log_sdfgs/deriche_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["imgIn"], outputs=["S"])

sdfg.save("log_sdfgs/deriche_backward.sdfg")
