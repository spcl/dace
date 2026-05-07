# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np

W = dp.symbol('W')


@dp.program
def subarray(A, B):

    @dp.map(_[0:W])
    def subarrays(i):
        a << A[:, i, i, i]
        a2 << A(1)[i, i, i, :]
        b >> B[i, :, i, i]
        b[i] = a[i] + a2[i]


def test():
    W = 3

    A = dp.ndarray([W, W, W, W])
    B = dp.ndarray([W, W, W, W])

    A[:] = np.mgrid[0:W, 0:W, 0:W]
    for i in range(W):
        A[i, :] += 10 * (i + 1)
    B[:] = dp.float32(0.0)

    subarray(A, B, W=W)


if __name__ == "__main__":
    test()
