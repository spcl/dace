# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np
from scipy import ndimage

W = dace.symbol('W')
H = dace.symbol('H')
MAXITER = dace.symbol('MAXITER')


@dace.program(dace.float32[H, W], dace.int32)
def jacobi(A, iterations):
    # Transient variable
    tmp = dace.define_local([H, W], dtype=A.dtype)

    @dace.map(_[0:H, 0:W])
    def reset_tmp(y, x):

        out >> tmp[y, x]
        out = dace.float32(0.0)

    for t in range(iterations):

        @dace.map(_[1:H - 1, 1:W - 1])
        def a2b(y, x):
            in_N << A[y - 1, x]
            in_S << A[y + 1, x]
            in_W << A[y, x - 1]
            in_E << A[y, x + 1]
            in_C << A[y, x]
            out >> tmp[y, x]

            out = dace.float32(0.2) * (in_C + in_N + in_S + in_W + in_E)

        # Double buffering
        @dace.map(_[1:H - 1, 1:W - 1])
        def b2a(y, x):
            in_N << tmp[y - 1, x]
            in_S << tmp[y + 1, x]
            in_W << tmp[y, x - 1]
            in_E << tmp[y, x + 1]
            in_C << tmp[y, x]
            out >> A[y, x]

            out = dace.float32(0.2) * (in_C + in_N + in_S + in_W + in_E)


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=12)
    parser.add_argument("H", type=int, nargs="?", default=12)
    parser.add_argument("MAXITER", type=int, nargs="?", default=30)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])
    MAXITER.set(args["MAXITER"])

    print('Jacobi 5-point Stencil %dx%d (%d steps)' % (W.get(), H.get(), MAXITER.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    A[1:H.get() - 1, 1:W.get() - 1] = np.random.rand((H.get() - 2), (W.get() - 2)).astype(dace.float32.type)
    regression = np.ndarray([H.get() - 2, W.get() - 2], dtype=np.float32)
    regression[:] = A[1:H.get() - 1, 1:W.get() - 1]

    #print(A.view(type=np.ndarray))

    #############################################
    # Run DaCe program

    jacobi(A, MAXITER)

    # Regression
    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]], dtype=np.float32)
    for i in range(2 * MAXITER.get()):
        regression = ndimage.convolve(regression, kernel, mode='constant', cval=0.0)

    residual = np.linalg.norm(A[1:H.get() - 1, 1:W.get() - 1] - regression) / (H.get() * W.get())
    print("Residual:", residual)

    #print(A.view(type=np.ndarray))
    #print(regression.view(type=np.ndarray))

    print("==== Program end ====")
    exit(0 if residual <= 0.05 else 1)
