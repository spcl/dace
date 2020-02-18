#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from scipy import ndimage

N = dace.symbol('N')
N.set(20)
KERNEL = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], dtype=np.float32)


@dace.program(dace.float32[N, N], dace.float32[N, N])
def stencil3x3(A, B):
    @dace.map(_[1:N - 1, 1:N - 1])
    def a2b(y, x):
        input << A[y - 1:y + 2, x - 1:x + 2]
        out >> B[y, x]
        out = (kernel[0, 0] * input[0, 0] + kernel[0, 1] * input[0, 1] +
               kernel[0, 2] * input[0, 2] + kernel[1, 0] * input[1, 0] +
               kernel[1, 1] * input[1, 1] + kernel[1, 2] * input[1, 2] +
               kernel[2, 0] * input[2, 0] + kernel[2, 1] * input[2, 1] +
               kernel[2, 2] * input[2, 2])


if __name__ == "__main__":
    print("==== Program start ====")

    print('Conv2D %dx%d' % (N.get(), N.get()))

    A = dace.ndarray([N, N], dtype=dace.float32)
    B = dace.ndarray([N, N], dtype=dace.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    B[:] = dace.float32(0)
    A[1:N.get() - 1, 1:N.get() - 1] = np.random.rand(
        (N.get() - 2), (N.get() - 2)).astype(dace.float32.type)
    regression = np.ndarray([N.get() - 2, N.get() - 2], dtype=np.float32)
    regression[:] = A[1:N.get() - 1, 1:N.get() - 1]

    #print(A.view(type=np.ndarray))

    #############################################
    # Run DaCe program

    sdfg = stencil3x3.to_sdfg()
    sdfg.add_constant('kernel', KERNEL)
    sdfg(A=A, B=B, N=N)

    # Regression
    regression = ndimage.convolve(
        regression, KERNEL, mode='constant', cval=0.0)

    residual = np.linalg.norm(B[1:N.get() - 1, 1:N.get() - 1] - regression) / (
        (N.get() - 2)**2)
    print("Residual:", residual)

    #print(A.view(type=np.ndarray))
    #print(regression.view(type=np.ndarray))

    print("==== Program end ====")
    exit(0 if residual <= 0.05 else 1)
