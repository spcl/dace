# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program(dace.float64[M, N], dace.float64[N, K], dace.float64[M, K],
              dace.float64[M, K, N])
def mapreduce_test_4(A, B, C, D):
    # Transient variable
    tmp = dace.define_local([M, K, N], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(j, k, i):
        in_A << A[i, k]
        in_B << B[k, j]
        scale >> D[i, j, k]
        out >> tmp[i, j, k]

        out = in_A * in_B
        scale = in_A * 5

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


if __name__ == "__main__":

    M.set(50)
    N.set(20)
    K.set(5)

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = dace.ndarray([M, N], dtype=dace.float64)
    B = dace.ndarray([N, K], dtype=dace.float64)
    C = dace.ndarray([M, K], dtype=dace.float64)
    D = dace.ndarray([M, K, N], dtype=dace.float64)
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)
    D[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    mapreduce_test_4(A, B, C, D)
    np.dot(A_regression, B_regression, C_regression)

    diff = np.linalg.norm(C_regression - C) / (M.get() * K.get())
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)
