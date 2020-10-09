# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program(dace.float64[M, N], dace.float64[N, K], dace.float64[M, K])
def mapreduce_test_2(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, K, N], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(j, k, i):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=2, identity=0)


if __name__ == "__main__":

    M.set(50)
    N.set(20)
    K.set(5)

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = dace.ndarray([M, N], dtype=dace.float64)
    B = dace.ndarray([N, K], dtype=dace.float64)
    C = dace.ndarray([M, K], dtype=dace.float64)
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    mapreduce_test_2(A, B, C)
    np.dot(A_regression, B_regression, C_regression)

    diff = np.linalg.norm(C_regression - C) / np.linalg.norm(C_regression)
    print(C_regression)
    print(C)
    print("Difference:", diff)
    exit(0 if diff <= 1e-10 else 1)
