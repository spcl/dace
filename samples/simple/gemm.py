# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


@dace.program(dace.float64[M, K], dace.float64[K, N], dace.float64[M, N])
def gemm(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, N, K], dtype=A.dtype)

    @dace.map(_[0:M, 0:N, 0:K])
    def multiplication(i, j, k):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=24)
    parser.add_argument("K", type=int, nargs="?", default=24)
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    M.set(args["M"])
    K.set(args["K"])
    N.set(args["N"])

    print('Matrix multiplication %dx%dx%d' % (M.get(), K.get(), N.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(M.get(), K.get()).astype(np.float64)
    B = np.random.rand(K.get(), N.get()).astype(np.float64)
    C = np.zeros([M.get(), N.get()], dtype=np.float64)
    C_regression = np.zeros_like(C)

    gemm(A, B, C)

    if dace.Config.get_bool('profiling'):
        dace.timethis('gemm', 'numpy', (2 * M.get() * K.get() * N.get()),
                      np.dot, A, B, C_regression)
    else:
        np.dot(A, B, C_regression)

    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
