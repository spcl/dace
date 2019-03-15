#!/usr/bin/env python
from __future__ import print_function

import gc
import argparse
import dace
import numpy as np

import os
from timeit import default_timer as timer

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

A = dace.ndarray([M, N], dtype=dace.float64)
B = dace.ndarray([N, K], dtype=dace.float64)
C = dace.ndarray([M, K], dtype=dace.float64)


@dace.program(dace.float64[M, N], dace.float64[N, K], dace.float64[M, K])
def gemm(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, K, N], dtype=A.dtype)

    @dace.map(_[0:M, 0:K, 0:N])
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
    parser.add_argument("N", type=int, nargs="?", default=24)
    parser.add_argument("K", type=int, nargs="?", default=24)
    parser.add_argument("contender", nargs="?", type=str, default="MKL")
    parser.add_argument(
        "--compile-only",
        default=False,
        action="store_true",
        dest="compile-only")
    parser.add_argument("--sdfg", type=str, nargs="?", default=None)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    K.set(args["K"])
    contender = args["contender"]

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    if args["sdfg"] is not None:
        sdfg = dace.SDFG.from_file(args["sdfg"])
        gemmfunc = dace.compile(sdfg)
    else:
        gemmfunc = dace.compile(gemm, A, B, C)

    if not args["compile-only"]:
        gemmfunc(A, B, C)

    if dace.Config.get_bool('profiling'):
        dace.timethis('gemm', contender, dace.eval(2 * N * N * N), np.dot,
                      A_regression, B_regression, C_regression)
    else:
        np.dot(A_regression, B_regression, C_regression)

    #print(C.view(type=np.ndarray))

    diff = np.linalg.norm(C_regression - C) / float(dace.eval(M * K))
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
