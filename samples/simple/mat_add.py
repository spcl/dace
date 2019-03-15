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

A = dace.ndarray([M, N], dtype=dace.float64)
B = dace.ndarray([M, N], dtype=dace.float64)
C = dace.ndarray([M, N], dtype=dace.float64)


@dace.program(dace.float64[M, N], dace.float64[M, N], dace.float64[M, N])
def mat_add(A, B, C):
    @dace.map(_[0:M, 0:N])
    def addition(i, j):
        in_A << A[i, j]
        in_B << B[i, j]
        out >> C[i, j]

        out = in_A + in_B


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=24)
    parser.add_argument("N", type=int, nargs="?", default=24)
    parser.add_argument("contender", nargs="?", type=str, default="MKL")
    parser.add_argument(
        "--compile-only",
        default=False,
        action="store_true",
        dest="compile-only")
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    contender = args["contender"]

    print('Matrix addition %dx%d' % (M.get(), N.get()))

    # Initialize arrays: Randomize A and B, zero C
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    if args["compile-only"]:
        dace.compile(mat_add, A, B, C)
    else:
        mat_add(A, B, C)

        if dace.Config.get_bool('profiling'):
            dace.timethis('mat_add', contender, dace.eval(2 * N * N * N),
                          np.dot, A_regression, B_regression, C_regression)
        else:
            np.add(A_regression, B_regression, C_regression)

    diff = np.linalg.norm(C_regression - C) / float(dace.eval(M * N))
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
