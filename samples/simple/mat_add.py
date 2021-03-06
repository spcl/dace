# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import gc
import argparse
import dace
import numpy as np

import os
from timeit import default_timer as timer

M = dace.symbol('M')
N = dace.symbol('N')


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
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])

    print('Matrix addition %dx%d' % (M.get(), N.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C = np.zeros_like(A)
    C_regression = np.copy(C)

    mat_add(A, B, C)
    np.add(A, B, C_regression)

    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
