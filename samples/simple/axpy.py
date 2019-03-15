#!/usr/bin/env python
from __future__ import print_function

import gc
import argparse
import dace
import numpy as np
import scipy as sp

import os
from timeit import default_timer as timer

N = dace.symbol('N')

A = dace.float64
X = dace.ndarray([N], dtype=dace.float64)
Y = dace.ndarray([N], dtype=dace.float64)


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    parser.add_argument("contender", nargs="?", type=str, default="MKL")
    parser.add_argument(
        "--compile-only",
        default=False,
        action="store_true",
        dest="compile-only")
    args = vars(parser.parse_args())

    N.set(args["N"])
    contender = args["contender"]

    print('Scalar-vector multiplication %d' % (N.get()))

    # Initialize arrays: Randomize A and X, zero Y
    A = dace.float64(np.random.rand())
    X[:] = np.random.rand(N.get()).astype(dace.float64.type)
    Y[:] = np.random.rand(N.get()).astype(dace.float64.type)

    A_regression = np.float64()
    X_regression = np.ndarray([N.get()], dtype=np.float64)
    Y_regression = np.ndarray([N.get()], dtype=np.float64)
    A_regression = A
    X_regression[:] = X[:]
    Y_regression[:] = Y[:]

    if args["compile-only"]:
        dace.compile(axpy, A, X, Y)
    else:
        axpy(A, X, Y)

        c_axpy = sp.linalg.blas.get_blas_funcs(
            'axpy', arrays=(X_regression, Y_regression))
        if dace.Config.get_bool('profiling'):
            dace.timethis('axpy', contender, dace.eval(2 * N), c_axpy,
                          X_regression, Y_regression, N.get(), A_regression)
        else:
            c_axpy(X_regression, Y_regression, N.get(), A_regression)

    diff = np.linalg.norm(Y_regression - Y) / float(dace.eval(N))
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
