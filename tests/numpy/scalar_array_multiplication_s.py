#!/usr/bin/env python
from __future__ import print_function

import gc
import argparse
import dace
import numpy as np
import dace.frontend.common as np_frontend

import os
from timeit import default_timer as timer

SDFG = dace.sdfg.SDFG

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
L = dace.symbol('L')
O = dace.symbol('O')

alpha = dace.ndarray([L, O], dtype=dace.float64)
A = dace.ndarray([M, N, K], dtype=dace.float64)
B = dace.ndarray([M, N, K], dtype=dace.float64)

if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=128)
    parser.add_argument("N", type=int, nargs="?", default=128)
    parser.add_argument("K", type=int, nargs="?", default=128)
    parser.add_argument("L", type=int, nargs="?", default=5)
    parser.add_argument("O", type=int, nargs="?", default=10)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    K.set(args["K"])
    L.set(args["L"])
    O.set(args["O"])

    print('Scalar-Array multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize alpha and A
    alpha = np.random.rand(L.get(), O.get()).astype(dace.float64.type)
    A[:] = np.random.rand(M.get(), N.get(), K.get()).astype(dace.float64.type)

    alpha_regression = np.ndarray([L.get(), O.get()], dtype=np.float64)
    alpha_regression[:] = alpha[:]
    A_regression = np.ndarray([M.get(), N.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]

    samul = SDFG(name='samul')
    samul.add_node(
        np_frontend.op_impl.scalar_array_multiplication_s('alpha',
                                                          alpha.shape,
                                                          dace.float64,
                                                          'A',
                                                          A.shape,
                                                          dace.float64,
                                                          False,
                                                          'B',
                                                          B.shape,
                                                          dace.float64,
                                                          alpha_index=[2, 7],
                                                          label='samul'))

    samul(alpha=alpha, A=A, B=B)
    B_regression = alpha_regression[2, 7] * A_regression

    rel_error = (np.linalg.norm((B_regression - B).flatten(), ord=2) /
                 np.linalg.norm(B_regression.flatten(), ord=2))
    print("Relative error:", rel_error)
    print("==== Program end ====")
    exit(0 if rel_error <= 1e-15 else 1)
