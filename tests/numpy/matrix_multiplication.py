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

A = dace.ndarray([3, 7, 9, M, N], dtype=dace.float64)
B = dace.ndarray([2, 5, 8, 4, N, K], dtype=dace.float64)
C = dace.ndarray([3, 3, M, K], dtype=dace.float64)

if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=128)
    parser.add_argument("N", type=int, nargs="?", default=128)
    parser.add_argument("K", type=int, nargs="?", default=128)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    K.set(args["K"])

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A[1, 2, 3] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[1, 3, 2, 1] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[2, 2] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[1, 2, 3]
    B_regression[:] = B[1, 3, 2, 1]
    C_regression[:] = C[2, 2]

    mmul = SDFG(name='mmul')
    state = mmul.add_state(label='mmul')
    A_node = state.add_array('A', A.shape, dace.float64)
    B_node = state.add_array('B', B.shape, dace.float64)
    C_node = state.add_array('C', C.shape, dace.float64)
    np_frontend.op_impl.matrix_multiplication(state,
                                              A_node,
                                              A_node,
                                              B_node,
                                              B_node,
                                              C_node,
                                              C_node,
                                              A_index=[1, 2, 3],
                                              B_index=[1, 3, 2, 1],
                                              C_index=[2, 2],
                                              label='mmul')

    mmul(A=A, B=B, C=C)
    np.dot(A_regression, B_regression, C_regression)

    rel_error = (np.linalg.norm(C_regression - C[2, 2], ord=2) /
                 np.linalg.norm(C_regression, ord=2))
    print("Relative error:", rel_error)
    print("==== Program end ====")
    exit(0 if rel_error <= 1e-15 else 1)
