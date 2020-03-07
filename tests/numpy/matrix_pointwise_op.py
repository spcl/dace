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

A = dace.ndarray([3, 7, 9, M, N], dtype=dace.float64)
B = dace.ndarray([2, 5, 8, 4, M, N], dtype=dace.float64)
C = dace.ndarray([3, 3, 1], dtype=dace.float64)

if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=128)
    parser.add_argument("N", type=int, nargs="?", default=128)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])

    print('Matrix point-wise op %dx%d' % (M.get(), N.get()))

    # Initialize arrays: Randomize A and B, zero C
    A[1, 2, 3] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[1, 3, 2, 1] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    C[2, 2, 0] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    A_regression[:] = A[1, 2, 3]
    B_regression[:] = B[1, 3, 2, 1]
    C_regression = C[2, 2, 0]

    mpwop = SDFG(name='mpwop')
    state = mpwop.add_state(label='mpwop')
    A_node = state.add_array('A', A.shape, dace.float64)
    B_node = state.add_array('B', B.shape, dace.float64)
    C_node = state.add_array('C', C.shape, dace.float64)
    np_frontend.op_impl.matrix_pointwise_op(state,
                                            A_node,
                                            A_node,
                                            B_node,
                                            B_node,
                                            C_node,
                                            C_node,
                                            op='*',
                                            reduce=True,
                                            reduce_op='+',
                                            A_index=[1, 2, 3],
                                            B_index=[1, 3, 2, 1],
                                            C_index=[2, 2, 0],
                                            label='mpwop')

    mpwop(A=A, B=B, C=C)
    C_regression = np.dot(A_regression.flatten(), B_regression.flatten())

    if C_regression != 0.0:
        rel_error = np.abs(C_regression - C[2, 2, 0]) / np.abs(C_regression)
    else:
        rel_error = np.abs(C_regression - C[2, 2, 0])
    print("Relative error:", rel_error)
    print("==== Program end ====")
    exit(0 if rel_error <= 1e-15 else 1)
