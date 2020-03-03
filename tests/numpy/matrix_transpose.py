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

A = dace.ndarray([L, K, M, N], dtype=dace.float64)
B = dace.ndarray([L, N, M], dtype=dace.float64)

if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=128)
    parser.add_argument("N", type=int, nargs="?", default=128)
    parser.add_argument("L", type=int, nargs="?", default=5)
    parser.add_argument("K", type=int, nargs="?", default=10)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    K.set(args["K"])
    L.set(args["L"])

    print('Matrix transpose %dx%dx' % (M.get(), N.get()))

    # Initialize arrays: Randomize A and B
    A[:] = np.random.rand(L.get(), K.get(), M.get(),
                          N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(L.get(), N.get(), M.get()).astype(dace.float64.type)

    A_regression = np.ndarray(
        [L.get(), K.get(), M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([L.get(), N.get(), M.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]

    mtr = SDFG(name='mtr')
    state = mtr.add_state(label='mtr')
    A_node = state.add_array('A', A.shape, dace.float64)
    B_node = state.add_array('B', B.shape, dace.float64)
    np_frontend.op_impl.matrix_transpose(state,
                                         A_node,
                                         A_node,
                                         B_node,
                                         B_node, [2, 3], [4],
                                         label='mtr')

    mtr(A=A, B=B)
    B_regression[4] = np.transpose(A_regression[2, 3])

    rel_error = (np.linalg.norm((B_regression - B).flatten(), ord=2) /
                 np.linalg.norm(B_regression.flatten(), ord=2))
    print("Relative error:", rel_error)
    print("==== Program end ====")
    exit(0 if rel_error <= 1e-15 else 1)
