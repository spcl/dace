#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.transformation.dataflow.transient_reuse import TransientReuse

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


@dace.program
def operation(A: dace.float64[M, M], B: dace.float64[M, M], C: dace.float64[M, M]):
    C[:] = (((A @ B) @ A) @ B) @ A


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=5)
    parser.add_argument("K", type=int, nargs="?", default=5)
    parser.add_argument("N", type=int, nargs="?", default=5)
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

    sdfg = operation.to_sdfg()
    sdfg.apply_transformations(TransientReuse)
    sdfg(A=A, B=B, C=C, M=M)
    if dace.Config.get_bool('profiling'):
        dace.timethis('gemm', 'numpy', (2 * M.get() * K.get() * N.get()),
                      np.dot, A, B, C_regression)
    else:
        C_regression = np.dot(np.dot(np.dot(np.dot(A, B), A), B), A)
    print(C)
    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)