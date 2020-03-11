#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.transformation.interstate.transient_reuse import TransientReuse

M = dace.symbol('M')
N = dace.symbol('N')

@dace.program
def operation(A: dace.float64[5, 5], B: dace.float64[5, 5], C: dace.float64[5, 100], D: dace.float64[5, 100]):
    C[:] = A @ (A @ B) @ (A @ B) @ (B @ D)



if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=5)
    parser.add_argument("N", type=int, nargs="?", default=100)
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(5, 5).astype(np.float64)
    B = np.random.rand(5, 5).astype(np.float64)
    D = np.random.rand(5, 100).astype(np.float64)
    C = np.zeros([5, 100], dtype=np.float64)
    C_regression = np.zeros_like(C)

    sdfg = operation.to_sdfg()
    sdfg.apply_transformations(TransientReuse)
    sdfg(A=A, B=B, C=C, D=D)

    if dace.Config.get_bool('profiling'):
        dace.timethis('gemm', 'numpy', (2 * M.get() * M.get() * M.get()),
                      np.dot, A, B, C_regression)
    else:
        C_regression = np.dot(np.dot(A, np.dot(np.dot(A, B), np.dot(A, B))), np.dot(B,D))

    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)