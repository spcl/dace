#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.transformation.interstate.transient_reuse import TransientReuse

M = dace.symbol('M')


@dace.program
def operation(A: dace.float64[M, M], B: dace.float64[M, M], C: dace.float64[M, M]):
    C[:] = A @ B @ A @ B @ A


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=5)
    args = vars(parser.parse_args())

    M.set(args["M"])

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(M.get(), M.get()).astype(np.float64)
    B = np.random.rand(M.get(), M.get()).astype(np.float64)
    C = np.zeros([M.get(), M.get()], dtype=np.float64)
    C_regression = np.zeros_like(C)

    sdfg = operation.to_sdfg()
    sdfg.apply_transformations(TransientReuse)
    sdfg(A=A, B=B, C=C, M=M)

    if dace.Config.get_bool('profiling'):
        dace.timethis('gemm', 'numpy', (2 * M.get() * M.get() * M.get()),
                      np.dot, A, B, C_regression)
    else:
        C_regression = np.dot(np.dot(np.dot(np.dot(A, B), A), B), A)

    diff = np.linalg.norm(C_regression - C) / (M.get() * M.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)