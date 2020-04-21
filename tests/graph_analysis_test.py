#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.graph.analysis.live_sets import live_sets

M = dace.symbol('M')
N = dace.symbol('N')

@dace.program
def operation(A: dace.float64[M, M], B: dace.float64[M, M], C: dace.float64[M, N], D: dace.float64[M, N]):
    tmp = dace.define_local([M, M, M], dtype=A.dtype)
    E = dace.define_local([M,M], dtype=A.dtype)

    @dace.map(_[0:M, 0:M, 0:M])
    def multiplication(i, j, k):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B
    dace.reduce(lambda a, b: a + b, tmp, E, axis=2, identity=0)

    C[:] = A @ E @ (A @ B) @ (B @ D)


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=5)
    parser.add_argument("N", type=int, nargs="?", default=100)
    args = vars(parser.parse_args())

    M.value = args["M"]
    N.value = args["N"]

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(M.value, M.value).astype(np.float64)
    B = np.random.rand(M.value, M.value).astype(np.float64)
    D = np.random.rand(M.value, N.value).astype(np.float64)
    C = np.zeros([M.value, N.value], dtype=np.float64)
    C_regression = np.zeros_like(C)

    sdfg = operation.to_sdfg(args)
    live_sets(sdfg=sdfg)
    sdfg(A=A, B=B, C=C, D=D, M=M, N=N)
    C_regression = np.dot(np.dot(A, np.dot(np.dot(A, B), np.dot(A, B))), np.dot(B,D))

    diff = np.linalg.norm(C_regression - C) / (M.value * N.value)
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)