#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.codegen.targets.memorypool import extend_dace

M = dace.symbol('M')
N = dace.symbol('N')

@dace.program
def operation(A: dace.float64[5, 5], B: dace.float64[5, 5], C: dace.float64[5, 10], D: dace.float64[5, 10]):
    tmp = dace.define_local([5, 5, 5], dtype=A.dtype)
    E = dace.define_local([5,5], dtype=A.dtype)

    @dace.map(_[0:5, 0:5, 0:5])
    def multiplication(i, j, k):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B
    dace.reduce(lambda a, b: a + b, tmp, E, axis=2, identity=0)

    C[:] =  A @ E @ (A @ B) @ (B @ D)
    #ctile[:] = C
    #ctile[:] = A @ E @ (A @ B) @ (B @ D)
    #C[:] = ctile

if __name__ == "__main__":
    extend_dace()
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=5)
    parser.add_argument("N", type=int, nargs="?", default=10)
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

    sdfg(A=A, B=B, C=C, D=D)
    C_regression = np.dot(np.dot(A, np.dot(np.dot(A, B), np.dot(A, B))), np.dot(B,D))

    diff = np.linalg.norm(C_regression - C) / (M.value * N.value)
    print("Difference:", diff, 'C_reg', C_regression, 'C', C)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
