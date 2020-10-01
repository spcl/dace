# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np
from dace.transformation.interstate.transient_reuse import TransientReuse

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

    m = args['M']
    n = args['N']

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(m, m).astype(np.float64)
    B = np.random.rand(m, m).astype(np.float64)
    D = np.random.rand(m, n).astype(np.float64)
    C = np.zeros([m, n], dtype=np.float64)
    C_regression = np.zeros_like(C)

    sdfg = operation.to_sdfg()
    sdfg.apply_transformations(TransientReuse)
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n)

    C_regression = np.dot(np.dot(A, np.dot(np.dot(A, B), np.dot(A, B))), np.dot(B,D))

    diff = np.linalg.norm(C_regression - C) / (m * n)
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
