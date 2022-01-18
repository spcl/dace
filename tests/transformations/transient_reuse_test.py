# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    E = dace.define_local([M, M], dtype=A.dtype)

    @dace.map(_[0:M, 0:M, 0:M])
    def multiplication(i, j, k):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    dace.reduce(lambda a, b: a + b, tmp, E, axis=2, identity=0)

    C[:] = A @ E @ (A @ B) @ (B @ D)


def test_reuse():
    m = 5
    n = 100

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(m, m).astype(np.float64)
    B = np.random.rand(m, m).astype(np.float64)
    D = np.random.rand(m, n).astype(np.float64)
    C = np.zeros([m, n], dtype=np.float64)
    C_regression = np.zeros_like(C)

    sdfg = operation.to_sdfg()
    assert sdfg.apply_transformations(TransientReuse) == 1
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n)

    C_regression = np.dot(np.dot(A, np.dot(np.dot(A, B), np.dot(A, B))), np.dot(B, D))

    diff = np.linalg.norm(C_regression - C) / (m * n)
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_reuse()
