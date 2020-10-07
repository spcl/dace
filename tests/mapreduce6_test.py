# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np
from dace.transformation.dataflow import MapFusion, MapWCRFusion

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program(dace.float64[M, K], dace.float64[K, N], dace.float64[M, N])
def mapreduce_twomaps(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, N, K], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(k, j, i):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    @dace.mapscope
    def summation_outer(i: _[0:M], j: _[0:N]):
        @dace.map
        def summation_inner(k: _[0:K]):
            ti << tmp[i, j, k]
            out_C >> C(1, lambda a, b: a + b)[i, j]
            out_C = ti


@dace.program(dace.float64[M, K], dace.float64[K, N], dace.float64[M, N])
def mapreduce_onemap(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, N, K], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(k, j, i):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    @dace.map
    def summation_outer(i: _[0:M], k: _[0:K], j: _[0:N]):
        ti << tmp[i, j, k]
        out_C >> C(1, lambda a, b: a + b)[i, j]
        out_C = ti


def onetest(program):
    M.set(50)
    N.set(20)
    K.set(5)

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    A = np.random.rand(M.get(), K.get())
    B = np.random.rand(K.get(), N.get())
    C = np.zeros([M.get(), N.get()], np.float64)
    C_regression = A @ B

    sdfg = program.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations([MapFusion, MapWCRFusion])
    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    assert diff <= 1e-5


def test_mapreduce_twomaps():
    onetest(mapreduce_twomaps)


def test_mapreduce_onemap():
    onetest(mapreduce_onemap)


if __name__ == "__main__":
    test_mapreduce_onemap()
    test_mapreduce_twomaps()
