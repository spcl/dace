# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import (MapReduceFusion, MapFusion, MapWCRFusion)

W = dace.symbol('W')
H = dace.symbol('H')

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
BINS = 256


@dace.program(dace.uint8[H, W], dace.uint32[BINS])
def histogram(A, hist):
    # Declarative version
    tmp = dace.define_local([BINS, H, W], dace.uint32)

    @dace.map(_[0:H, 0:W, 0:BINS])
    def zero_tmp(i, j, b):
        t >> tmp[b, i, j]
        t = 0

    @dace.map(_[0:H, 0:W])
    def compute_declarative(i, j):
        a << A[i, j]
        out >> tmp(1)[:, i, j]
        out[a] = 1

    dace.reduce(lambda a, b: a + b, tmp, hist, axis=(1, 2))


@dace.program(dace.float32[H, W], dace.float32[H, W], dace.float32[1])
def mapreduce_test(A, B, sum):
    tmp = dace.define_local([H, W], dace.float32)

    @dace.map(_[0:H, 0:W])
    def compute_tile(i, j):
        a << A[i, j]
        b >> B[i, j]
        t >> tmp[i, j]

        b = a * 5
        t = a * 5

    sum[:] = dace.reduce(lambda a, b: a + b, tmp, identity=0)


@dace.program(dace.float64[M, N], dace.float64[N, K], dace.float64[M, K])
def mapreduce_test_2(A, B, C):
    # Transient variable
    tmp = dace.define_local([M, K, N], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(j, k, i):
        in_A << A[i, k]
        in_B << B[k, j]
        out >> tmp[i, j, k]

        out = in_A * in_B

    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=2, identity=0)


@dace.program(dace.float32[1, H, 1, W, 1], dace.float32[H, W], dace.float32[1])
def mapreduce_test_3(A, B, sum):
    tmp = dace.define_local([1, H, 1, W, 1], dace.float32)

    @dace.map(_[0:H, 0:W])
    def compute_tile(i, j):
        a << A[0, i, 0, j, 0]
        b >> B[i, j]
        t >> tmp[0, i, 0, j, 0]

        b = a * 5
        t = a * 5

    dace.reduce(lambda a, b: a + b, tmp, sum)


@dace.program(dace.float64[M, N], dace.float64[N, K], dace.float64[M, K], dace.float64[M, K, N])
def mapreduce_test_4(A, B, C, D):
    # Transient variable
    tmp = dace.define_local([M, K, N], dtype=A.dtype)

    @dace.map(_[0:K, 0:N, 0:M])
    def multiplication(j, k, i):
        in_A << A[i, k]
        in_B << B[k, j]
        scale >> D[i, j, k]
        out >> tmp[i, j, k]

        out = in_A * in_B
        scale = in_A * 5

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


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
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations([MapFusion, MapWCRFusion])
    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

    diff = np.linalg.norm(C_regression - C) / (M.get() * N.get())
    print("Difference:", diff)
    assert diff <= 1e-5


def test_basic():
    W.set(128)
    H.set(128)

    print('Map-Reduce Test %dx%d' % (W.get(), H.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    res = dace.ndarray([1], dtype=dace.float32)
    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)
    res[:] = dace.float32(0)

    mapreduce_test(A, B, res)

    diff = np.linalg.norm(5 * A - B) / np.linalg.norm(5 * A)
    diff_res = np.linalg.norm(np.sum(B) - res[0]) / np.linalg.norm(np.sum(B))
    # diff_res = abs((np.sum(B) - res[0])).view(type=np.ndarray)
    print("Difference:", diff, diff_res)
    print("==== Program end ====")
    assert diff <= 1e-5 and diff_res <= 1


def test_mmm():
    M.set(50)
    N.set(20)
    K.set(5)

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = dace.ndarray([M, N], dtype=dace.float64)
    B = dace.ndarray([N, K], dtype=dace.float64)
    C = dace.ndarray([M, K], dtype=dace.float64)
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    mapreduce_test_2(A, B, C)
    np.dot(A_regression, B_regression, C_regression)

    diff = np.linalg.norm(C_regression - C) / np.linalg.norm(C_regression)
    print(C_regression)
    print(C)
    print("Difference:", diff)
    assert diff <= 1e-10


def test_extradims():
    W.set(128)
    H.set(128)

    print('Map-Reduce Test %dx%d' % (W.get(), H.get()))

    A = dace.ndarray([1, H, 1, W, 1], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    res = dace.ndarray([1], dtype=dace.float32)
    A[:] = np.random.rand(1, H.get(), 1, W.get(), 1).astype(dace.float32.type)
    B[:] = dace.float32(0)
    res[:] = dace.float32(0)

    mapreduce_test_3(A, B, res)

    diff = np.linalg.norm(5 * A.reshape((H.get(), W.get())) - B) / (H.get() * W.get())
    diff_res = abs((np.sum(B) - res[0])).view(type=np.ndarray)
    print("Difference:", diff, diff_res)
    print("==== Program end ====")
    assert diff <= 1e-5 and diff_res <= 1


def test_permuted():
    M.set(50)
    N.set(20)
    K.set(5)

    print('Matrix multiplication %dx%dx%d' % (M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = dace.ndarray([M, N], dtype=dace.float64)
    B = dace.ndarray([N, K], dtype=dace.float64)
    C = dace.ndarray([M, K], dtype=dace.float64)
    D = dace.ndarray([M, K, N], dtype=dace.float64)
    A[:] = np.random.rand(M.get(), N.get()).astype(dace.float64.type)
    B[:] = np.random.rand(N.get(), K.get()).astype(dace.float64.type)
    C[:] = dace.float64(0)
    D[:] = dace.float64(0)

    A_regression = np.ndarray([M.get(), N.get()], dtype=np.float64)
    B_regression = np.ndarray([N.get(), K.get()], dtype=np.float64)
    C_regression = np.ndarray([M.get(), K.get()], dtype=np.float64)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    mapreduce_test_4(A, B, C, D)
    np.dot(A_regression, B_regression, C_regression)

    diff = np.linalg.norm(C_regression - C) / (M.get() * K.get())
    print("Difference:", diff)
    assert diff <= 1e-5


def test_histogram():
    W.set(32)
    H.set(32)

    print('Histogram (dec) %dx%d' % (W.get(), H.get()))

    A = np.random.randint(0, BINS, (H.get(), W.get())).astype(np.uint8)
    hist = np.zeros([BINS], dtype=np.uint32)

    sdfg = histogram.to_sdfg()
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations(MapReduceFusion)
    sdfg(A=A, hist=hist, H=H, W=W)

    diff = np.linalg.norm(np.histogram(A, bins=BINS, range=(0, BINS))[0] - hist)
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


def test_mapreduce_twomaps():
    onetest(mapreduce_twomaps)


def test_mapreduce_onemap():
    onetest(mapreduce_onemap)


if __name__ == "__main__":
    test_basic()
    test_mmm()
    test_extradims()
    test_permuted()
    test_histogram()
    test_mapreduce_onemap()
    test_mapreduce_twomaps()
