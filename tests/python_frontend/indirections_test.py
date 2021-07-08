# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


@dc.program
def indirection_scalar(A: dc.float32[10]):
    i = 0
    return A[i]


def test_indirection_scalar():
    A = np.random.randn(10).astype(np.float32)
    res = indirection_scalar(A)[0]
    assert (res == A[0])


@dc.program
def indirection_scalar_multi(A: dc.float32[10, 10]):
    i = 0
    # TODO: This doesn't work with j = -1
    j = 9
    return A[i, j]


def test_indirection_scalar_multi():
    A = np.random.randn(10, 10).astype(np.float32)
    res = indirection_scalar_multi(A)[0]
    assert (res == A[0, 9])


@dc.program
def indirection_scalar_op(A: dc.float32[10]):
    i = 0
    return A[i+5]


def test_indirection_scalar_op():
    A = np.random.randn(10).astype(np.float32)
    res = indirection_scalar_op(A)[0]
    assert (res == A[5])


@dc.program
def indirection_scalar_range(A: dc.float32[10]):
    i = 1
    # TODO: This doesn't work with j = -1
    j = 9
    return np.sum(A[i:j])


def test_indirection_scalar_range():
    A = np.random.randn(10).astype(np.float32)
    res = indirection_scalar_range(A)[0]
    assert (np.allclose(res, np.sum(A[1:9])))


@dc.program
def indirection_array(A: dc.float32[10], x: dc.int32[10]):
    i = 0
    return A[x[i]]


def test_indirection_array():
    A = np.random.randn(10).astype(np.float32)
    x = np.random.randint(0, 10, size=(10,), dtype=np.int32)
    res = indirection_array(A, x)[0]
    assert (res == A[x[0]])


@dc.program
def indirection_array_multi(A: dc.float32[10, 10], x: dc.int32[10]):
    i = 0
    j = 9
    return A[x[i], x[j]]


def test_indirection_array_multi():
    A = np.random.randn(10, 10).astype(np.float32)
    x = np.random.randint(0, 10, size=(10,), dtype=np.int32)
    res = indirection_array_multi(A, x)[0]
    assert (res == A[x[0], x[9]])


@dc.program
def indirection_array_op(A: dc.float32[10], x: dc.int32[10]):
    i = 0
    return A[x[i] + 2]


def test_indirection_array_op():
    A = np.random.randn(10).astype(np.float32)
    x = np.random.randint(0, 8, size=(10,), dtype=np.int32)
    res = indirection_array_op(A, x)[0]
    assert (res == A[x[0] + 2])


@dc.program
def indirection_array_range(A: dc.float32[10], x: dc.int32[10]):
    i = 5
    return np.sum(A[x[i]:x[i]+1])


def test_indirection_array_range():
    A = np.random.randn(10).astype(np.float32)
    x = np.random.randint(0, 9, size=(10,), dtype=np.int32)
    res = indirection_array_range(A, x)[0]
    assert (np.allclose(res, np.sum(A[x[5]:x[5]+1])))


@dc.program
def indirection_array_nested(A: dc.float32[10], x: dc.int32[10]):
    i = 0
    return A[x[x[i]]]


def test_indirection_array_nested():
    A = np.random.randn(10).astype(np.float32)
    x = np.random.randint(0, 10, size=(10,), dtype=np.int32)
    res = indirection_array_nested(A, x)[0]
    assert (res == A[x[x[0]]])


@dc.program
def spmv(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


def test_spmv():

    M, N, nnz = 1000, 1000, 100

    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    from scipy.sparse import random

    matrix = random(M,
                    N,
                    density=nnz / (M * N),
                    format='csr',
                    dtype=np.float64,
                    random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data

    y = spmv(rows, cols, vals, x)
    ref = matrix @ x

    assert (np.allclose(y, ref))


if __name__ == "__main__":
    test_indirection_scalar()
    test_indirection_scalar_multi()
    test_indirection_scalar_op()
    test_indirection_scalar_range()
    test_indirection_array()
    test_indirection_array_multi()
    test_indirection_array_op()
    test_indirection_array_range()
    test_indirection_array_nested()
    test_spmv()
