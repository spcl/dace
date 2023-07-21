# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


@dc.program
def indirection_scalar(A: dc.float64[10]):
    i = 0
    return A[i]


def test_indirection_scalar():
    A = np.random.randn(10).astype(np.float64)
    res = indirection_scalar(A)[0]
    assert (res == A[0])


@dc.program
def indirection_scalar_assign(A: dc.float64[10]):
    i = 2
    A[i] = 5
    return A[i]


def test_indirection_scalar_assign():
    A = np.random.randn(10).astype(np.float64)
    res = indirection_scalar_assign(A)[0]
    assert (res == 5)


@dc.program
def indirection_scalar_augassign(A: dc.float64[10]):
    i = 2
    j = 3
    A[i] += A[j]
    return A[i]


def test_indirection_scalar_augassign():
    A = np.random.randn(10).astype(np.float64)
    res = indirection_scalar_augassign(np.copy(A))[0]
    assert (np.allclose(res, A[2] + A[3]))


@dc.program
def indirection_scalar_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        a = x[i]
        B[i] = A[a]
    return B


def test_indirection_scalar_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_scalar_nsdfg(A, x)
    assert (np.allclose(res, A[x]))


@dc.program
def indirection_scalar2_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    for i in dc.map[0:A.shape[0]]:
        a = x[i]
        B[i] = A[a]
        B[i] = A[a]
    return B


def test_indirection_scalar2_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_scalar2_nsdfg(A, x)
    assert (np.allclose(res, A[x]))


@dc.program
def indirection_scalar_assign_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        a = x[i]
        B[a] = A[a]
    return B


def test_indirection_scalar_assign_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_scalar_assign_nsdfg(A, x)
    assert (np.allclose(res[x], A[x]))


@dc.program
def indirection_scalar_augassign_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.full_like(A, 5)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        a = x[i]
        B[a] += A[a]
    return B


def test_indirection_scalar_augassign_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_scalar_augassign_nsdfg(A, x)
    assert (np.allclose(res, indirection_scalar_augassign_nsdfg.f(A, x)))


@dc.program
def indirection_scalar_multi(A: dc.float64[10, 10]):
    i = 0
    # TODO: This doesn't work with j = -1
    j = 9
    return A[i, j]


def test_indirection_scalar_multi():
    A = np.random.randn(10, 10).astype(np.float64)
    res = indirection_scalar_multi(A)[0]
    assert (res == A[0, 9])


@dc.program
def indirection_scalar_multi_nsdfg(A: dc.float64[10, 10], x: dc.int32[10, 10], y: dc.int32[10, 10]):
    B = np.empty_like(A)
    for i, j in dc.map[0:10, 0:10]:
        i0 = x[i, j]
        i1 = y[i, j]
        B[i, j] = A[i0, i1]
    return B


def test_indirection_scalar_multi_nsdfg():
    A = np.random.randn(10, 10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    y = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    res = indirection_scalar_multi_nsdfg(A, x, y)
    assert (np.allclose(res, A[x, y]))


@dc.program
def indirection_scalar_op(A: dc.float64[10]):
    i = 0
    return A[i + 5]


def test_indirection_scalar_op():
    A = np.random.randn(10).astype(np.float64)
    res = indirection_scalar_op(A)[0]
    assert (res == A[5])


@dc.program
def indirection_scalar_op_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        a = x[i]
        B[i] = A[a + 5]
    return B


def test_indirection_scalar_op_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 5, size=(10, ), dtype=np.int32)
    res = indirection_scalar_op_nsdfg(A, x)
    assert (np.allclose(res, A[x + 5]))


@dc.program
def indirection_scalar_range(A: dc.float64[10]):
    i = 1
    # TODO: This doesn't work with j = -1
    j = 9
    return np.sum(A[i:j])


def test_indirection_scalar_range():
    A = np.random.randn(10).astype(np.float64)
    res = indirection_scalar_range(A)[0]
    assert (np.allclose(res, np.sum(A[1:9])))


def test_indirection_scalar_range_nsdfg():

    @dc.program
    def indirection_scalar_range_nsdfg(A: dc.float64[10], x: dc.int32[11]):
        B = np.empty_like(A)
        for i in dc.map[0:A.shape[0]]:
            i0 = min(x[i], x[i + 1])
            i1 = max(x[i], x[i + 1]) + 1
            B[i] = np.sum(A[i0:i1])
        return B

    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 9, size=(11, ), dtype=np.int32)
    res = indirection_scalar_range_nsdfg(A, x)
    for i in range(10):
        i0 = min(x[i], x[i + 1])
        i1 = max(x[i], x[i + 1]) + 1
        assert (np.allclose(res[i], np.sum(A[i0:i1])))


@dc.program
def indirection_array(A: dc.float64[10], x: dc.int32[10]):
    i = 0
    return A[x[i]]


def test_indirection_array():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_array(A, x)[0]
    assert (res == A[x[0]])


@dc.program
def indirection_array_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        B[i] = A[x[i]]
    return B


def test_indirection_array_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_array_nsdfg(A, x)
    assert (np.allclose(res, A[x]))


@dc.program
def indirection_array_multi(A: dc.float64[10, 10], x: dc.int32[10]):
    i = 0
    j = 9
    return A[x[i], x[j]]


def test_indirection_array_multi():
    A = np.random.randn(10, 10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_array_multi(A, x)[0]
    assert (res == A[x[0], x[9]])


@dc.program
def indirection_array_multi_nsdfg(A: dc.float64[10, 10], x: dc.int32[10, 10], y: dc.int32[10, 10]):
    B = np.empty_like(A)
    for i, j in dc.map[0:10, 0:10]:
        B[i, j] = A[x[i, j], y[i, j]]
    return B


def test_indirection_array_multi_nsdfg():
    A = np.random.randn(10, 10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    y = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    res = indirection_array_multi_nsdfg(A, x, y)
    assert (np.allclose(res, A[x, y]))


@dc.program
def indirection_array_op(A: dc.float64[10], x: dc.int32[10]):
    i = 0
    return A[x[i] + 2]


def test_indirection_array_op():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 8, size=(10, ), dtype=np.int32)
    res = indirection_array_op(A, x)[0]
    assert (res == A[x[0] + 2])


@dc.program
def indirection_array_op_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        B[i] = A[x[i] + 5]
    return B


def test_indirection_array_op_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 5, size=(10, ), dtype=np.int32)
    res = indirection_array_op_nsdfg(A, x)
    assert (np.allclose(res, A[x + 5]))


@dc.program
def indirection_array_range(A: dc.float64[10], x: dc.int32[10]):
    i = 5
    return np.sum(A[x[i]:x[i] + 1])


def test_indirection_array_range():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 9, size=(10, ), dtype=np.int32)
    res = indirection_array_range(A, x)[0]
    assert (np.allclose(res, np.sum(A[x[5]:x[5] + 1])))


@dc.program
def indirection_array_range_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        B[i] = np.sum(A[x[i]:x[i] + 1])
    return B


def test_indirection_array_range_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 9, size=(10, ), dtype=np.int32)
    res = indirection_array_range_nsdfg(A, x)
    assert (np.allclose(res, A[x]))


@dc.program
def indirection_array_nested(A: dc.float64[10], x: dc.int32[10]):
    i = 0
    return A[x[x[i]]]


def test_indirection_array_nested():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_array_nested(A, x)[0]
    assert (res == A[x[x[0]]])


@dc.program
def indirection_array_nested_nsdfg(A: dc.float64[10], x: dc.int32[10]):
    B = np.empty_like(A)
    # TODO: This doesn't work with 0:A.shape[0]
    for i in dc.map[0:10]:
        B[i] = A[x[x[i]]]
    return B


def test_indirection_array_nested_nsdfg():
    A = np.random.randn(10).astype(np.float64)
    x = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    res = indirection_array_nested_nsdfg(A, x)
    assert (np.allclose(res, A[x[x]]))


@dc.program
def spmv(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


def test_spmv():

    with dc.config.set_temporary('compiler', 'allow_view_arguments', value=True):

        M, N, nnz = 1000, 1000, 100

        from numpy.random import default_rng
        rng = default_rng(42)

        x = rng.random((N, ))

        from scipy.sparse import random

        matrix = random(M, N, density=nnz / (M * N), format='csr', dtype=np.float64, random_state=rng)
        rows = np.uint32(matrix.indptr)
        cols = np.uint32(matrix.indices)
        vals = matrix.data

        y = spmv(rows, cols, vals, x)
        ref = matrix @ x

        assert (np.allclose(y, ref))


def test_indirection_size_1():

    def compute_index(scal: dc.int32[5]):
        result = 0
        with dace.tasklet:
            s << scal
            r >> result
            r = s[1] + 1 - 1
        return result

    @dc.program
    def tester(a: dc.float64[1, 2, 3], scal: dc.int32[5]):
        ind = compute_index(scal)
        a[0, ind, 0] = 1

    arr = np.random.rand(1, 2, 3)
    scal = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    tester(arr, scal)
    assert arr[0, 1, 0] == 1


if __name__ == "__main__":
    test_indirection_scalar()
    test_indirection_scalar_assign()
    test_indirection_scalar_augassign()
    test_indirection_scalar_nsdfg()
    test_indirection_scalar2_nsdfg()
    test_indirection_scalar_assign_nsdfg()
    test_indirection_scalar_augassign_nsdfg()
    test_indirection_scalar_multi()
    test_indirection_scalar_multi_nsdfg()
    test_indirection_scalar_op()
    test_indirection_scalar_op_nsdfg()
    test_indirection_scalar_range()
    test_indirection_scalar_range_nsdfg()
    test_indirection_array()
    test_indirection_array_nsdfg()
    test_indirection_array_multi()
    test_indirection_array_multi_nsdfg()
    test_indirection_array_op()
    test_indirection_array_op_nsdfg()
    test_indirection_array_range()
    test_indirection_array_range_nsdfg()
    test_indirection_array_nested()
    test_indirection_array_nested_nsdfg()
    test_spmv()
    test_indirection_size_1()
