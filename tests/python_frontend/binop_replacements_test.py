import dace
import numpy as np


N = dace.symbol('N')
N.set(10)
M = dace.symbol('M')
M.set(5)


@dace.program
def array_array(A: dace.int32[N], B:dace.int32[N]):
    return A + B


def test_array_array():
    A = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    B = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    C = array_array(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def array_array1(A: dace.int32[N], B:dace.int32[1]):
    return A + B


def test_array_array1():
    A = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    B = np.random.randint(10, size=(1, ), dtype=np.int32)
    C = array_array1(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def array1_array(A: dace.int32[1], B:dace.int32[N]):
    return A + B


def test_array1_array():
    A = np.random.randint(10, size=(1, ), dtype=np.int32)
    B = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    C = array1_array(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def array1_array1(A: dace.int32[1], B:dace.int32[1]):
    return A + B


def test_array1_array1():
    A = np.random.randint(10, size=(1, ), dtype=np.int32)
    B = np.random.randint(10, size=(1, ), dtype=np.int32)
    C = array1_array1(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def array_scalar(A: dace.int32[N], B:dace.int32):
    return A + B


def test_array_scalar():
    A = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    B = np.random.randint(10, size=(1, ), dtype=np.int32)[0]
    C = array_scalar(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def scalar_array(A: dace.int32, B:dace.int32[N]):
    return A + B


def test_scalar_array():
    A = np.random.randint(10, size=(1, ), dtype=np.int32)[0]
    B = np.random.randint(10, size=(N.get(), ), dtype=np.int32)
    C = scalar_array(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def array_num(A: dace.int64[N]):
    return A + 5


def test_array_num():
    A = np.random.randint(10, size=(N.get(), ), dtype=np.int64)
    B = array_num(A)
    assert(np.array_equal(A + 5, B))


@dace.program
def num_array(A: dace.int64[N]):
    return 5 + A


def test_num_array():
    A = np.random.randint(10, size=(N.get(), ), dtype=np.int64)
    B = array_num(A)
    assert(np.array_equal(5 + A, B))


@dace.program
def array_bool(A: dace.bool[N]):
    return A and True


def test_array_bool():
    A = np.random.randint(0, high=2, size=(N.get(), ), dtype=np.bool)
    B = array_bool(A)
    assert(np.array_equal(np.logical_and(A, True), B))


@dace.program
def bool_array(A: dace.bool[N]):
    return True and A


def test_bool_array():
    A = np.random.randint(0, high=2, size=(N.get(), ), dtype=np.bool)
    B = bool_array(A)
    assert(np.array_equal(np.logical_and(True, A), B))


@dace.program
def array_sym(A: dace.int64[N]):
    return A + N


def test_array_sym():
    A = np.random.randint(0, high=2, size=(N.get(), ), dtype=np.int64)
    B = array_sym(A)
    assert(np.array_equal(A + N.get(), B))


@dace.program
def sym_array(A: dace.int64[N]):
    return A + N


def test_sym_array():
    A = np.random.randint(0, high=2, size=(N.get(), ), dtype=np.int64)
    B = sym_array(A)
    assert(np.array_equal(N.get() + A, B))


@dace.program
def scal_scal(A: dace.int32, B:dace.int32):
    return A + B


def test_scal_scal():
    A = np.random.randint(10, size=(1, ), dtype=np.int32)[0]
    B = np.random.randint(10, size=(1, ), dtype=np.int32)[0]
    C = scal_scal(A, B)
    assert(np.array_equal(A + B, C[0]))


@dace.program
def scal_num(A: dace.int64):
    return A + 5


def test_scal_num():
    A = np.random.randint(10, size=(1, ), dtype=np.int64)[0]
    B = scal_num(A)
    assert(np.array_equal(A + 5, B[0]))


@dace.program
def num_scal(A: dace.int64):
    return 5 + A


def test_num_scal():
    A = np.random.randint(10, size=(1, ), dtype=np.int64)[0]
    B = num_scal(A)
    assert(np.array_equal(5 + A, B[0]))


@dace.program
def scal_bool(A: dace.bool):
    return A and True


def test_scal_bool():
    A = np.random.randint(0, high=2, size=(1, ), dtype=np.bool)[0]
    B = scal_bool(A)
    assert(np.array_equal(np.logical_and(A, True), B[0]))


@dace.program
def bool_scal(A: dace.bool):
    return True and A


def test_bool_scal():
    A = np.random.randint(0, high=2, size=(1, ), dtype=np.bool)[0]
    B = bool_scal(A)
    assert(np.array_equal(np.logical_and(True, A), B[0]))


@dace.program
def scal_sym(A: dace.int64, tmp: dace.int64[N]):
    return A + N


def test_scal_sym():
    A = np.random.randint(0, high=2, size=(1, ), dtype=np.int64)[0]
    tmp = np.zeros((N.get(), ), dtype=np.int64)
    B = scal_sym(A, tmp)
    assert(np.array_equal(A + N.get(), B[0]))


@dace.program
def sym_scal(A: dace.int64, tmp: dace.int64[N]):
    return A + N


def test_sym_scal():
    A = np.random.randint(0, high=2, size=(1, ), dtype=np.int64)[0]
    tmp = np.zeros((N.get(), ), dtype=np.int64)
    B = sym_scal(A, tmp)
    assert(np.array_equal(N.get() + A, B[0]))


@dace.program
def num_num():
    return 5 + 6


def test_num_num():
    A = num_num()
    assert(A[0] == 11)


@dace.program
def num_sym(tmp: dace.int64[N]):
    return 5 + N


def test_num_sym():
    tmp = np.zeros((N.get(), ), dtype=np.int64)
    A = num_sym(tmp)
    assert(A[0] == 5 + N.get())


@dace.program
def sym_num(tmp: dace.int64[N]):
    return N + 5


def test_sym_num():
    tmp = np.zeros((N.get(), ), dtype=np.int64)
    A = sym_num(tmp)
    assert(A[0] == N.get() + 5)


@dace.program
def sym_sym(tmp: dace.int64[M, N]):
    return M + N


def test_sym_sym():
    tmp = np.zeros((M.get(), N.get()), dtype=np.int64)
    A = sym_sym(tmp)
    assert(A[0] == M.get() + N.get())


if __name__ == "__main__":
    test_array_array()
    test_array_array1()
    test_array1_array()
    test_array1_array1()
    test_array_scalar()
    test_scalar_array()
    test_array_num()
    test_num_array()
    test_array_bool()
    test_bool_array()
    test_array_sym()
    test_sym_array()
    test_scal_scal()
    test_scal_num()
    test_num_scal()
    test_scal_bool()
    test_bool_scal()
    test_scal_sym()
    test_sym_scal()
    test_num_num()
    # test_num_sym()
    # test_sym_num()
    # test_sym_sym()
