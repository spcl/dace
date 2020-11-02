import dace
import numpy as np

import dace.frontend.python.replacements as repl


N = dace.symbol('N', dtype=dace.int32)


def test_broadcast_success():
    A = np.empty((5, 1))
    B = np.empty((1, 6))
    C = np.empty((6,))
    D = np.empty((1,))
    array_shapes = [arr.shape for arr in [A, B, C, D]]
    repl._broadcast(array_shapes)


def test_broadcast_fail():
    A = np.empty((5, 1))
    B = np.empty((1, 6))
    C = np.empty((5,))
    D = np.empty((1,))
    array_shapes = [arr.shape for arr in [A, B, C, D]]
    try:
        repl._broadcast(array_shapes)
    except SyntaxError:
        return
    assert(False)


@dace.program
def ufunc_add_simple(A: dace.int32[10], B: dace.int32[10]):
    return np.add(A, B)


def test_ufunc_add_simple():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    B = np.random.randint(10, size=(10,), dtype=np.int32)
    C = ufunc_add_simple(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_simple2(A: dace.int32[10], B: dace.int32):
    return np.add(A, B)


def test_ufunc_add_simple2():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    B = np.random.randint(10, dtype=np.int32)
    C = ufunc_add_simple2(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_simple3(A: dace.int32[10]):
    return np.add(A, 10)


def test_ufunc_add_simple3():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    C = ufunc_add_simple3(A)
    assert(np.array_equal(A + 10, C))


@dace.program
def ufunc_add_simple4(A: dace.int32[N]):
    return np.add(A, N)


def test_ufunc_add_simple4():
    N.set(10)
    A = np.random.randint(10, size=(N.get(),), dtype=np.int32)
    C = ufunc_add_simple4(A)
    assert(np.array_equal(A + N.get(), C))


@dace.program
def ufunc_add_out(A: dace.int32[10], B: dace.int32[10], C: dace.int32[10]):
    np.add(A, B, out=C)


def test_ufunc_add_out():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    B = np.random.randint(10, size=(10,), dtype=np.int32)
    C = np.empty((10,), dtype=np.int32)
    ufunc_add_out(A, B, C)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_out2(A: dace.int32[10], B: dace.int32[10], C: dace.int32[10]):
    np.add(A, B, out=(C))


def test_ufunc_add_out2():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    B = np.random.randint(10, size=(10,), dtype=np.int32)
    C = np.empty((10,), dtype=np.int32)
    ufunc_add_out2(A, B, C)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_out3(A: dace.int32[10], B: dace.int32[10], C: dace.int32[10]):
    np.add(A, B, C)


def test_ufunc_add_out3():
    A = np.random.randint(10, size=(10,), dtype=np.int32)
    B = np.random.randint(10, size=(10,), dtype=np.int32)
    C = np.empty((10,), dtype=np.int32)
    ufunc_add_out3(A, B, C)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_where(A: dace.int32[10], B: dace.int32[10], W: dace.bool[10]):
    return np.add(A, B, where=W)


def test_ufunc_add_where():
    A = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    W = np.random.randint(2, size=(10,), dtype=np.bool_)
    C = ufunc_add_where(A, B, W)
    assert(np.array_equal(np.add(A, B, where=W)[W], C[W]))
    assert(not np.array_equal((A + B)[np.logical_not(W)], C[np.logical_not(W)]))


@dace.program
def ufunc_add_where_true(A: dace.int32[10], B: dace.int32[10]):
    return np.add(A, B, where=True)


def test_ufunc_add_where_true():
    A = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    C = ufunc_add_where_true(A, B)
    assert(np.array_equal(np.add(A, B, where=True), C))


@dace.program
def ufunc_add_where_false(A: dace.int32[10], B: dace.int32[10]):
    return np.add(A, B, where=False)


def test_ufunc_add_where_false():
    A = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    C = ufunc_add_where_false(A, B)
    assert(not np.array_equal(A + B, C))


@dace.program
def ufunc_add_where_false(A: dace.int32[10], B: dace.int32[10]):
    return np.add(A, B, where=False)


def test_ufunc_add_where_false():
    A = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    C = ufunc_add_where_false(A, B)
    assert(not np.array_equal(A + B, C))


@dace.program
def ufunc_add_where_list(A: dace.int32[2], B: dace.int32[2]):
    return np.add(A, B, where=[True, False])


def test_ufunc_add_where_list():
    A = np.random.randint(1, 10, size=(2,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(2,), dtype=np.int32)
    try:
        C = ufunc_add_where_list(A, B)
    except:
        assert(True)
        return
    assert(False)


@dace.program
def ufunc_add_where1(A: dace.int32[1], B: dace.int32[1], W: dace.bool[1]):
    return np.add(A, B, where=W)


def test_ufunc_add_where1():
    A = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    W = np.random.randint(2, size=(1,), dtype=np.bool_)
    C = ufunc_add_where1(A, B, W)
    if W[0]:
        assert(np.array_equal(A + B, C))
    else:
        assert(not np.array_equal(A + B, C))


@dace.program
def ufunc_add_where1_true(A: dace.int32[1], B: dace.int32[1]):
    return np.add(A, B, where=True)


def test_ufunc_add_where1_true():
    A = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    C = ufunc_add_where1_true(A, B)
    assert(np.array_equal(A + B, C))


@dace.program
def ufunc_add_where1_false(A: dace.int32[1], B: dace.int32[1]):
    return np.add(A, B, where=False)


def test_ufunc_add_where1_false():
    A = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    B = np.random.randint(1, 10, size=(1,), dtype=np.int32)
    C = ufunc_add_where_false(A, B)
    assert(not np.array_equal(A + B, C))


@dace.program
def ufunc_add_reduce_simple(A: dace.int32[10]):
    return np.add.reduce(A)


def test_ufunc_add_reduce_simple():
    A = np.random.randint(1, 10, size=(10,), dtype=np.int32)
    s = ufunc_add_reduce_simple(A)[0]
    assert(np.array_equal(np.add.reduce(A), s))


@dace.program
def ufunc_add_reduce_axis(A: dace.int32[2, 2, 2, 2, 2]):
    return np.add.reduce(A, axis=(0, 2, 4))


def test_ufunc_add_reduce_axis():
    A = np.random.randint(1, 10, size=(2, 2, 2, 2, 2), dtype=np.int32)
    s = ufunc_add_reduce_axis(A)
    assert(np.array_equal(np.add.reduce(A, axis=(0, 2, 4)), s))


if __name__ == "__main__":
    test_broadcast_success()
    test_broadcast_fail()
    test_ufunc_add_simple()
    test_ufunc_add_simple2()
    test_ufunc_add_simple3()
    test_ufunc_add_simple4()
    test_ufunc_add_out()
    test_ufunc_add_out2()
    test_ufunc_add_out3()
    test_ufunc_add_where()
    test_ufunc_add_where_true()
    test_ufunc_add_where_false()
    test_ufunc_add_where_list()
    test_ufunc_add_where1()
    test_ufunc_add_where1_true()
    test_ufunc_add_where1_false()
    test_ufunc_add_reduce_simple()
    test_ufunc_add_reduce_axis()
