# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace

_ = np.s_  # Ranges of decorated maps outside of programs: _[0:N] is slice(0, N)


def test_simple():
    A = np.random.rand(4)
    B = np.random.rand(4)
    with dace.tasklet():
        a << A[0]
        b >> B[1]
        b = a

    assert np.allclose(B[1], A[0])


def test_locals():
    a = 1
    b = "aa"
    c = 3
    A = np.random.rand(4)
    B = np.random.rand(4)
    with dace.tasklet():
        a << A[0]
        b >> B[1]
        b = a + c

    assert np.allclose(B[1], A[0] + c)


def test_wcr():
    A = np.random.rand(4)
    B = np.random.rand(4)
    C = np.copy(B)

    with dace.tasklet:
        a << A[0]
        b >> B(1, lambda a, b: a * b)[1]
        b = a

    assert np.allclose(B[1], A[0] * C[1])


def test_dynamic_input():
    A = np.random.rand(4, 5)
    B = np.random.rand(4)

    with dace.tasklet:
        a << A(-1)
        b >> B[1]
        b = a[0, 3]

    assert np.allclose(A[0, 3], B[1])


def test_nested_range():
    A = np.random.rand(4, 5)
    B = np.random.rand(4)

    with dace.tasklet:
        a << A(-1)[:, 1]
        b >> B[1]
        b = a[2]

    assert np.allclose(A[2, 1], B[1])


def test_dynamic_output():
    A = np.random.rand(4)
    B = np.random.rand(4)
    C = np.copy(B)

    def doit(A, B, i):
        with dace.tasklet:
            a << A(-1)
            b >> B(-1)[i]
            if a[i] > 0.5:
                b = 0

    A[2] = 0.7
    doit(A, B, 2)
    assert np.allclose(B[2], 0)

    A[3] = 0.2
    doit(A, B, 3)
    assert np.allclose(B[3], C[3])


def test_dynamic_output_wcr():
    A = np.random.rand(20)
    B = np.zeros([1], dtype=np.int32)

    # Count number of elements >= 0.5
    for i in dace.map[0 : A.shape[0]]:
        with dace.tasklet:
            a << A[i]
            b >> B(-1, lambda a, b: a + b)
            if a >= 0.5:
                b = 1

    expected = np.where(A >= 0.5)[0].shape[0]
    assert B[0] == expected


def test_push_to_stream():
    """A write of a whole stream pushes, with or without a dynamic memlet."""
    S = dace.define_stream(dace.int32)
    with dace.tasklet:
        s >> S
        s = 5
    with dace.tasklet:
        d >> S(-1)
        d = 6
        d = 7
    assert list(S) == [5, 6, 7]


def test_whole_array_write():
    """A write of a whole array, which is not a stream, sets all of its elements."""
    B = np.zeros(3)
    with dace.tasklet:
        b >> B
        b = 2
    assert np.array_equal(B, [2, 2, 2])


def test_decorated_map():
    A = np.arange(12.0).reshape(3, 4)
    B = np.zeros((3, 4))

    @dace.map(_[0:3, 0:4])
    def doubled(i, j):
        a << A[i, j]
        b >> B[i, j]
        b = 2 * a

    assert np.array_equal(B, 2 * A)


def test_decorated_mapscope():
    B = np.zeros(5)

    @dace.mapscope(_[0:5])
    def squares(i):
        B[i] = i * i

    assert np.array_equal(B, np.arange(5) ** 2)


def test_decorated_consume():
    """The body of a consume scope is a tasklet that pushes to the stream it consumes."""
    S = dace.define_stream(dace.int32)
    S.append(10)
    res = np.zeros(1)

    @dace.consume(S, 4)
    def fibonacci(elem, p):
        sout >> S(-1)
        val >> res(-1, lambda a, b: a + b)
        if elem == 1:
            val = 1
        elif elem > 1:
            sout = elem - 1
            sout = elem - 2

    assert res[0] == 55


def test_decorated_consumescope():
    S = dace.stream([3, 1, 2])
    seen = np.zeros(4)

    @dace.consumescope(S, 2)
    def visit(elem, p):
        seen[elem] = p + 1

    assert np.array_equal(seen, [0, 2, 1, 1])  # Popped as 2, 1, 3 by processing elements 0, 1, 0


if __name__ == "__main__":
    test_simple()
    test_locals()
    test_wcr()
    test_dynamic_input()
    test_nested_range()
    test_dynamic_output()
    test_dynamic_output_wcr()
    test_push_to_stream()
    test_whole_array_write()
    test_decorated_map()
    test_decorated_mapscope()
    test_decorated_consume()
    test_decorated_consumescope()
