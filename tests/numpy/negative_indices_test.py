# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace


@dace.program
def negative_index(A: dace.int64[10]):
    return A[-2]


def test_negative_index():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)
    out = negative_index(A)
    assert out[0] == A[-2]


@dace.program
def nested_negative_index(A: dace.int64[10]):
    out = np.ndarray([2], dtype=np.int64)
    for i in dace.map[0:2]:
        out[i] = A[-1]
    return out


def test_nested_negative_index():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)
    out = nested_negative_index(A)
    assert out[0] == A[-1]
    assert out[1] == A[-1]


@dace.program
def negative_range(A: dace.int64[10]):
    return A[-5:-1]


def test_negative_range():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)
    out = negative_range(A)
    assert np.array_equal(out, A[-5:-1])


@dace.program
def nested_negative_range(A: dace.int64[10]):
    out = np.ndarray([10], dtype=np.int64)
    for i in dace.map[0:2]:
        out[i * 5:i * 5 + 5] = A[-6:-1]
    return out


def test_nested_negative_range():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)
    out = nested_negative_range(A)
    assert np.array_equal(out[:5], A[-6:-1])
    assert np.array_equal(out[5:], A[-6:-1])


@dace.program
def jacobi_2d(A: dace.float64[10, 10], B: dace.float64[10, 10]):
    for t in range(1, 10):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


def test_jacobi_2d():
    A = np.ones([10, 10], dtype=np.float64)
    B = np.ones([10, 10], dtype=np.float64)
    Ar = np.ones([10, 10], dtype=np.float64)
    Br = np.ones([10, 10], dtype=np.float64)
    jacobi_2d(A, B)
    for t in range(1, 10):
        Br[1:-1, 1:-1] = 0.2 * (Ar[1:-1, 1:-1] + Ar[1:-1, :-2] + Ar[1:-1, 2:] + Ar[2:, 1:-1] + Ar[:-2, 1:-1])
        Ar[1:-1, 1:-1] = 0.2 * (Br[1:-1, 1:-1] + Br[1:-1, :-2] + Br[1:-1, 2:] + Br[2:, 1:-1] + Br[:-2, 1:-1])
    assert np.allclose(A, Ar)
    assert np.allclose(B, Br)


if __name__ == '__main__':
    test_negative_index()
    test_nested_negative_index()
    test_negative_range()
    test_nested_negative_range()
    test_jacobi_2d()
