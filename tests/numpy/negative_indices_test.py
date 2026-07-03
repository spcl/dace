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
def runtime_negative_index(A: dace.int64[10], i: dace.int64):
    return A[i]


def test_runtime_negative_index():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)

    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=True):
        sdfg = runtime_negative_index.to_sdfg(A, np.int64(-2), simplify=False)
        code = sdfg.generate_code()[0].clean_code
        out = sdfg(A=A, i=np.int64(-2))

    assert 'py_mod(__sym_i, 10)' in code
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
def runtime_nested_negative_range(A: dace.int64[10], offset: dace.int64, offset2: dace.int64):
    out = np.ndarray([10], dtype=np.int64)
    for i in dace.map[0:2]:
        out[i * 5:i * 5 + 5] = np.sum(A[offset:offset2])
    return out


def test_runtime_nested_negative_range():
    A = np.random.randint(0, 100, size=10, dtype=np.int64)

    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=True):
        sdfg = runtime_nested_negative_range.to_sdfg(A, np.int64(-6), np.int64(-1), simplify=False)
        runtime_code = sdfg.generate_code()[0].clean_code
        out = sdfg(A=A, offset=np.int64(-6), offset2=np.int64(-1))

    expected = np.full(5, np.sum(A[-6:-1]), dtype=np.int64)
    assert 'py_mod(' not in runtime_code.split('reduce_1_1_6', 1)[-1]
    assert np.array_equal(out[:5], expected)
    assert np.array_equal(out[5:], expected)


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
