# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.frontend.python.common import DaceSyntaxError
import numpy as np
from common import compare_numpy_output
import pytest

# M = dace.symbol('M')
# N = dace.symbol('N')

M = 10
N = 20


@dace.program
def empty():
    return np.empty([M, N], dtype=np.uint32)


def test_empty():
    out = empty()
    assert (list(out.shape) == [M, N])
    assert (out.dtype == np.uint32)


@dace.program
def empty_like1(A: dace.complex64[N, M, 2]):
    return np.empty_like(A)


def test_empty_like1():
    A = np.ndarray([N, M, 2], dtype=np.complex64)
    out = empty_like1(A)
    assert (list(out.shape) == [N, M, 2])
    assert (out.dtype == np.complex64)


@dace.program
def empty_like2(A: dace.complex64[N, M, 2]):
    return np.empty_like(A, shape=[2, N, N])


def test_empty_like2():
    A = np.ndarray([N, M, 2], dtype=np.complex64)
    out = empty_like2(A)
    assert (list(out.shape) == [2, N, N])
    assert (out.dtype == np.complex64)


@dace.program
def empty_like3(A: dace.complex64[N, M, 2]):
    return np.empty_like(A, dtype=np.uint8)


def test_empty_like3():
    A = np.ndarray([N, M, 2], dtype=np.complex64)
    out = empty_like3(A)
    assert (list(out.shape) == [N, M, 2])
    assert (out.dtype == np.uint8)


@compare_numpy_output()
def test_ones():
    return np.ones([N, N], dtype=np.float32)


@compare_numpy_output()
def test_ones_like(A: dace.complex64[N, M, 2]):
    return np.ones_like(A)


@compare_numpy_output()
def test_zeros():
    return np.zeros([N, N], dtype=np.float32)


@compare_numpy_output()
def test_zeros_like(A: dace.complex64[N, M, 2]):
    return np.zeros_like(A)


@compare_numpy_output()
def test_full():
    return np.full([N, N], fill_value=np.complex32(5 + 6j))


@compare_numpy_output()
def test_full_like(A: dace.complex64[N, M, 2]):
    return np.full_like(A, fill_value=5)


@compare_numpy_output()
def test_copy(A: dace.complex64[N, M, 2]):
    return np.copy(A)


@compare_numpy_output()
def test_identity():
    return np.identity(M)


@compare_numpy_output()
def test_array(A: dace.float64[N, M]):
    return np.array(A)


cst = np.random.rand(10, 10).astype(np.float32)


@compare_numpy_output()
def test_array_constant():
    return np.array(cst, dtype=np.float32)


@compare_numpy_output()
def test_array_literal():
    return np.array([[1, 2], [3, 4]], dtype=np.float32)


@compare_numpy_output()
def test_arange_0():
    return np.arange(10, dtype=np.int32)


@compare_numpy_output()
def test_arange_1():
    return np.arange(2, 10, dtype=np.int32)


@compare_numpy_output()
def test_arange_2():
    return np.arange(2, 10, 3, dtype=np.int32)


@compare_numpy_output()
def test_arange_3():
    return np.arange(2.5, 10, 3, dtype=np.float32)


@compare_numpy_output()
def test_arange_4():
    return np.arange(2.5, 10, 3, dtype=np.int32)


@compare_numpy_output()
def test_arange_5():
    return np.arange(2, 10, 3)


@compare_numpy_output()
def test_arange_6():
    return np.arange(2.5, 10, 3)


@compare_numpy_output()
def test_linspace_1():
    return np.linspace(2.5, 10, num=3)


@compare_numpy_output()
def test_linspace_2():
    space, step = np.linspace(2.5, 10, num=3, retstep=True)
    return space, step


@compare_numpy_output()
def test_linspace_3():
    a = np.array([1, 2, 3])
    return np.linspace(a, 5, num=10)


@compare_numpy_output()
def test_linspace_4():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    space, step = np.linspace(a, 10, endpoint=False, retstep=True)
    return space, step


@compare_numpy_output()
def test_linspace_5():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[5], [10]])
    return np.linspace(a, b, endpoint=False, axis=1)


@compare_numpy_output()
def test_linspace_6():
    return np.linspace(-5, 5.5, dtype=np.float32)


@dace.program
def program_strides_0():
    A = dace.ndarray((2, 2), dtype=dace.int32, strides=(2, 1))
    for i, j in dace.map[0:2, 0:2]:
        A[i, j] = i * 2 + j
    return A


def test_strides_0():
    A = program_strides_0()
    assert A.strides == (8, 4)
    assert np.allclose(A, [[0, 1], [2, 3]])


@dace.program
def program_strides_1():
    A = dace.ndarray((2, 2), dtype=dace.int32, strides=(4, 2))
    for i, j in dace.map[0:2, 0:2]:
        A[i, j] = i * 2 + j
    return A


def test_strides_1():
    A = program_strides_1()
    assert A.strides == (16, 8)
    assert np.allclose(A, [[0, 1], [2, 3]])


@dace.program
def program_strides_2():
    A = dace.ndarray((2, 2), dtype=dace.int32, strides=(1, 2))
    for i, j in dace.map[0:2, 0:2]:
        A[i, j] = i * 2 + j
    return A


def test_strides_2():
    A = program_strides_2()
    assert A.strides == (4, 8)
    assert np.allclose(A, [[0, 1], [2, 3]])


@dace.program
def program_strides_3():
    A = dace.ndarray((2, 2), dtype=dace.int32, strides=(2, 4))
    for i, j in dace.map[0:2, 0:2]:
        A[i, j] = i * 2 + j
    return A


@pytest.mark.skip(reason='Temporarily skipping due to a sporadic issue on CI')
def test_strides_3():
    A = program_strides_3()
    assert A.strides == (8, 16)
    assert np.allclose(A, [[0, 1], [2, 3]])


def test_zeros_symbolic_size_scalar():
    K = dace.symbol('K')

    @dace.program
    def zeros_symbolic_size():
        return np.zeros((K), dtype=np.uint32)

    out = zeros_symbolic_size(K=10)
    assert (list(out.shape) == [10])
    assert (out.dtype == np.uint32)


def test_ones_scalar_size_scalar():

    @dace.program
    def ones_scalar_size(k: dace.int32):
        a = np.ones(k, dtype=np.uint32)
        return np.sum(a)

    with pytest.raises(DaceSyntaxError):
        out = ones_scalar_size(20)
        assert out == 20


def test_ones_scalar_size():

    @dace.program
    def ones_scalar_size(k: dace.int32):
        a = np.ones((k, k), dtype=np.uint32)
        return np.sum(a)

    with pytest.raises(DaceSyntaxError):
        out = ones_scalar_size(20)
        assert out == 20 * 20


if __name__ == "__main__":
    test_empty()
    test_empty_like1()
    test_empty_like2()
    test_empty_like3()
    test_ones()
    test_ones_like()
    test_zeros()
    test_zeros_like()
    test_full()
    test_full_like()
    test_copy()
    test_identity()
    test_array()
    test_array_constant()
    test_array_literal()
    test_arange_0()
    test_arange_1()
    test_arange_2()
    test_arange_3()
    test_arange_4()
    test_arange_5()
    test_arange_6()
    test_linspace_1()
    test_linspace_2()
    test_linspace_3()
    test_linspace_4()
    test_linspace_5()
    test_linspace_6()
    test_strides_0()
    test_strides_1()
    test_strides_2()
    test_strides_3()
    test_zeros_symbolic_size_scalar()
    test_ones_scalar_size_scalar()
    test_ones_scalar_size()
