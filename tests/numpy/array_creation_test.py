# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output

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

@compare_numpy_output(check_dtype=True)
def test_linspace_0():
    return np.linspace(0, 9, 10)

@compare_numpy_output(check_dtype=True)
def test_linspace_1():
    return np.linspace(0, 9, 10, dtype=np.int32)

@compare_numpy_output(check_dtype=True)
def test_linspace_2():
    return np.linspace(0, 10, 10)

@compare_numpy_output(check_dtype=True)
def test_linspace_3():
    return np.linspace(0, 10, 10, endpoint=False)

@compare_numpy_output(check_dtype=True)
def test_linspace_4():
    return np.linspace(0, 10, 10, dtype=np.int16)


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
    test_linspace_0()
    test_linspace_1()
    test_linspace_2()
    test_linspace_3()
    test_linspace_4()
