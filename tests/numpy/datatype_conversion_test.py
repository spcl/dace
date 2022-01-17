# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def simple_array_conversion(A: dace.int32[10]):
    return dace.int64(A)


def test_simple_array_conversion():
    A = np.random.randint(0, 1000, size=(10, ), dtype=np.int32)
    B = simple_array_conversion(A)

    assert (B.dtype == np.int64)
    assert (np.array_equal(B, np.int64(A)))


@dace.program
def simple_array_conversion2(A: dace.int32[10]):
    return dace.complex128(A)


def test_simple_array_conversion2():
    A = np.random.randint(0, 1000, size=(10, ), dtype=np.int32)
    B = simple_array_conversion2(A)

    assert (B.dtype == np.complex128)
    assert (np.array_equal(B, np.complex128(A)))


@dace.program
def simple_scalar_conversion(A: dace.int32):
    return dace.int64(A)


def test_simple_scalar_conversion():
    A = np.random.randint(0, 1000, size=(10, ), dtype=np.int32)[0]
    B = simple_scalar_conversion(A)

    assert (B.dtype == np.int64)
    assert (B[0] == A)


@dace.program
def simple_constant_conversion():
    return dace.float64(0)


def test_simple_constant_conversion():
    A = simple_constant_conversion()

    assert (A.dtype == np.float64)
    assert (A[0] == np.float64(0))


N = dace.symbol('N', dtype=dace.int32)


@dace.program
def simple_symbol_conversion(A: dace.int32[N]):
    return dace.int64(N)


def test_simple_symbol_conversion():
    N.set(10)
    A = np.random.randint(0, 1000, size=(N.get(), ), dtype=np.int32)
    B = simple_symbol_conversion(A)

    assert (B.dtype == np.int64)
    assert (B[0] == np.int64(N.get()))


if __name__ == "__main__":
    test_simple_array_conversion()
    test_simple_array_conversion2()
    test_simple_scalar_conversion()
    test_simple_constant_conversion()
    test_simple_symbol_conversion()
