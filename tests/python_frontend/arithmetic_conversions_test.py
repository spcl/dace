# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def add(A: dace.complex64[5, 5], B: dace.float64[5, 5]):
    return A + B


def test_add():
    A = np.random.randint(0, high=10, size=(5, 5), dtype=np.uint64).astype(np.complex64)
    B = np.random.randint(-10, high=0, size=(5, 5), dtype=np.int32).astype(np.float64)
    C = add(A, B)
    assert (np.linalg.norm(C - A - B) / np.linalg.norm(A + B) < 1e-12)


@dace.program
def complex_conversion(a: dace.complex128[1], b: dace.int32):
    return a[0] + b


def test_complex_conversion():
    a = np.zeros((1, ), dtype=np.complex128)
    a[0] = 5 + 6j
    b = 7
    c = complex_conversion(a=a, b=b)
    assert (c[0] == 12 + 6j)


@dace.program
def float_conversion(a: dace.float32, b: dace.int64):
    return a + b


def test_float_conversion():
    a = np.float32(5.2)
    b = np.int64(7)
    c = float_conversion(a=a, b=b)
    assert (c[0] == a + b)


if __name__ == "__main__":
    test_add()
    test_complex_conversion()
    test_float_conversion()
