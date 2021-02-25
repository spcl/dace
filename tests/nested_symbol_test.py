# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import warnings

N = dace.symbol('N')
N.set(12345)


@dace.program
def nested(A: dace.float64[N], B: dace.float64[N], factor: dace.float64):
    B[:] = A * factor


@dace.program
def nested_symbol(A: dace.float64[N], B: dace.float64[N]):
    nested(A[0:5], B[0:5], 0.5)
    nested(A=A[5:N], B=B[5:N], factor=2.0)


@dace.program
def nested_symbol_dynamic(A: dace.float64[N]):
    for i in range(5):
        nested(A[0:i], A[0:i], i)


def test_nested_symbol():
    A = np.random.rand(20)
    B = np.random.rand(20)
    nested_symbol(A, B)
    assert np.allclose(B[0:5], A[0:5] / 2) and np.allclose(
        B[5:20], A[5:20] * 2)


def test_nested_symbol_dynamic():
    if not dace.Config.get_bool('optimizer',
                                'automatic_strict_transformations'):
        warnings.warn("Test disabled (missing allocation lifetime support)")
        return

    A = np.random.rand(5)
    expected = A.copy()
    for i in range(5):
        expected[0:i] *= i
    nested_symbol_dynamic(A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_nested_symbol()
    test_nested_symbol_dynamic()
