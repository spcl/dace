import dace
import numpy as np
from copy import deepcopy as dc
from common import compare_numpy_output
import pytest


@compare_numpy_output
def test_add(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A + B


@compare_numpy_output
def test_sub(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A - B


@compare_numpy_output
def test_mult(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A * B


@compare_numpy_output
def test_div(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    return A / B


@compare_numpy_output
def test_floordiv(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A // B


@compare_numpy_output
def test_mod(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A % B


@compare_numpy_output
def test_pow(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A**B


@compare_numpy_output
def test_matmult(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A @ B


@compare_numpy_output
def test_lshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A << B


@compare_numpy_output
def test_rshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A >> B


@compare_numpy_output
def test_bitor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A | B


@compare_numpy_output
def test_bitxor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A ^ B


@compare_numpy_output
def test_bit(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A & B


@compare_numpy_output
def test_eq(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A == B


@compare_numpy_output
def test_noteq(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A != B


@compare_numpy_output
def test_lt(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A < B


@compare_numpy_output
def test_lte(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A <= B


@compare_numpy_output
def test_gt(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A > B


@compare_numpy_output
def test_gte(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A >= B


if __name__ == '__main__':
    import __main__ as main
    exit(pytest.main([main.__file__]))
