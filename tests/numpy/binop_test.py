# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from common import compare_numpy_output


@compare_numpy_output()
def test_add(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A + B


@compare_numpy_output()
def test_sub(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A - B


@compare_numpy_output()
def test_mult(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A * B


@compare_numpy_output(non_zero=True)
def test_div(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    return A / B


# A // B is not implemented correctly in dace for negative numbers
@compare_numpy_output(non_zero=True, positive=True)
def test_floordiv(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A // B


# A % B is not implemented correctly in dace for negative numbers
@compare_numpy_output(non_zero=True, positive=True)
def test_mod(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A % B


# numpy throws an error for negative B, dace doesn't
@compare_numpy_output(positive=True, casting=np.float64)
def test_pow(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A**B


@compare_numpy_output()
def test_matmult(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    return A @ B


# Python/NumPy define left/right shift as multiplying/dividing a by 2**b.
# Therefore, they may return a different result than C languages.
# For example, something like 24 << 82 will result in Python in 24 * 2**82.
# This is well outside the range of numbers that can be represented by int64.
# NumPy will set such results to 0. C languages just wrap around. Their result
# is 6291456.
@compare_numpy_output(positive=True, max_value=10)
def test_lshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A << B


@compare_numpy_output(positive=True, max_value=10)
def test_rshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A >> B


@compare_numpy_output()
def test_bitor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A | B


@compare_numpy_output()
def test_bitxor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A ^ B


@compare_numpy_output()
def test_bit(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A & B


@compare_numpy_output()
def test_eq(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A == B


@compare_numpy_output()
def test_noteq(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A != B


@compare_numpy_output()
def test_lt(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A < B


@compare_numpy_output()
def test_lte(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A <= B


@compare_numpy_output()
def test_gt(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A > B


@compare_numpy_output()
def test_gte(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A >= B


@compare_numpy_output(check_dtype=True)
def test_bitand_bool(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    return A & B


@compare_numpy_output(check_dtype=True)
def test_bitor_bool(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    return A | B


@compare_numpy_output(check_dtype=True)
def test_bitxor_bool(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    return A ^ B


@compare_numpy_output(check_dtype=True)
def test_lshift_bool(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    return A << B


@compare_numpy_output(positive=True, max_value=6, check_dtype=True)
def test_lshift_keeps_left_dtype(A: dace.int32[5, 5]):
    return A << 2


@compare_numpy_output(check_dtype=True)
def test_bitand_widens_to_int64(A: dace.int32[5, 5], B: dace.int64[5, 5]):
    return A & B


def bitand(A, B):
    return A & B


def bitor(A, B):
    return A | B


def bitxor(A, B):
    return A ^ B


def lshift(A, B):
    return A << B


def rshift(A, B):
    return A >> B


BITWISE_BINOPS = (bitand, bitor, bitxor, lshift, rshift)


@pytest.mark.parametrize('kernel', BITWISE_BINOPS, ids=lambda k: k.__name__)
def test_bitwise_signed_uint64_rejected(kernel):
    """int32 op uint64 has no common integer type, so numpy's bitwise ufuncs reject it."""
    with pytest.raises(TypeError):
        dace.program(kernel).to_sdfg(simplify=False, A=np.ones((5, 5), np.int32), B=np.ones((5, 5), np.uint64))


@pytest.mark.parametrize('kernel', BITWISE_BINOPS, ids=lambda k: k.__name__)
def test_bitwise_float_rejected(kernel):
    with pytest.raises(TypeError):
        dace.program(kernel).to_sdfg(simplify=False, A=np.ones((5, 5), np.float64), B=np.ones((5, 5), np.float64))


if __name__ == '__main__':
    # generate this with
    # cat binop_test.py | grep -oP '(?<=f ).*(?=\()' | awk '{print $0 "()"}'
    test_add()
    test_sub()
    test_mult()
    test_div()
    test_floordiv()
    test_mod()
    test_pow()
    test_matmult()
    test_lshift()
    test_rshift()
    test_bitor()
    test_bitxor()
    test_bit()
    test_eq()
    test_noteq()
    test_lt()
    test_lte()
    test_gt()
    test_gte()
    test_bitand_bool()
    test_bitor_bool()
    test_bitxor_bool()
    test_lshift_bool()
    test_lshift_keeps_left_dtype()
    test_bitand_widens_to_int64()
    for k in BITWISE_BINOPS:
        test_bitwise_signed_uint64_rejected(k)
        test_bitwise_float_rejected(k)
