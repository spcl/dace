# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from copy import deepcopy as dc
from common import compare_numpy_output
import pytest


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
@compare_numpy_output(positive=True)
def test_pow(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A**B


@compare_numpy_output()
def test_matmult(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A @ B


# dace has weird behavior here too
@compare_numpy_output(positive=True)
def test_lshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A << B


@compare_numpy_output(positive=True)
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
