# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that an elementwise operation is typed -- and computed -- the way NumPy
types it, rather than the way C would.

The tasklet an operator lowers to computes in the types of its CONNECTORS, so
the result descriptor alone does not settle what the generated code does: an
integer ``/`` whose result container is ``float64`` still truncates unless the
operands are cast on the way in.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python.common import InvalidOperandTypes


def test_integer_true_division_is_float():

    @dace.program
    def divide(A: dace.int32[8], B: dace.int32[8]):
        return A / B

    A = np.arange(1, 9, dtype=np.int32)
    B = np.full(8, 3, dtype=np.int32)
    result = divide(A.copy(), B.copy())

    assert result.dtype == np.float64
    assert np.allclose(result, A / B)


def test_a_ufunc_divides_the_same_way_as_the_operator():

    @dace.program
    def divide(A: dace.uint32[8], B: dace.uint32[8]):
        return np.divide(A, B)

    A = np.arange(1, 9, dtype=np.uint32)
    B = np.full(8, 3, dtype=np.uint32)
    result = divide(A.copy(), B.copy())

    assert result.dtype == np.float64
    assert np.allclose(result, np.divide(A, B))


def test_a_complex_operand_keeps_its_imaginary_part():
    # C promotion picks the wider of two 8-byte types and would answer float64
    # here, silently dropping the imaginary part.

    @dace.program
    def scale(A: dace.complex64[4], B: dace.float64[4]):
        return A * B

    A = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    B = np.full(4, 2.0, dtype=np.float64)
    result = scale(A.copy(), B.copy())

    assert np.iscomplexobj(result)
    assert np.allclose(result, A * B)


def test_a_comparison_compares_in_the_promoted_type():
    # The result is bool, but the operands must still be compared as float64 --
    # casting them to the RESULT type would compare truth values instead.

    @dace.program
    def below(A: dace.float64[4], B: dace.int32[4]):
        return A < B

    A = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    B = np.array([1, 1, 3, 3], dtype=np.int32)
    result = below(A.copy(), B.copy())

    assert np.array_equal(result, A < B)


def test_an_augmented_assignment_keeps_its_target():
    # ``B **= A`` promotes to float64 by DaCe's power rule, but the augmented
    # form reads B in the same statement: rebinding the name would leave the
    # read resolving to a fresh, never-written container.

    @dace.program
    def power(A: dace.int64[4], B: dace.int64[4]):
        B **= A
        return B

    A = np.array([1, 2, 3, 2], dtype=np.int64)
    B = np.array([2, 3, 2, 5], dtype=np.int64)
    expected = B.copy()
    expected **= A
    result = power(A.copy(), B.copy())

    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_an_invalid_operand_type_is_reported_not_deferred():
    # NumPy rejects this outright. Lowering it to a Python callback would defer
    # the same error to run time, across the C callback boundary where it
    # cannot propagate.

    @dace.program
    def absolute(A: dace.complex64[4]):
        return np.fabs(A)

    with pytest.raises(InvalidOperandTypes):
        absolute.to_schedule_tree()


def test_a_bitwise_operator_rejects_floats():

    @dace.program
    def mask(A: dace.float64[4], B: dace.float64[4]):
        return A & B

    with pytest.raises(InvalidOperandTypes):
        mask.to_schedule_tree()


if __name__ == '__main__':
    test_integer_true_division_is_float()
    test_a_ufunc_divides_the_same_way_as_the_operator()
    test_a_complex_operand_keeps_its_imaginary_part()
    test_a_comparison_compares_in_the_promoted_type()
    test_an_augmented_assignment_keeps_its_target()
    test_an_invalid_operand_type_is_reported_not_deferred()
    test_a_bitwise_operator_rejects_floats()
