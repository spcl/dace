# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``int_floor`` must floor in C++, where ``/`` truncates: ``-7 // 2`` is -4, C's ``-7 / 2`` is -3."""

import numpy as np

import dace
from dace.symbolic import symstr

N = dace.symbol("N")


def test_int_floor_of_unknown_sign_prints_py_floor():
    assert symstr(N // 2, cpp_mode=True) == "(py_floor(N, 2))"


def test_int_floor_of_nonnegative_operands_prints_division():
    nonnegative = dace.symbol("K", nonnegative=True)
    assert symstr(nonnegative // 2, cpp_mode=True) == "(((K) / (2)))"


@dace.program
def floor_indexed_copy(A: dace.int64[4], B: dace.int64[1]):
    B[0] = A[N // 2 + 4]


def test_int_floor_negative_operand_in_memlet_index():
    A = np.arange(4, dtype=np.int64)
    B = np.full(1, -1, dtype=np.int64)
    floor_indexed_copy(A, B, N=-7)
    assert B[0] == A[(-7) // 2 + 4]


if __name__ == "__main__":
    test_int_floor_of_unknown_sign_prints_py_floor()
    test_int_floor_of_nonnegative_operands_prints_division()
    test_int_floor_negative_operand_in_memlet_index()
