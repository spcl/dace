# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``//`` is Python's floor division (``py_floor``); ``int_floor`` is C's truncating ``/``.

``-7 // 2`` is -4 and C's ``-7 / 2`` is -3: the two agree only on a nonnegative dividend and a positive divisor.
"""

import numpy as np

import dace
from dace import symbolic
from dace.symbolic import symstr

N = dace.symbol("N")


def test_floor_division_parses_to_py_floor_and_prints_py_floor_in_cpp():
    expr = symbolic.pystr_to_symbolic("N // 2")
    assert expr.func is symbolic.py_floor
    assert symstr(expr, cpp_mode=True) == "(py_floor(N, 2))"
    assert symstr(expr) == "(((N) // (2)))"


def test_floor_division_of_nonnegative_operands_prints_division():
    nonnegative = dace.symbol("K", nonnegative=True)
    expr = symbolic.pystr_to_symbolic("K // 2", symbol_map={"K": nonnegative})
    assert expr.func is symbolic.int_floor
    assert symstr(expr, cpp_mode=True) == "(((K) / (2)))"


def test_int_floor_truncates_and_py_floor_floors():
    assert symbolic.int_floor(-7, 2) == -3
    assert symbolic.int_floor(7, -2) == -3
    assert symbolic.int_floor(7, 2) == 3
    assert symbolic.py_floor(-7, 2) == -4
    assert symbolic.py_floor(7, -2) == -4
    assert symbolic.py_floor(7, 2) == 3


def test_explicit_int_floor_keeps_its_name_in_python_and_divides_in_cpp():
    expr = symbolic.pystr_to_symbolic("int_floor(N, 2)")
    assert expr.func is symbolic.int_floor
    assert symstr(expr) == "(int_floor(N, 2))"
    assert symstr(expr, cpp_mode=True) == "(((N) / (2)))"


@dace.program
def floor_indexed_copy(A: dace.int64[4], B: dace.int64[1]):
    B[0] = A[N // 2 + 4]


def test_floor_division_negative_operand_in_memlet_index():
    A = np.arange(4, dtype=np.int64)
    B = np.full(1, -1, dtype=np.int64)
    floor_indexed_copy(A, B, N=-7)
    assert B[0] == A[(-7) // 2 + 4]


if __name__ == "__main__":
    test_floor_division_parses_to_py_floor_and_prints_py_floor_in_cpp()
    test_floor_division_of_nonnegative_operands_prints_division()
    test_int_floor_truncates_and_py_floor_floors()
    test_explicit_int_floor_keeps_its_name_in_python_and_divides_in_cpp()
    test_floor_division_negative_operand_in_memlet_index()
