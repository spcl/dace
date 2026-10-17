# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ipow`` of integer operands is integer arithmetic: an array extent, and a loop bound that needs no rounding.

Program symbols carry no SymPy assumptions, so ``ipow(N, 2).is_integer`` is unknown; integrality has to come from the
operations, as it does for ``N**2``. Missing that, ``RelaxIntegerPowers`` made a ``LEN_2D**2`` extent invalid (TSVC
offload) and a ``ceiling(R**K)`` map end printed as ``ceil(...)``, a double OpenMP rejects (npbench stockham_fft).
"""

import sympy

import dace
from dace import symbolic


def test_an_ipow_extent_is_integral():
    N = symbolic.symbol("N")
    assert symbolic.integral_index_expression(symbolic.ipow(N, 2))
    sdfg = dace.SDFG("ipow_extent")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", [symbolic.ipow(N, 2)], dace.float64)
    sdfg.arrays["A"].validate()


def test_the_ceiling_of_an_ipow_prints_the_integer_itself():
    R, K = symbolic.symbol("R"), symbolic.symbol("K")
    printed = symbolic.symstr(sympy.ceiling(symbolic.ipow(R, K)), cpp_mode=True)
    assert "ceil" not in printed, printed


def test_the_ceiling_of_a_symbolic_pow_still_rounds():
    # ``R**(K - 1)`` is a fraction for K = 0, so its ceiling is no identity
    R, K = symbolic.symbol("R"), symbolic.symbol("K")
    assert "ceil" in symbolic.symstr(sympy.ceiling(R ** (K - 1)), cpp_mode=True)


if __name__ == "__main__":
    test_an_ipow_extent_is_integral()
    test_the_ceiling_of_an_ipow_prints_the_integer_itself()
    test_the_ceiling_of_a_symbolic_pow_still_rounds()
