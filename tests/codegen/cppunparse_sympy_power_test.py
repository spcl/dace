# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A SymPy expression whose C++ spelling is not Python (``dace::math::ipow``) still converts to C++."""

import ast

import dace
from dace import cpf_lowering, symbolic
from dace.codegen import cppunparse
from dace.symbolic import ipow


def test_an_ipow_of_a_symbol_converts_to_its_cpp_spelling():
    cells = symbolic.symbol("cells_per_dim", dace.int64)
    expr = 4 * ipow(cells, 3)
    text = cppunparse.py2cpp(expr, expr_semicolon=False)
    assert text == symbolic.symstr(expr, cpp_mode=True)
    assert "ipow" in text


def test_fractional_exponent_is_real_in_standalone_dialects():
    M = symbolic.symbol("M", dace.float32)
    N = symbolic.symbol("N", dace.int64)
    expected = {
        cpf_lowering.Dialect.STANDALONE: "std::pow(M, (float(N) / float(3)))",
        cpf_lowering.Dialect.STANDALONE_C: "powf(M, (((float)(N)) / ((float)(3))))",
    }
    for dialect, unparsed in expected.items():
        with cpf_lowering.dialect_scope(dialect):
            tree = ast.parse("M**(N/3)", mode="eval").body
            symbols = {"M": dace.float32, "N": dace.int64}
            assert cppunparse.cppunparse(tree, expr_semicolon=False, defined_symbols=symbols) == unparsed
            assert symbolic.symstr(M ** (N / 3), cpp_mode=True) == f"({unparsed})"


if __name__ == "__main__":
    test_an_ipow_of_a_symbol_converts_to_its_cpp_spelling()
    test_fractional_exponent_is_real_in_standalone_dialects()
