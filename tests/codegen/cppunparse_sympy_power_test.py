# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A SymPy expression whose C++ spelling is not Python (``dace::math::ipow``) still converts to C++."""

import dace
from dace import symbolic
from dace.codegen import cppunparse
from dace.symbolic import ipow


def test_an_ipow_of_a_symbol_converts_to_its_cpp_spelling():
    cells = symbolic.symbol("cells_per_dim", dace.int64)
    expr = 4 * ipow(cells, 3)
    text = cppunparse.py2cpp(expr, expr_semicolon=False)
    assert text == symbolic.symstr(expr, cpp_mode=True)
    assert "ipow" in text


if __name__ == "__main__":
    test_an_ipow_of_a_symbol_converts_to_its_cpp_spelling()
