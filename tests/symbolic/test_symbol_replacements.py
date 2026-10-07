# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for simultaneous symbol replacement: ``symbolic.symbol_replacements``, ``symbolic.replace_symbols`` and
``symbolic.safe_replace``."""

from typing import Dict, List

import pytest
import sympy

import dace
from dace import symbolic

A, B, C = (symbolic.symbol(name) for name in "ABC")
f = sympy.Function("f")

# (mapping, input expression, expected expression after a simultaneous replacement)
CASES = [
    (dict(A="B", B="A"), A + 2 * B, B + 2 * A),
    (dict(A="B", B="C"), A + 2 * B, B + 2 * C),
    (dict(A="A + 1"), 2 * A, 2 * (A + 1)),
    (dict(A="f(A)"), A + B, f(A) + B),
    (dict(A="f(B)", B="f(A)"), A - B, f(B) - f(A)),
    (dict(A=5, B="A"), A + 2 * B, 5 + 2 * A),
    (dict(A="B", B=5), A + 2 * B, B + 10),
]


@pytest.mark.parametrize("mapping, expr, expected", CASES)
def test_symbol_replacements(mapping: Dict, expr: sympy.Basic, expected: sympy.Basic):
    replacements = symbolic.symbol_replacements(mapping)
    assert sympy.simplify(symbolic.replace_symbols(expr, replacements) - expected) == 0


def test_symbol_replacements_identity():
    assert symbolic.symbol_replacements(None) is None
    assert symbolic.symbol_replacements({}) is None
    assert symbolic.symbol_replacements({"A": "A", B: B}) is None
    assert symbolic.symbol_replacements({"A": "A", "B": "C"}) == {"B": C}


def test_replace_symbols_of_any_type():
    """A mapping names its symbols, which are replaced whatever type they have."""
    N = symbolic.symbol("N", dtype=dace.int64)
    M = symbolic.symbol("M", dtype=dace.int32)
    replacements = symbolic.symbol_replacements({"N": "M", "M": 8})
    assert symbolic.replace_symbols(N * M + 1, replacements) == 8 * symbolic.pystr_to_symbolic("M") + 1
    assert symbolic.replace_symbols(5, replacements) == 5
    assert symbolic.replace_symbols(N, None) is N


@pytest.mark.parametrize("value_as_string", [False, True])
@pytest.mark.parametrize("mapping, expr, expected", CASES)
def test_safe_replace(mapping: Dict, expr: sympy.Basic, expected: sympy.Basic, value_as_string: bool):
    exprs: List[sympy.Basic] = [expr]

    def replace_callback(repl: Dict[str, str]):
        # Sequential substitution, as performed when replacing in SDFG properties and memlets
        symrepl = {symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v) for k, v in repl.items()}
        exprs[0] = exprs[0].subs(symrepl)

    symbolic.safe_replace(mapping, replace_callback, value_as_string=value_as_string)
    assert sympy.simplify(exprs[0] - expected) == 0


if __name__ == "__main__":
    for case in CASES:
        test_symbol_replacements(*case)
    test_symbol_replacements_identity()
    test_replace_symbols_of_any_type()
    for case in CASES:
        test_safe_replace(*case, False)
        test_safe_replace(*case, True)
