# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A data descriptor must classify a symbolic value by the SEAM head, not by a sympy class.

`data/core.py` used to decide "is this a symbolic value?" by naming sympy (`symbolic.sympy.Basic`,
`sp.Expr`). For a value another backend built, that answer is No, which is wrong in two different
ways: `validate()` rejects a legal shape (loud), and `used_symbols` silently returns an EMPTY set
(quiet, and worse -- a shape symbol that goes unreported becomes a missing SDFG argument).

Both halves are asserted here whichever backend is active, so the sympy default path pins the
neutral heads as equivalent to the sympy ones and the idxalg path pins the real cross-engine case.
"""
import dace
import sympy

from dace import data, symbolic, symbolic_engine


def test_neutral_heads_agree_with_sympy_on_sympy_values():
    """The seam heads must accept genuine sympy values under EITHER backend -- during the hybrid
    period a descriptor still receives them, so a head that only knew the new engine would swap one
    silent misclassification for another."""
    expr = sympy.Symbol("N") * 2 + sympy.Symbol("M")
    assert isinstance(expr, symbolic.SymbolicBasic)
    assert isinstance(expr, symbolic.SymbolicExpr)
    assert issubclass(sympy.Basic, symbolic.SymbolicBasic)


def test_descriptor_accepts_and_reports_native_symbols():
    """A shape/stride term the active engine built natively is accepted, and its symbols reported.

    Under the sympy backend no foreign value can exist -- the seam's own head IS the sympy class --
    so the assertion there is the complementary one: the capability reports itself absent rather
    than answering by accident, and the descriptor still classifies the values that DO exist.
    """
    if symbolic_engine.native_parse is None:
        assert symbolic_engine.NATIVE_EXPR is None, "no parser, so no native type either"
        assert not symbolic_engine.is_native_symbol(sympy.Symbol("N")), "a sympy value is not native here"
        term = symbolic.pystr_to_symbolic("N * 2 + M")
        assert isinstance(term, symbolic.SymbolicBasic), "the seam head must accept the backend's own value"
    else:
        term = symbolic_engine.native_parse("N * 2 + M")
        assert not isinstance(term, symbolic.sympy.Basic), "the cross-engine case needs a foreign value"

    arr = data.Array(dace.float64, (term, 8), transient=True)
    arr.validate()
    assert {str(s) for s in arr.used_symbols(all_symbols=True)} == {"M", "N"}

    strided = data.Array(dace.float64, (16, 16), strides=(term, 1), transient=True)
    assert {str(s) for s in strided.used_symbols(all_symbols=True)} == {"M", "N"}


def test_descriptor_reports_symbols_of_parsed_shape():
    """The same guarantee through the public parse, which is what every frontend actually calls."""
    n, m = symbolic.pystr_to_symbolic("N"), symbolic.pystr_to_symbolic("M")
    arr = data.Array(dace.float64, (n, m * 2), transient=True)
    arr.validate()
    assert {str(s) for s in arr.free_symbols} == {"M", "N"}

    stream = data.Stream(dace.float64, buffer_size=n, transient=True)
    assert {str(s) for s in stream.used_symbols(all_symbols=True)} == {"N"}


if __name__ == "__main__":
    test_neutral_heads_agree_with_sympy_on_sympy_values()
    test_descriptor_accepts_and_reports_native_symbols()
    test_descriptor_reports_symbols_of_parsed_shape()
