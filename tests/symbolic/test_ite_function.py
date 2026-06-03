# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the ``ITE`` SymPy ``Function`` registered in
:mod:`dace.symbolic`.

``ITE(c, a, b)`` is the symbolic stand-in for ``c ? a : b`` used by the
branch-normalization passes (M3.1+). The vectorizer's emission utility lowers
calls to ``ITE`` into a SIMD blend (``vector_select`` / ``_mm512_mask_blend_pd``
/ ``svsel_*``); fallback codegen pulls the templated helper from
``dace/runtime/include/dace/ITE.h``.

These tests pin the contracts the passes will rely on:
- ``SymExpr("ITE(c, a, b)")`` parses and round-trips through
  ``DaceSympyPrinter``.
- Free-symbol collection returns the arg symbols, never ``ITE`` itself.
- ``ITE`` folds eagerly when the predicate is a SymPy boolean literal
  (matches :class:`IfExpr`'s behaviour).
- The header on disk is well-formed and has the expected template signature.
"""
import os
import re

import numpy as np
import sympy

import dace
from dace.symbolic import DaceSympyPrinter, SymExpr, ITE


def test_ite_parses_via_symexpr():
    expr = SymExpr("ITE(c, a, b)")
    assert isinstance(expr, ITE)
    args = list(expr.args)
    assert len(args) == 3
    assert {str(a) for a in args} == {"c", "a", "b"}


def test_ite_round_trips_through_dace_sympy_printer():
    expr = SymExpr("ITE(c, a, b)")
    printer = DaceSympyPrinter(arrays=None)
    rendered = printer.doprint(expr)
    # Default sympy function printing: "ITE(c, a, b)" (commutative-arg order
    # is preserved because nargs=3 disables sympy's reorder).
    assert rendered.startswith("ITE(")
    assert rendered.endswith(")")
    assert set(re.findall(r"\b[a-z]\b", rendered)) == {"c", "a", "b"}


def test_ite_free_symbols_are_args_only():
    expr = SymExpr("ITE(c, a, b)")
    free = {str(s) for s in expr.free_symbols}
    assert free == {"a", "b", "c"}
    assert "ITE" not in free


def test_ite_folds_when_predicate_is_true():
    a, b = sympy.symbols("a b")
    folded = ITE(sympy.true, a, b)
    assert folded == a


def test_ite_folds_when_predicate_is_false():
    a, b = sympy.symbols("a b")
    folded = ITE(sympy.false, a, b)
    assert folded == b


def test_ite_stays_symbolic_when_predicate_is_a_symbol():
    c, a, b = sympy.symbols("c a b")
    expr = ITE(c, a, b)
    assert isinstance(expr, ITE)
    assert tuple(expr.args) == (c, a, b)


def test_ite_subs_predicate_evaluates_lazily():
    """``ITE(c, a, b).subs(c, sympy.true)`` should still collapse to ``a``,
    consistent with eager-on-boolean-literal evaluation."""
    c, a, b = sympy.symbols("c a b")
    expr = ITE(c, a, b)
    assert expr.subs(c, sympy.true) == a
    assert expr.subs(c, sympy.false) == b


def test_ite_nested():
    """Nested ``ITE`` should not collapse when predicates remain symbolic."""
    c1, c2, a, b, d = sympy.symbols("c1 c2 a b d")
    expr = ITE(c1, ITE(c2, a, b), d)
    # Outer head is still ``ITE``.
    assert isinstance(expr, ITE)
    # The inner ``ITE`` is in the argument tree.
    found_inner = any(isinstance(arg, ITE) for arg in expr.args)
    assert found_inner


def test_ite_header_exists_and_declares_template():
    """The C++ runtime helper must exist and expose the expected signature."""
    header = os.path.join(os.path.dirname(dace.__file__), "runtime", "include", "dace", "ITE.h")
    assert os.path.isfile(header), header
    with open(header) as f:
        text = f.read()
    assert "template" in text
    # Loose signature check -- must accept (bool, T, T). The function lives at
    # top level (un-namespaced) to match the codegen convention used for
    # ``left_shift``/``ROUND`` in ``dace/math.h``.
    assert re.search(r"ITE\s*\(\s*bool\s+\w+\s*,\s*T\s+\w+\s*,\s*T\s+\w+\s*\)", text), text


def test_ite_compiled_in_sdfg_matches_python_reference():
    """End-to-end: build an SDFG whose tasklet body calls ``ITE(c, a, b)``,
    compile, run, and compare against the plain-Python ternary for every
    combination of ``c`` and a small set of operand values."""
    sdfg = dace.SDFG("ite_e2e")
    sdfg.add_array("a", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("b", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("out", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)

    state = sdfg.add_state("s", is_start_block=True)
    ra = state.add_access("a")
    rb = state.add_access("b")
    wo = state.add_access("out")
    t = state.add_tasklet("m", {"_a", "_b"}, {"_o"}, "_o = ITE(c, _a, _b)")
    state.add_edge(ra, None, t, "_a", dace.Memlet("a[0]"))
    state.add_edge(rb, None, t, "_b", dace.Memlet("b[0]"))
    state.add_edge(t, "_o", wo, None, dace.Memlet("out[0]"))

    csdfg = sdfg.compile()
    for cv in (True, False):
        for av, bv in ((3.0, -1.0), (0.0, 9.5), (-7.25, 4.5)):
            a = np.array([av], dtype=np.float64)
            b = np.array([bv], dtype=np.float64)
            out = np.zeros((1, ), dtype=np.float64)
            csdfg(a=a, b=b, out=out, c=cv)
            expected = av if cv else bv
            np.testing.assert_allclose(out, np.array([expected]), err_msg=f"c={cv} a={av} b={bv} got={out[0]}")
