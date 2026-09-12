# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``NormalizeTernaryTasklets``: a tasklet-body Python ternary (``ast.IfExp``)
``_o = _t if _cond else _e`` is rewritten to the first-class ``_o = ITE(_cond, _t, _e)``
function form.

Closes a gap left by ``SplitTasklets``: its ``ASTSplitter`` already knows how to lower a
ternary to ``ITE(...)`` (``dace/transformation/passes/split_tasklets.py``), but only as a
by-product of splitting a multi-op body. A tasklet whose ENTIRE body is one flat ternary
collapses to a single SSA line, which ``SplitTasklets`` treats as "already split" and leaves
untouched -- so the original raw ternary survives to ``ConvertTaskletsToTileOps``, where
``_detect_ite``'s raw-ternary branch only matches an exact permutation of THREE in-connector
names. A literal arm, a free-symbol arm, or any parenthesisation/whitespace variant is missed
entirely. This test pins the rewrite for every arm shape ``_detect_ite``'s ``ITE(...)`` branch
already understands (connector / free symbol / literal), plus nesting -- and proves the pass is
what does it (non-vacuity: the pass is reverted and the same assertions are shown to fail).
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import ast

import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.normalize_masked_write_tasklets import NormalizeTernaryTasklets


def _tasklet_sdfg(body, arrays_by_conn, symbols=None):
    """Minimal single-state SDFG: one tasklet whose in-connectors are each fed by a
    length-1 float64 array (``arrays_by_conn``: ``{connector: array_name}``), with a single
    out-connector ``_o`` written to array ``O``. Symbols (``{name: dtype}``) are declared on
    the SDFG but wired to nothing -- a tasklet may reference an outer symbol by name directly.

    :returns: ``(sdfg, tasklet)``.
    """
    sdfg = dace.SDFG("ternary_fixture")
    for sym, dtype in (symbols or {}).items():
        sdfg.add_symbol(sym, dtype)
    for arr_name in arrays_by_conn.values():
        if arr_name not in sdfg.arrays:
            sdfg.add_array(arr_name, (1, ), dace.float64, transient=False)
    sdfg.add_array("O", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    tasklet = state.add_tasklet("t", set(arrays_by_conn.keys()), {"_o"}, body)
    for conn, arr_name in arrays_by_conn.items():
        an = state.add_access(arr_name)
        state.add_edge(an, None, tasklet, conn, Memlet(f"{arr_name}[0]"))
    o_an = state.add_access("O")
    state.add_edge(tasklet, "_o", o_an, None, Memlet("O[0]"))
    return sdfg, tasklet


def _has_ifexp(code_str):
    """True if ``code_str`` still contains a raw Python ternary (``ast.IfExp``) anywhere."""
    tree = ast.parse(code_str)
    return any(isinstance(n, ast.IfExp) for n in ast.walk(tree))


def test_connector_arms():
    """All three of cond / t / e are in-connectors: ``_o = _t if _cond else _e``."""
    sdfg, tasklet = _tasklet_sdfg("_o = _t if _cond else _e", {"_cond": "COND", "_t": "T", "_e": "E"})
    in_before, out_before = set(tasklet.in_connectors), set(tasklet.out_connectors)
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_cond, _t, _e)"
    assert not _has_ifexp(tasklet.code.as_string)
    # Body-text only: no connector invented, dropped, or renamed.
    assert set(tasklet.in_connectors) == in_before
    assert set(tasklet.out_connectors) == out_before
    sdfg.validate()


def test_symbol_arm():
    """The else-arm is a free symbol, not a connector: ``_o = _t if _cond else N``."""
    sdfg, tasklet = _tasklet_sdfg("_o = _t if _cond else N", {"_cond": "COND", "_t": "T"}, symbols={"N": dace.int64})
    in_before = set(tasklet.in_connectors)
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_cond, _t, N)"
    # N stays a free symbol -- never promoted to a connector or renamed. ``Tasklet.free_symbols``
    # resolves via ``dace.symbolic`` internally (properties.CodeBlock.get_free_symbols), never
    # calling sympy directly here.
    assert "N" in tasklet.free_symbols
    assert set(tasklet.in_connectors) == in_before
    assert "N" not in tasklet.in_connectors
    rhs = tasklet.code.as_string.split(" = ", 1)[1]
    assert dace.symbolic.pystr_to_symbolic("N") in dace.symbolic.pystr_to_symbolic(rhs).free_symbols
    sdfg.validate()


def test_literal_arm():
    """The else-arm is a float literal: ``_o = _t if _cond else 0.0``."""
    sdfg, tasklet = _tasklet_sdfg("_o = _t if _cond else 0.0", {"_cond": "COND", "_t": "T"})
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_cond, _t, 0.0)"
    sdfg.validate()


def test_both_arms_non_connector():
    """Neither arm is a connector -- one literal, one free symbol: only ``_cond`` is wired.
    ``_o = 1.0 if _cond else N``."""
    sdfg, tasklet = _tasklet_sdfg("_o = 1.0 if _cond else N", {"_cond": "COND"}, symbols={"N": dace.int64})
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_cond, 1.0, N)"
    assert set(tasklet.in_connectors) == {"_cond"}
    assert "N" in tasklet.free_symbols
    sdfg.validate()


def test_nested_ternary():
    """A ternary in the else-arm recurses to a nested ``ITE``:
    ``_o = _t if _c1 else (_e if _c2 else _f)``."""
    sdfg, tasklet = _tasklet_sdfg("_o = _t if _c1 else (_e if _c2 else _f)", {
        "_c1": "C1",
        "_t": "T",
        "_c2": "C2",
        "_e": "E",
        "_f": "F"
    })
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_c1, _t, ITE(_c2, _e, _f))"
    assert not _has_ifexp(tasklet.code.as_string)
    sdfg.validate()


@pytest.mark.parametrize("body", [
    "_o = _t if _cond else _e",
    "_o = (_t if _cond else _e)",
    "_o  =   _t   if _cond   else   _e",
    "_o=(_t if _cond else _e)",
])
def test_parenthesisation_and_whitespace_variants(body):
    """Parens / odd whitespace don't matter -- normalization parses the AST, not the text."""
    sdfg, tasklet = _tasklet_sdfg(body, {"_cond": "COND", "_t": "T", "_e": "E"})
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n == 1
    assert tasklet.code.as_string == "_o = ITE(_cond, _t, _e)"


def test_negative_no_ternary_left_alone():
    """A tasklet with no ternary at all is NOT mutated: ``apply_pass`` returns ``None`` and
    the body is byte-identical (pass-does-not-apply-must-not-mutate contract)."""
    sdfg, tasklet = _tasklet_sdfg("_o = _a + _b", {"_a": "A", "_b": "B"})
    original = tasklet.code.as_string
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n is None
    assert tasklet.code.as_string == original


def test_negative_target_not_out_connector_left_alone():
    """The assignment target must be a real out-connector -- a malformed tasklet whose sole
    statement writes to a name that is NOT an out-connector is left alone rather than guessed
    at."""
    sdfg = dace.SDFG("malformed_fixture")
    sdfg.add_array("COND", (1, ), dace.float64, transient=False)
    sdfg.add_array("T", (1, ), dace.float64, transient=False)
    sdfg.add_array("E", (1, ), dace.float64, transient=False)
    sdfg.add_array("O", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    # Out-connector is "_o", but the body assigns to "_other" -- not a real output write.
    tasklet = state.add_tasklet("t", {"_cond", "_t", "_e"}, {"_o"}, "_other = _t if _cond else _e")
    for conn, arr in (("_cond", "COND"), ("_t", "T"), ("_e", "E")):
        an = state.add_access(arr)
        state.add_edge(an, None, tasklet, conn, Memlet(f"{arr}[0]"))
    original = tasklet.code.as_string
    n = NormalizeTernaryTasklets().apply_pass(sdfg, {})
    assert n is None
    assert tasklet.code.as_string == original


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
