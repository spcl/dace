# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``out = IT(cond, value)`` must lower to a guarded statement on every assignment path.

``IT`` is a write-only conditional write, not a C++ function (there is no value to return when the
predicate is false), so any path that emits it as a call produces code that does not compile.
:meth:`CPPUnparser._Assign` covers assignments unparsed from a Python AST, but codegen builds the
destination itself for a dynamic memlet -- a pointer name, a view expression, an array-interface
subscript -- and used to glue ``"%s = %s;"`` around an unparsed right-hand side, which skipped the
assignment unparser entirely. :func:`dace.codegen.cppunparse.cpp_assignment` is the one place both
paths share.
"""

import ast

import pytest

import dace
from dace.codegen import cppunparse


def _unparse(code: str) -> str:
    return cppunparse.cppunparse(ast.parse(code), expr_semicolon=True)


def test_connector_target_is_guarded():
    assert _unparse("_o = IT(_c, _t)") == "if (_c) { _o = _t; }"


def test_substituted_scalar_target_is_guarded():
    """A dynamic memlet substitutes the connector with its destination, which this unparser never
    declared -- lowering must not depend on knowing that the name is declared, because falling
    through emits a call to a function that does not exist."""
    assert _unparse("s = IT(_c, _t)") == "if (_c) { s = _t; }"


def test_subscript_target_is_guarded():
    assert _unparse("_o[0] = IT(_c, _t)") == "if (_c) { _o[0] = _t; }"


def test_codegen_built_target_is_guarded():
    """The path codegen takes for a dynamic write, where the target is already rendered C++."""
    value = ast.parse("IT(_c, _t)", mode="eval").body
    assert cppunparse.cpp_assignment("__state->s", value) == "if (_c) { __state->s = _t; }"
    assert cppunparse.cpp_assignment("a[i + 1]", value) == "if (_c) { a[i + 1] = _t; }"


def test_an_ordinary_value_is_a_plain_assignment():
    value = ast.parse("_a + _b", mode="eval").body
    assert cppunparse.cpp_assignment("a[i]", value) == "a[i] = (_a + _b);"


def test_a_three_argument_call_is_not_a_conditional_write():
    """``ITE`` is the blend and stays a function call; only the two-argument ``IT`` is a
    statement."""
    assert "if (" not in _unparse("_o = ITE(_c, _t, _e)")


def test_a_local_the_readable_generator_writes_directly_is_guarded():
    """CloudSC's nested BranchNormalization: ``IT`` writes a transient the readable CPU generator names directly."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("it_into_a_local")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.bool_)
    sdfg.add_array("tmp", [1], dace.float64, transient=True)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    masked = state.add_tasklet("masked", {"_c": None, "_new": None}, {"_o": None}, "_o = IT(_c, _new)")
    state.add_memlet_path(state.add_read("C"), entry, masked, dst_conn="_c", memlet=dace.Memlet("C[i]"))
    state.add_memlet_path(state.add_read("A"), entry, masked, dst_conn="_new", memlet=dace.Memlet("A[i]"))
    tmp = state.add_access("tmp")
    state.add_edge(masked, "_o", tmp, None, dace.Memlet("tmp[0]"))
    copy = state.add_tasklet("copy", {"x": None}, {"y": None}, "y = x")
    state.add_edge(tmp, None, copy, "x", dace.Memlet("tmp[0]"))
    state.add_memlet_path(copy, exit_, state.add_write("A"), src_conn="y", memlet=dace.Memlet("A[i]"))
    with dace.config.set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = sdfg.generate_code()[0].clean_code
    assert "IT(" not in code
    assert "if (C[" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
