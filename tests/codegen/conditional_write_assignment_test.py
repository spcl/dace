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

from dace.codegen import cppunparse


def _unparse(code: str) -> str:
    return cppunparse.cppunparse(ast.parse(code), expr_semicolon=True)


def test_connector_target_is_guarded():
    assert _unparse('_o = IT(_c, _t)') == 'if (_c) { _o = _t; }'


def test_substituted_scalar_target_is_guarded():
    """A dynamic memlet substitutes the connector with its destination, which this unparser never
    declared -- lowering must not depend on knowing that the name is declared, because falling
    through emits a call to a function that does not exist."""
    assert _unparse('s = IT(_c, _t)') == 'if (_c) { s = _t; }'


def test_subscript_target_is_guarded():
    assert _unparse('_o[0] = IT(_c, _t)') == 'if (_c) { _o[0] = _t; }'


def test_codegen_built_target_is_guarded():
    """The path codegen takes for a dynamic write, where the target is already rendered C++."""
    value = ast.parse('IT(_c, _t)', mode='eval').body
    assert cppunparse.cpp_assignment('__state->s', value) == 'if (_c) { __state->s = _t; }'
    assert cppunparse.cpp_assignment('a[i + 1]', value) == 'if (_c) { a[i + 1] = _t; }'


def test_an_ordinary_value_is_a_plain_assignment():
    value = ast.parse('_a + _b', mode='eval').body
    assert cppunparse.cpp_assignment('a[i]', value) == 'a[i] = (_a + _b);'


def test_a_three_argument_call_is_not_a_conditional_write():
    """``ITE`` is the blend and stays a function call; only the two-argument ``IT`` is a
    statement."""
    assert 'if (' not in _unparse('_o = ITE(_c, _t, _e)')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
