# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``utils.reductions`` (R-1: identity table + emitters).

``reductions.py`` is the single source of truth for how a W-wide vector
accumulator is folded to a scalar. The emitters are pure string builders;
these tests pin (a) the identity-element table, (b) the exact emitted
shape per operator class (infix chain vs. function-call nest vs. tree),
(c) the ``W == 1`` short-circuit, (d) the validation raises, and (e)
numerical equivalence of the emitted expression against a reference fold
for every supported operator.
"""
import math

import pytest

from dace.transformation.passes.vectorization.utils.reductions import (
    IDENTITY,
    emit_chain_reduction,
    emit_tree_reduction,
)

_INFIX = ["+", "-", "*", "/", "&", "|", "^"]
_FUNCALL = ["max", "min"]
_ALL = _INFIX + _FUNCALL
# ``-`` and ``/`` are valid chain *syntax* but are not identity-bearing
# (associative) reductions, so they are deliberately absent from IDENTITY.
_IDENTITY_OPS = ["+", "*", "&", "|", "^", "max", "min"]


def _ref_fold(op: str, values):
    """Left-associated reference fold matching the chain emitter."""
    import operator
    binop = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "&": operator.and_,
        "|": operator.or_,
        "^": operator.xor,
        "max": max,
        "min": min,
    }[op]
    acc = values[0]
    for v in values[1:]:
        acc = binop(acc, v)
    return acc


def _eval(expr: str, input_var: str, values):
    """Evaluate an emitted expression with ``input_var`` bound to ``values``."""
    return eval(expr, {"max": max, "min": min}, {input_var: list(values)})


def test_identity_table_is_complete_and_correct():
    assert set(IDENTITY) == set(_IDENTITY_OPS)
    assert IDENTITY["+"] == "0"
    assert IDENTITY["*"] == "1"
    assert IDENTITY["&"] == "~0"
    assert IDENTITY["|"] == "0"
    assert IDENTITY["^"] == "0"
    assert IDENTITY["max"] == "-inf"
    assert IDENTITY["min"] == "+inf"


@pytest.mark.parametrize("op", _INFIX)
def test_chain_infix_shape(op):
    assert emit_chain_reduction("_in", 4, op) == f"_in[0] {op} _in[1] {op} _in[2] {op} _in[3]"


@pytest.mark.parametrize("op", _FUNCALL)
def test_chain_funcall_shape(op):
    # Function-call ops nest left-associatively (no infix syntax for max/min).
    assert emit_chain_reduction("_in", 4, op) == f"{op}({op}({op}(_in[0], _in[1]), _in[2]), _in[3])"


def test_tree_infix_shape_pow2():
    assert emit_tree_reduction("_in", 4, "+") == "((_in[0] + _in[1]) + (_in[2] + _in[3]))"


def test_tree_funcall_shape_odd_width_trails_last_lane():
    # W=5: (0,1)(2,3) pair, lane 4 trails -> level2 ((01),(23)),(4) -> final.
    assert emit_tree_reduction("_in", 5, "max") == "max(max(max(_in[0], _in[1]), max(_in[2], _in[3])), _in[4])"


@pytest.mark.parametrize("op", _ALL)
def test_width_one_short_circuits_to_single_lane(op):
    assert emit_chain_reduction("_in", 1, op) == "_in[0]"
    assert emit_tree_reduction("_in", 1, op) == "_in[0]"


def test_validate_rejects_unsupported_op():
    with pytest.raises(NotImplementedError, match="unsupported op"):
        emit_chain_reduction("_in", 4, "//")
    with pytest.raises(NotImplementedError, match="unsupported op"):
        emit_tree_reduction("_in", 4, "matmul")


def test_validate_rejects_nonpositive_width():
    with pytest.raises(ValueError, match="vector_width must be >= 1"):
        emit_chain_reduction("_in", 0, "+")
    with pytest.raises(ValueError, match="vector_width must be >= 1"):
        emit_tree_reduction("_in", -1, "*")


@pytest.mark.parametrize("op", ["+", "*", "max", "min"])
def test_chain_evaluates_to_reference_fold(op):
    vals = [1.0, 3.0, 2.0, 5.0, 4.0, 0.5, 8.0, 6.0]
    got = _eval(emit_chain_reduction("_in", 8, op), "_in", vals)
    assert math.isclose(got, _ref_fold(op, vals), rel_tol=1e-12)


@pytest.mark.parametrize("op", ["&", "|", "^"])
def test_chain_bitwise_evaluates_to_reference_fold(op):
    vals = [0b1011, 0b0110, 0b1100, 0b0101]
    got = _eval(emit_chain_reduction("_in", 4, op), "_in", vals)
    assert got == _ref_fold(op, vals)


@pytest.mark.parametrize("op", ["+", "*", "max", "min"])
def test_tree_equals_chain_for_associative_ops(op):
    # +, *, max, min are associative: the balanced tree and the linear
    # chain must produce the same value (float values chosen so + / * do
    # not reassociate measurably).
    vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    chain = _eval(emit_chain_reduction("_in", 8, op), "_in", vals)
    tree = _eval(emit_tree_reduction("_in", 8, op), "_in", vals)
    assert math.isclose(chain, tree, rel_tol=1e-12)


@pytest.mark.parametrize("op", ["max", "min"])
def test_tree_odd_width_evaluates_correctly(op):
    vals = [3.0, 1.0, 4.0, 1.0, 5.0]
    got = _eval(emit_tree_reduction("_in", 5, op), "_in", vals)
    assert got == _ref_fold(op, vals)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
