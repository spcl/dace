# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``utils/code_rewrite.py``.

The functions under test are pure string/AST transformers — ideal for
parametrized ``(input, expected_output)`` tables. Each row encodes a
*documented contract case* (lane-suffix vs offset, array-subscript round-trip,
symbol-absent no-op, etc.). Rows are intentionally shape-driven, not branch-
driven: if the implementation is rewritten the rows still encode the same
user-visible contract.
"""
import pytest

from dace.transformation.passes.vectorization.utils.code_rewrite import (
    offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)


@pytest.mark.parametrize(
    "expr,sym,offset,arrays,expected",
    [
        # Matching symbol: sympy folds the (i + offset) + existing const.
        ("i + 1", "i", 5, None, "i + 6"),
        # Symbol absent from expression: input unchanged.
        ("j + 1", "i", 5, None, "j + 1"),
        # Inside an array subscript: arrays={"a"} keeps the bracket form
        # (without it, dace's symbolic printer emits ``a(i+3)`` instead).
        ("a[i]", "i", 3, {"a"}, "a[i + 3]"),
        # Zero offset is a no-op semantically (i + 0 = i).
        ("i", "i", 0, None, "i"),
    ])
def test_offset_symbol_in_expression(expr, sym, offset, arrays, expected):
    assert offset_symbol_in_expression(expr, sym, offset, arrays=arrays) == expected


@pytest.mark.parametrize(
    "expr,sym,offset,vp,expected",
    [
        # Generic symbol: gets the canonical ``_lane0id_<offset>`` chunk
        # (Option B; the legacy ``_laneid_<offset>`` form parses back-compat).
        ("idx + 0", "idx", 3, None, "idx_lane0id_3"),
        # Symbol IS the vector_map_param: offset rather than suffix.
        ("i + 0", "i", 3, "i", "i + 3"),
        # Generic symbol with vector_map_param set to a DIFFERENT name:
        # still lane-suffixed (only matches when sym == vector_map_param).
        ("idx + 0", "idx", 5, "i", "idx_lane0id_5"),
    ])
def test_use_laneid_symbol_in_expression(expr, sym, offset, vp, expected):
    assert use_laneid_symbol_in_expression(expr, sym, offset, vector_map_param=vp) == expected
