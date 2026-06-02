# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``utils/code_rewrite.py``.

The functions under test are pure string/AST transformers — ideal for
parametrized ``(input, expected_output)`` tables. Each row encodes a
*documented contract case* (the docstrings explicitly name multi-
occurrence, no-match, nested brackets, quoted strings, graceful syntax-
error fallback, lane-suffix vs offset, etc.). Rows are intentionally
shape-driven, not branch-driven: if the implementation is rewritten
(e.g. a different AST visitor strategy) the rows still encode the same
user-visible contract.
"""
import pytest

from dace.transformation.passes.vectorization.utils.code_rewrite import (
    drop_dims_from_str,
    extract_bracket_contents,
    offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)


@pytest.mark.parametrize(
    "expr,name,full,parts",
    [
        # Plain multi-dim access.
        ("A[i, j+1, k]", "A", "A[i, j+1, k]", ["i", "j+1", "k"]),
        # Nested inside an outer function call + sibling bracket — still finds A.
        ("foo(B[i, j], A[k])", "A", "A[k]", ["k"]),
        # Quoted strings and nested parens: commas inside them must NOT split.
        ('A["i,j", (k,m)]', "A", 'A["i,j", (k,m)]', ['"i,j"', "(k,m)"]),
        # Missing name: empty match + empty parts.
        ("B[i]", "A", "", []),
        # Single dim.
        ("Y[i]", "Y", "Y[i]", ["i"]),
    ])
def test_extract_bracket_contents(expr, name, full, parts):
    assert extract_bracket_contents(expr, name) == (full, parts)


@pytest.mark.parametrize(
    "src,mask,name,expected",
    [
        # Single occurrence, drop one dim.
        ("A[i, j]", (1, 0), "A", "A[i]"),
        # Multi-occurrence in one expression: BOTH must be rewritten (the
        # Phase-0 audit bug #8 was "regex rewrote only the first" — this row
        # would have failed under that bug).
        ("A[i, j] + A[i, j]", (1, 0), "A", "A[i] + A[i]"),
        # Name doesn't match: input untouched.
        ("A[i, j]", (1, 0), "B", "A[i, j]"),
        # 3D → 2D with mid-dim kept.
        ("A[i, j, k]", (1, 0, 1), "A", "A[i, k]"),
        # Inside a function call (recurses into arg lists).
        ("foo(A[i, j], A[i, j])", (0, 1), "A", "foo(A[j], A[j])"),
        # Non-Python input: graceful fallback (returns unchanged).
        ("not python !!!", (1, 0), "A", "not python !!!"),
    ])
def test_drop_dims_from_str(src, mask, name, expected):
    assert drop_dims_from_str(src, mask, name) == expected


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


def test_drop_dims_from_str_idempotent_after_full_collapse():
    """Once an access has been collapsed to all-kept dims, re-applying
    the same mask is a no-op (the indices are already correct)."""
    once = drop_dims_from_str("A[i, j]", (1, 0), "A")
    twice = drop_dims_from_str(once, (1, ), "A")
    assert twice == "A[i]"
