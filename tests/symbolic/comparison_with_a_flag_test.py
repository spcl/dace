# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A comparison's truth compared with a flag symbol parses and keeps its meaning (distribution_search)."""

import pytest

from dace import symbolic


@pytest.mark.parametrize(
    "expr, truth",
    [
        ("(a > 0) != b", {(1, True): False, (-1, True): True}),
        ("(a > 0) == b", {(1, True): True, (-1, True): False}),
        ("b != (a > 0)", {(1, True): False, (-1, True): True}),
        ("(a > 0) != 1", {(1, True): False, (-1, True): True}),
    ],
)
def test_a_comparison_against_a_flag_parses_and_keeps_its_meaning(expr: str, truth: dict) -> None:
    """``over != b_top`` once read the flag ``b_top`` as a literal and raised ``'Name' object has no attribute``."""
    parsed = symbolic.pystr_to_symbolic(expr)
    for (a, b), want in truth.items():
        assert bool(parsed.subs({"a": a, "b": b})) is want
