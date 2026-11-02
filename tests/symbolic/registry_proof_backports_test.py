# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Proofs the facts could not make, each found porting a downstream branch onto the registry."""

import sympy

import dace
from dace import symbolic
from dace.symbolic import Facts, Predicate, predicate_relation


def test_a_nonzero_fact_reaches_the_sympy_assumptions():
    """``P != 0`` is what lets SymPy fold a division by ``P``; the assumptions only ever carried a sign."""
    facts = Facts((predicate_relation(Predicate.NONZERO, symbolic.symbol("P")),), frozenset({"P"}))
    assert facts.assumptions("P").get("nonzero") is True


def test_a_real_ceiling_is_bounded_for_any_sign_of_its_argument():
    """SymPy's ``ceiling`` is exact whatever its argument's sign: ``ceiling(N/3) >= N/3`` needs no fact at all."""
    n = symbolic.symbol("N")
    assert symbolic.provably_nonnegative(sympy.ceiling(n / 3) - n / 3, Facts.none())


def test_a_case_split_over_a_clamp_of_a_division_is_decided():
    """``min(x, N - 1) <= max(x, 0)`` for ``x = N // 2``: the case relations hold a division, which the solver
    cannot solve for, but the claim holds for every value of ``x`` and so for this one."""
    n = symbolic.symbol("N")
    x = symbolic.int_floor(n, 2)
    assert symbolic.provably_le(sympy.Min(x, n - 1), sympy.Max(x, 0), Facts.none())


def test_a_strided_range_counts_whole_elements_on_both_sides_of_a_copy():
    """``src[0:N:2] -> [0:N/2]``: both sides hold ``ceiling(N/2)`` elements; a step of one returned the
    non-integral ``N/2`` unrounded and validation reported a dimensionality mismatch."""
    memlet = dace.Memlet("src[0:N:2] -> [0:N/2]")
    assert memlet.subset.num_elements() == memlet.other_subset.num_elements()


def test_a_sum_of_quotients_divides_once():
    """``floor(x/2 + y/2)`` printed term by term truncates twice: 0 at ``x = y = 1``, where the proof, like SymPy, has 1."""
    x, y = symbolic.symbol("x"), symbolic.symbol("y")
    assert symbolic.symstr(sympy.floor(x / 2 + y / 2), cpp_mode=True) == "(((x + y) / (2)))"
    assert symbolic.symstr(sympy.ceiling(x / 2 + y / 2), cpp_mode=True) == "(int_ceil(x + y, 2))"


def test_a_ceiling_of_a_quotient_truncates_like_int_ceil():
    """``ceiling(x/4)`` emits ``(x + 3) / 4``, which is 0 at ``x = -10``: above ``x/4 + 1`` unless ``x >= 0``."""
    x = symbolic.symbol("x", dace.int64)
    upper = symbolic.Relation(symbolic.RelationKind.LE, sympy.ceiling(x / 4), x / 4 + 1)
    assert symbolic.ask(upper, Facts.none()) is symbolic.Truth.UNKNOWN
    assert symbolic.holds(upper, Facts((predicate_relation(Predicate.NONNEGATIVE, x),), frozenset({"x"})))
    # ``int_ceil`` is ``(x + y - 1) / y``, which holds for ``y > 0`` only
    assert symbolic.symstr(sympy.ceiling(-x / 4), cpp_mode=True) == "(int_ceil(-x, 4))"


if __name__ == "__main__":
    test_a_nonzero_fact_reaches_the_sympy_assumptions()
    test_a_real_ceiling_is_bounded_for_any_sign_of_its_argument()
    test_a_case_split_over_a_clamp_of_a_division_is_decided()
    test_a_strided_range_counts_whole_elements_on_both_sides_of_a_copy()
    test_a_sum_of_quotients_divides_once()
    test_a_ceiling_of_a_quotient_truncates_like_int_ceil()
