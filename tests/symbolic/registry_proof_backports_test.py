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


if __name__ == "__main__":
    test_a_nonzero_fact_reaches_the_sympy_assumptions()
    test_a_real_ceiling_is_bounded_for_any_sign_of_its_argument()
    test_a_case_split_over_a_clamp_of_a_division_is_decided()
    test_a_strided_range_counts_whole_elements_on_both_sides_of_a_copy()
