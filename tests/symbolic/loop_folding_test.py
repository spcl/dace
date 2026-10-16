# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The loop-shaped expressions that must fold: range sizes, integer division and extrema. Symbols carry no
assumptions of their own, so they fold from what the IR guarantees by definition, or from explicit facts."""

import sympy

import dace
from dace import subsets, symbolic
from dace.codegen.targets.cpp import sym2cpp

N, M, K, i = (dace.symbol(name) for name in ("N", "M", "K", "i"))


def size(begin, end, step=1):
    return subsets.Range([(begin, end, step)]).size()[0]


def test_unit_step_range_size_is_its_extent():
    assert size(0, N - 1) == N
    assert size(i, N - 1) == N - i
    assert size(i, sympy.Min(N, i + 32) - 1) == sympy.Min(N, i + 32) - i
    assert subsets.Range([(0, N - 1, 1), (0, M - 1, 1)]).num_elements() == M * N
    assert sym2cpp(size(0, N - 1)) == "N"


def test_strided_range_size_rounds_up():
    assert size(0, N - 1, 2) == sympy.ceiling(N / 2)
    assert sym2cpp(size(0, N - 1, 2)) == "int_ceil(N, 2)"
    assert sym2cpp(size(0, N - 1, K)) == "int_ceil(N, K)"


def test_exact_integer_division_folds():
    assert symbolic.int_floor(N, 1) == N
    assert symbolic.int_ceil(N, 1) == N
    assert symbolic.int_floor(2 * N, 2) == N
    assert symbolic.int_ceil(2 * N, 2) == N
    assert symbolic.int_floor(4 * N + 8, 4) == N + 2
    assert sym2cpp(symbolic.int_floor(N, K)) == "(N / K)"


def test_inexact_integer_division_stays():
    assert isinstance(symbolic.int_floor(N, 2), symbolic.int_floor)
    assert isinstance(symbolic.int_ceil(N + 1, 2), symbolic.int_ceil)


def test_extrema_of_ordered_arguments_fold():
    assert sympy.Min(N, N + 1) == N
    assert sympy.Max(N, N - 1) == N
    assert sympy.Min(N, M) == sympy.Min(M, N)


def facts(*relations: symbolic.Relation, integers: str = "NMKi") -> symbolic.Facts:
    return symbolic.Facts(relations, frozenset(integers))


def lt(lhs, rhs) -> symbolic.Relation:
    return symbolic.Relation(symbolic.RelationKind.LT, sympy.sympify(lhs), sympy.sympify(rhs))


def test_integer_facts_fold_rounding():
    assert symbolic.simplify(sympy.ceiling(N), facts()) == N
    assert symbolic.simplify(sympy.floor(N + 1), facts()) == N + 1
    assert symbolic.simplify(sympy.Mod(2 * N, 2), facts()) == 0
    # Nothing is assumed without facts
    assert symbolic.simplify(sympy.ceiling(N), symbolic.Facts.none()) == sympy.ceiling(N)


def test_sign_facts_fold_extrema():
    positive = facts(lt(0, K))
    assert symbolic.simplify(sympy.Max(0, K), positive) == K
    assert symbolic.simplify(sympy.Min(0, K), positive) == 0
    assert symbolic.simplify(sympy.Max(0, K), facts()) == sympy.Max(0, K)


def test_relations_fold_extrema():
    # N > 5 and the loop range 0 <= i < M
    known = facts(lt(5, N), symbolic.Relation(symbolic.RelationKind.LE, sympy.Integer(0), i), lt(i, M))
    assert symbolic.simplify(sympy.Max(N, 6), known) == N
    assert symbolic.simplify(sympy.Min(i, M), known) == i
    assert symbolic.simplify(sympy.Max(0, M - 1), known) == M - 1
    # Both bounds of 3 < N < 11 are kept
    bounded = facts(lt(3, N), lt(N, 11))
    assert symbolic.simplify(sympy.Max(N, 11) + sympy.Max(N, 3), bounded) == N + 11


if __name__ == "__main__":
    test_unit_step_range_size_is_its_extent()
    test_strided_range_size_rounds_up()
    test_exact_integer_division_folds()
    test_inexact_integer_division_stays()
    test_extrema_of_ordered_arguments_fold()
    test_integer_facts_fold_rounding()
    test_sign_facts_fold_extrema()
    test_relations_fold_extrema()
