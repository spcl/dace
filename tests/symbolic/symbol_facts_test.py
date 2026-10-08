# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import cast

import pytest
import sympy

from dace import dtypes, symbolic
from dace.symbolic_facts import (
    Facts,
    InconsistentAssumptionsError,
    Predicate,
    Relation,
    RelationKind,
    Truth,
    ask,
    comparison_relation,
    predicate_relation,
)

N = symbolic.symbol("N", dtypes.int64)
K = symbolic.symbol("K", dtypes.int32)
M = symbolic.symbol("M", dtypes.int64)
ITERATOR = symbolic.symbol("i", dtypes.int32)
ZERO = sympy.Integer(0)
ONE = sympy.Integer(1)
INTEGERS = frozenset({"N", "K", "M", "i"})


def facts_of(*relations: Relation) -> Facts:
    return Facts(relations, INTEGERS)


def test_positive_integer_is_at_least_one():
    facts = facts_of(predicate_relation(Predicate.POSITIVE, N))
    assert ask(Relation(RelationKind.LE, ZERO, N - 1), facts) is Truth.TRUE


def test_strict_relation_implies_weak():
    facts = facts_of(Relation(RelationKind.LT, K, N))
    assert ask(Relation(RelationKind.LE, K, N), facts) is Truth.TRUE


def test_relations_chain_transitively():
    facts = facts_of(Relation(RelationKind.LT, K, N), Relation(RelationKind.LE, N, M))
    assert ask(Relation(RelationKind.LT, K, M), facts) is Truth.TRUE


def test_iterator_range_bounds_the_last_index():
    facts = facts_of(Relation(RelationKind.LE, ZERO, ITERATOR), Relation(RelationKind.LT, ITERATOR, N))
    assert ask(Relation(RelationKind.LE, ITERATOR, N - 1), facts) is Truth.TRUE


def test_contrary_relation_is_false():
    facts = facts_of(Relation(RelationKind.LT, K, N))
    assert ask(Relation(RelationKind.LE, N, K), facts) is Truth.FALSE


def test_equality_is_proven():
    facts = facts_of(Relation(RelationKind.EQ, N, M))
    assert ask(Relation(RelationKind.EQ, N, M), facts) is Truth.TRUE


def test_nonzero_facts_answer_inequality():
    assert ask(Relation(RelationKind.NE, N, ZERO), facts_of(predicate_relation(Predicate.NONZERO, N))) is Truth.TRUE
    assert ask(Relation(RelationKind.NE, N, K), facts_of(Relation(RelationKind.NE, K, N))) is Truth.TRUE
    assert ask(Relation(RelationKind.NE, K, N), facts_of(Relation(RelationKind.LT, K, N))) is Truth.TRUE


def test_no_facts_prove_nothing():
    assert ask(Relation(RelationKind.LE, ZERO, N), Facts.none()) is Truth.UNKNOWN
    assert ask(Relation(RelationKind.NE, N, ZERO), Facts.none()) is Truth.UNKNOWN


def test_strictness_needs_declared_integers():
    facts = Facts((Relation(RelationKind.LT, K, N),), frozenset())
    assert ask(Relation(RelationKind.LE, K, N - 1), facts) is Truth.UNKNOWN


def test_predicates_survive_elimination():
    facts = facts_of(
        predicate_relation(Predicate.POSITIVE, N),
        predicate_relation(Predicate.POSITIVE, K),
        Relation(RelationKind.LT, K, N),
    )
    assert ask(Relation(RelationKind.LT, ZERO, N), facts) is Truth.TRUE


def test_int_floor_of_nonnegative_is_bounded():
    floor = cast(sympy.Expr, symbolic.int_floor(N, 2))
    facts = facts_of(predicate_relation(Predicate.NONNEGATIVE, N))
    assert ask(Relation(RelationKind.LE, ZERO, floor), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, floor, N), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, N - 1, 2 * floor), facts) is Truth.TRUE
    assert (
        ask(Relation(RelationKind.LE, ONE, floor), facts_of(Relation(RelationKind.LE, sympy.Integer(2), N)))
        is Truth.TRUE
    )


def test_division_of_unknown_sign_is_not_bounded():
    """C++ truncates where Python rounds down, so nothing is known unless the numerator is nonnegative."""
    assert (
        ask(Relation(RelationKind.LE, ZERO, cast(sympy.Expr, symbolic.int_floor(N, 2))), Facts.none()) is Truth.UNKNOWN
    )
    shifted = sympy.Mod(ITERATOR - 2, 2, evaluate=False)
    assert (
        ask(Relation(RelationKind.LE, ZERO, shifted), facts_of(Relation(RelationKind.LE, ZERO, ITERATOR)))
        is Truth.UNKNOWN
    )
    assert (
        ask(
            Relation(RelationKind.LE, ZERO, cast(sympy.Expr, symbolic.int_floor(N, K))),
            facts_of(predicate_relation(Predicate.NONNEGATIVE, N)),
        )
        is Truth.UNKNOWN
    )


def test_int_ceil_bounds():
    ceil = cast(sympy.Expr, symbolic.int_ceil(N, 32))
    facts = facts_of(predicate_relation(Predicate.NONNEGATIVE, N))
    assert ask(Relation(RelationKind.LE, N, 32 * ceil), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, ceil, N), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, ONE, ceil), facts_of(Relation(RelationKind.LE, ONE, N))) is Truth.TRUE


def test_mod_of_iterator_is_bounded():
    remainder = sympy.Mod(ITERATOR, N)
    facts = facts_of(Relation(RelationKind.LE, ZERO, ITERATOR), Relation(RelationKind.LT, ITERATOR, N))
    assert ask(Relation(RelationKind.LE, ZERO, remainder), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LT, remainder, N), facts) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, remainder, ITERATOR), facts) is Truth.TRUE
    nested = cast(sympy.Expr, symbolic.int_floor(remainder, 2))
    assert ask(Relation(RelationKind.LT, nested, N), facts) is Truth.TRUE


def test_min_and_max_are_split_into_cases():
    assert ask(Relation(RelationKind.LE, sympy.Min(N, K), N), Facts.none()) is Truth.TRUE
    assert ask(Relation(RelationKind.LE, K, sympy.Max(N, K)), Facts.none()) is Truth.TRUE
    assert ask(Relation(RelationKind.LT, N, sympy.Min(N, K)), Facts.none()) is Truth.FALSE
    only_n = facts_of(predicate_relation(Predicate.NONNEGATIVE, N))
    assert ask(Relation(RelationKind.LE, ZERO, sympy.Min(N, K)), only_n) is Truth.UNKNOWN


def test_tiled_range_bounds_the_index():
    tile = symbolic.symbol("T", dtypes.int64)
    facts = Facts(
        (Relation(RelationKind.LE, ZERO, ITERATOR), Relation(RelationKind.LE, ITERATOR, sympy.Min(N, tile + 32) - 1)),
        INTEGERS | {"T"},
    )
    assert ask(Relation(RelationKind.LT, ITERATOR, N), facts) is Truth.TRUE


def test_contradicting_predicates_raise():
    with pytest.raises(InconsistentAssumptionsError, match="Inconsistent facts about relations"):
        facts_of(predicate_relation(Predicate.POSITIVE, N), predicate_relation(Predicate.NEGATIVE, N))


def test_contradicting_relations_raise():
    with pytest.raises(InconsistentAssumptionsError, match="Inconsistent facts about relations"):
        facts_of(Relation(RelationKind.LT, N, K), Relation(RelationKind.LT, K, N))


def test_comparisons_become_relations_with_the_smaller_side_first():
    assert comparison_relation(sympy.Gt(N, K)) == Relation(RelationKind.LT, K, N)
    assert comparison_relation(sympy.Le(N, K)) == Relation(RelationKind.LE, N, K)
    assert comparison_relation(sympy.Eq(N, K)) is None


if __name__ == "__main__":
    test_positive_integer_is_at_least_one()
    test_strict_relation_implies_weak()
    test_relations_chain_transitively()
    test_iterator_range_bounds_the_last_index()
    test_contrary_relation_is_false()
    test_equality_is_proven()
    test_nonzero_facts_answer_inequality()
    test_no_facts_prove_nothing()
    test_strictness_needs_declared_integers()
    test_predicates_survive_elimination()
    test_int_floor_of_nonnegative_is_bounded()
    test_division_of_unknown_sign_is_not_bounded()
    test_int_ceil_bounds()
    test_mod_of_iterator_is_bounded()
    test_min_and_max_are_split_into_cases()
    test_tiled_range_bounds_the_index()
    test_contradicting_predicates_raise()
    test_contradicting_relations_raise()
    test_comparisons_become_relations_with_the_smaller_side_first()
