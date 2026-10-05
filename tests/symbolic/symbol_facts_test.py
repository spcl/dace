# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import cast

import pytest
import sympy

from dace import dtypes, symbolic
from dace.symbolic_facts import (Facts, InconsistentAssumptionsError, Predicate, Relation, RelationKind, Truth, ask,
                                 ask_predicate)

N = symbolic.symbol('N', dtypes.int64)
K = symbolic.symbol('K', dtypes.int32)
M = symbolic.symbol('M', dtypes.int64)
ITERATOR = symbolic.symbol('i', dtypes.int32)
INTEGERS = frozenset({'N', 'K', 'M', 'i'})


def relations_only(*relations: Relation) -> Facts:
    return Facts({}, relations, INTEGERS)


def test_positive_integer_is_at_least_one():
    facts = Facts({'N': frozenset({Predicate.POSITIVE})}, (), INTEGERS)
    assert ask_predicate(Predicate.NONNEGATIVE, N - 1, facts) is Truth.TRUE


def test_strict_relation_implies_weak():
    facts = relations_only(Relation(RelationKind.LT, K, N))
    assert ask(Relation(RelationKind.LE, K, N), facts) is Truth.TRUE


def test_relations_chain_transitively():
    facts = relations_only(Relation(RelationKind.LT, K, N), Relation(RelationKind.LE, N, M))
    assert ask(Relation(RelationKind.LT, K, M), facts) is Truth.TRUE


def test_iterator_range_bounds_the_last_index():
    facts = relations_only(Relation(RelationKind.LE, sympy.Integer(0), ITERATOR),
                           Relation(RelationKind.LT, ITERATOR, N))
    assert ask(Relation(RelationKind.LE, ITERATOR, N - 1), facts) is Truth.TRUE


def test_contrary_relation_is_false():
    facts = relations_only(Relation(RelationKind.LT, K, N))
    assert ask(Relation(RelationKind.LE, N, K), facts) is Truth.FALSE


def test_equality_is_proven():
    facts = relations_only(Relation(RelationKind.EQ, N, M))
    assert ask(Relation(RelationKind.EQ, N, M), facts) is Truth.TRUE


def test_no_facts_prove_nothing():
    assert ask_predicate(Predicate.NONNEGATIVE, N, Facts.none()) is Truth.UNKNOWN


def test_strictness_needs_declared_integers():
    facts = Facts({}, (Relation(RelationKind.LT, K, N), ), frozenset())
    assert ask(Relation(RelationKind.LE, K, N - 1), facts) is Truth.UNKNOWN


def test_predicates_survive_elimination():
    facts = Facts({
        'N': frozenset({Predicate.POSITIVE}),
        'K': frozenset({Predicate.POSITIVE})
    }, (Relation(RelationKind.LT, K, N), ), INTEGERS)
    assert ask_predicate(Predicate.POSITIVE, N, facts) is Truth.TRUE


def test_int_floor_sign_is_not_derived():
    facts = Facts({'N': frozenset({Predicate.NONNEGATIVE})}, (), INTEGERS)
    assert ask_predicate(Predicate.NONNEGATIVE, cast(sympy.Expr, symbolic.int_floor(N, 2)), facts) is Truth.UNKNOWN


def test_contradicting_predicates_raise():
    with pytest.raises(InconsistentAssumptionsError, match='Inconsistent facts about N'):
        Facts({'N': frozenset({Predicate.POSITIVE, Predicate.NEGATIVE})}, (), INTEGERS)


def test_contradicting_relations_raise():
    with pytest.raises(InconsistentAssumptionsError, match='Inconsistent facts about relations'):
        relations_only(Relation(RelationKind.LT, N, K), Relation(RelationKind.LT, K, N))


if __name__ == '__main__':
    test_positive_integer_is_at_least_one()
    test_strict_relation_implies_weak()
    test_relations_chain_transitively()
    test_iterator_range_bounds_the_last_index()
    test_contrary_relation_is_false()
    test_equality_is_proven()
    test_no_facts_prove_nothing()
    test_strictness_needs_declared_integers()
    test_predicates_survive_elimination()
    test_int_floor_sign_is_not_derived()
    test_contradicting_predicates_raise()
    test_contradicting_relations_raise()
