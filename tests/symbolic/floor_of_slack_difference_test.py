# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Proofs over floored quotients. Two nonnegative symbols become two slacks of the same name. Rebuilding an expression over integer symbols by name
merged them, so ``int_floor(t - p, 8)`` folded as an exact division (``t - p``) and facts with a nonempty solution
(``8*floor((t - p)/8) <= k <= t - p - 1``, the tail of a skewed loop in LU) were rejected as inconsistent."""

from dace import symbolic
from dace.symbolic import Facts, Relation, RelationKind


def test_the_tail_of_a_strip_mined_loop_is_consistent_over_nonnegative_bounds():
    t, p, k = (symbolic.symbol(name) for name in "tpk")
    facts = Facts(
        (
            Relation(RelationKind.LE, 0, t),
            Relation(RelationKind.LE, 0, p),
            Relation(RelationKind.LE, 8 * symbolic.int_floor(t - p, 8), k),
            Relation(RelationKind.LE, k, t - p - 1),
        ),
        frozenset("tpk"),
    )
    assert facts.relations


def test_a_floored_quotient_over_a_symbolic_divisor_is_nonnegative():
    """``N >= 0, M > 0`` prove ``N // M >= 0``: the quotient's linear lower bound ``(N - M + 1) / M`` is no linear
    expression over a symbolic ``M``, so the proof also needs the plain ``0`` bound (TSVC read-ahead ``a[i + N // M]``)."""
    n, m = symbolic.symbol("N"), symbolic.symbol("M")
    facts = Facts((Relation(RelationKind.LE, 0, n), Relation(RelationKind.LT, 0, m)), frozenset({"N", "M"}))
    assert symbolic.provably_nonnegative(symbolic.int_floor(n, m), facts)


if __name__ == "__main__":
    test_the_tail_of_a_strip_mined_loop_is_consistent_over_nonnegative_bounds()
    test_a_floored_quotient_over_a_symbolic_divisor_is_nonnegative()
