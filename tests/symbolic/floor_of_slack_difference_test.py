# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Two nonnegative symbols become two slacks of the same name. Rebuilding an expression over integer symbols by name
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


if __name__ == "__main__":
    test_the_tail_of_a_strip_mined_loop_is_consistent_over_nonnegative_bounds()
