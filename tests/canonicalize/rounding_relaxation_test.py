# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The floor/ceil relaxation that lets a subset overlap test decide an index-set split.

``subsets.Range.intersects`` decides overlap with raw sympy relationals, and sympy gives up on a
bound holding a ``floor``/``ceiling`` it cannot evaluate. So the two halves of a split at
``int_floor(N, 2)`` -- which provably cannot overlap -- read as "may overlap", and every consumer
that needs a disjointness proof (``LoopToMap`` above all) refuses them.

:func:`dace.symbolic.provably_nonnegative` closes that by REPLACING each rounding node with a bound
that makes the whole expression smaller, then asking sympy about the weakened expression:
``floor(t) in (t - 1, t]``, ``ceiling(t) in [t, t + 1)``. Because it only ever weakens, a positive
answer is a proof and a negative one is merely "not proven".

Widening ``intersects`` makes it answer more often EVERYWHERE in DaCe, so a wrong "disjoint" is a
silent miscompile far from the loop this was built for. The soundness half of this file is therefore
the important half: it pins the facts the relaxation must REFUSE, including ones that are true for
most values of ``N`` but false for some.
"""
import pytest

from dace import subsets, symbolic

N = symbolic.pystr_to_symbolic('N')
FLOOR = symbolic.int_floor(N, 2)
CEIL = symbolic.int_ceil(N, 2)


def test_relaxation_proves_both_halves_of_a_midpoint_split_disjoint():
    """The two facts an index-set split at ``int_floor(N, 2)`` needs, neither decidable by sympy.

    Lower half ``[0, x-1]`` writes ``[0, x-1]`` and reads ``[N-x, N-1]``: disjoint iff
    ``N - 2x >= 0``. Upper half ``[x+1, N-1]`` writes ``[x+1, N-1]`` and reads ``[0, N-2-x]``:
    disjoint iff ``2x + 2 - N >= 0``. Both hold for every integer ``N``, at either parity.
    """
    assert symbolic.provably_nonnegative(N - 2 * FLOOR) is True
    assert symbolic.provably_nonnegative(2 * FLOOR + 2 - N) is True


def test_relaxation_refuses_facts_that_fail_at_one_parity():
    """SOUNDNESS. Each of these holds for one parity of ``N`` and fails for the other, so a prover
    that "decided" them would report a real overlap as disjoint on half of all inputs."""
    assert symbolic.provably_nonnegative(N - 2 * FLOOR - 1) is False  # false whenever N is even
    assert symbolic.provably_nonnegative(2 * FLOOR + 1 - N) is False  # false whenever N is odd
    assert symbolic.provably_nonnegative(2 * CEIL - N - 1) is False  # false whenever N is even


def test_relaxation_refuses_plainly_false_and_undecidable_claims():
    """SOUNDNESS. Nothing about an unbounded symbol may be invented."""
    assert symbolic.provably_nonnegative(-N) is False
    assert symbolic.provably_nonnegative(N - 5) is False
    assert symbolic.provably_nonnegative(FLOOR - N) is False
    assert symbolic.provably_nonnegative(N - symbolic.pystr_to_symbolic('KOFF')) is False


def test_relaxation_refuses_a_rounding_it_cannot_bound_one_sidedly():
    """SOUNDNESS. A rounding node nested inside another's argument, or one the expression is not
    affine in, has no independent extreme -- substituting a bound there does not weaken the whole
    in a known direction, so no answer may be given."""
    nested = symbolic.int_floor(symbolic.int_floor(N, 2), 2)
    assert symbolic.provably_nonnegative(nested - N) is False
    assert symbolic.provably_nonnegative(FLOOR * FLOOR - N) is False


def test_nonnegative_symbol_contract_is_opt_in():
    """``int_floor(N, 2) >= 0`` needs ``N >= 0``, which is CANONICALIZATION's contract and not a
    property of SDFGs at large. It must not be assumed by the default (library-wide) caller."""
    assert symbolic.provably_nonnegative(FLOOR) is False
    assert symbolic.provably_nonnegative(FLOOR, assume_symbols_nonnegative=True) is True


def test_has_rounding_gates_the_expensive_path():
    """The cheap pre-filter callers on hot paths use to skip the relaxation entirely."""
    assert symbolic.has_rounding(N - 2 * FLOOR) is True
    assert symbolic.has_rounding(CEIL) is True
    assert symbolic.has_rounding(N - 1) is False
    assert symbolic.has_rounding(symbolic.pystr_to_symbolic('5')) is False


def _rng(begin, end):
    return subsets.Range([(begin, end, 1)])


def test_intersects_now_separates_the_two_halves_of_a_midpoint_split():
    """The end the whole relaxation exists for: the read range and the write range of each half of
    a split at ``int_floor(N, 2)`` are reported DISJOINT instead of indeterminate."""
    # lower half: reads [N-x, N-1], writes [0, x-1]
    assert subsets.intersects(_rng(N - FLOOR, N - 1), _rng(0, FLOOR - 1)) is False
    # upper half: reads [0, N-2-x], writes [x+1, N-1]
    assert subsets.intersects(_rng(0, N - 2 - FLOOR), _rng(FLOOR + 1, N - 1)) is False


def test_intersects_still_refuses_to_separate_ranges_that_really_touch():
    """SOUNDNESS. ``[0, x]`` and ``[x, N-1]`` share the element ``x``, and ``[0, x]`` vs
    ``[x-1, N-1]`` share two -- neither may be called disjoint. The widening must only ever turn a
    previous "undecided" into "disjoint", never manufacture one."""
    assert subsets.intersects(_rng(0, FLOOR), _rng(FLOOR, N - 1)) is not False
    assert subsets.intersects(_rng(0, FLOOR), _rng(FLOOR - 1, N - 1)) is not False
    # A genuinely overlapping pair with no rounding at all is untouched by the widening.
    assert subsets.intersects(_rng(0, N - 1), _rng(0, N - 1)) is True


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 100, 101])
def test_the_proved_facts_hold_when_evaluated(n):
    """Executable check of the two proofs above: the relaxation is only sound if the facts it
    proves are true for every concrete ``N``, so evaluate them rather than trust the algebra."""
    x = n // 2
    assert n - 2 * x >= 0
    assert 2 * x + 2 - n >= 0
    # And the halves a split at x produces really are disjoint index sets.
    lower_writes, lower_reads = set(range(0, x)), {n - 1 - i for i in range(0, x)}
    upper_writes, upper_reads = set(range(x + 1, n)), {n - 1 - i for i in range(x + 1, n)}
    assert not (lower_writes & lower_reads)
    assert not (upper_writes & upper_reads)


if __name__ == '__main__':
    test_relaxation_proves_both_halves_of_a_midpoint_split_disjoint()
    test_relaxation_refuses_facts_that_fail_at_one_parity()
    test_relaxation_refuses_plainly_false_and_undecidable_claims()
    test_relaxation_refuses_a_rounding_it_cannot_bound_one_sidedly()
    test_nonnegative_symbol_contract_is_opt_in()
    test_has_rounding_gates_the_expensive_path()
    test_intersects_now_separates_the_two_halves_of_a_midpoint_split()
    test_intersects_still_refuses_to_separate_ranges_that_really_touch()
    for _n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 100, 101]:
        test_the_proved_facts_hold_when_evaluated(_n)
    print('OK')
