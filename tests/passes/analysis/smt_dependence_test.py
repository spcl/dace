# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Verdicts of the SMT dependence oracle.

The oracle answers three questions for ``LoopToMap``: is a write injective over the iteration
domain, and does a read/write pair carry a RAW, a WAR, or nothing. Every case below is one the
affine classifier cannot decide, and each asserts the exact verdict rather than "not None" --
a wrong verdict here is a miscompile (``RAW`` reported as ``none`` parallelizes a recurrence),
not a missed optimization.
"""
import pytest

import sympy as sp

from dace import symbolic
from dace.transformation.passes.analysis import smt_dependence

pytestmark = pytest.mark.skipif(not smt_dependence.has_z3(), reason='needs z3')

I = symbolic.pystr_to_symbolic('i')
N = symbolic.pystr_to_symbolic('N')


def test_injective_writes_are_proven():
    """``2*i`` and ``i*i`` touch a distinct element per iteration on ``[0, N]``."""
    assert smt_dependence.prove_injective_write(2 * I, 'i', 0, N, 1) is True
    assert smt_dependence.prove_injective_write(I * I, 'i', 0, N, 1) is True


def test_colliding_write_is_refused():
    """``i % 2`` maps every iteration onto two slots, so the oracle must NOT certify it."""
    assert smt_dependence.prove_injective_write(sp.Mod(I, 2), 'i', 0, N, 1) is not True


def test_read_behind_the_write_is_a_raw():
    """``A[i-1]`` read against ``A[i]`` written: iteration ``k`` reads what ``k-1`` wrote."""
    assert smt_dependence.classify_read_write_pair(I - 1, I, 'i', 0, N) == 'RAW'


def test_read_ahead_of_the_write_is_a_war():
    """``A[i+1]`` read against ``A[i]`` written: the alias only ever runs read-before-write."""
    assert smt_dependence.classify_read_write_pair(I + 1, I, 'i', 0, N) == 'WAR'


def test_disjoint_accesses_carry_nothing():
    """The read sits past the end of the write's range for every in-domain iteration."""
    assert smt_dependence.classify_read_write_pair(I + N + 1, I, 'i', 0, N) == 'none'


def test_a_guard_can_break_the_raw_chain():
    """Guarding the read with ``i > N`` empties it of in-domain iterations, so no RAW remains
    -- the shape the oracle exists for, where the guard and not the subscript decides."""
    guard = sp.StrictGreaterThan(I, N)
    assert smt_dependence.classify_read_write_pair(I - 1, I, 'i', 0, N, read_guard=guard) == 'none'


def test_an_untranslatable_expression_is_inconclusive():
    """An opaque function is not modeled; the caller must fall back, not get a verdict."""
    opaque = sp.Function('f')(I)
    assert smt_dependence.classify_read_write_pair(opaque, I, 'i', 0, N) is None
    assert smt_dependence.prove_injective_write(opaque, 'i', 0, N, 1) is not True


def test_read_ahead_excludes_the_readers_own_iteration():
    """``prove_read_ahead`` is what a snapshot rewrite needs, and no verdict from
    :func:`classify_read_write_pair` implies it -- not even the strongest one.

    ``A[i]`` read against ``A[i]`` written classifies ``'none'``: two DIFFERENT iterations never
    touch a common element. The read still aliases the write of its OWN iteration, so redirecting
    it to a pre-loop snapshot would hand it the stale original instead of the value that iteration
    just produced. Only the read-ahead question excludes that, and it does."""
    assert smt_dependence.classify_read_write_pair(I, I, 'i', 0, N) == 'none'
    assert smt_dependence.prove_read_ahead(I, I, 'i', 0, N) is not True
    assert smt_dependence.prove_read_ahead(I + 1, I, 'i', 0, N) is True


def test_read_ahead_reads_the_guard():
    """A non-affine subscript with no guard could land anywhere, including on an element an
    earlier iteration wrote; the same subscript under ``P > i`` cannot. The guard, not the
    subscript, is what makes the second one provable -- the ``A[IDX[i]*IDX[i-1]]`` shape."""
    p = symbolic.pystr_to_symbolic('P')
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N) is not True
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N, read_guard=sp.StrictGreaterThan(p, I)) is True


def test_dace_connectives_reach_the_solver():
    """``pystr_to_symbolic`` keeps ``and`` / ``or`` / ``not`` as DaCe function nodes rather than
    folding them to sympy's, so a guard collected off a ``ConditionalBlock`` arrives in that form.
    It used to fall through the translator to ``None``, which silently disarmed every
    guard-dependent proof -- the failure mode is a REFUSAL, so nothing ever flagged it."""
    guard = symbolic.pystr_to_symbolic('(P > i) and (P < N)')
    assert str(guard.func) == 'AND', guard
    p = symbolic.pystr_to_symbolic('P')
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N, read_guard=guard) is True


def test_chunked_ranges_are_proven_disjoint():
    """The anti-dependence chunk shape: iteration ``i`` owns ``[i, Min(N - 2, i + 4095)]`` and the
    loop steps by 4096, so the intervals tile the array without overlapping. The tail clamp is the
    whole point -- it is what defeats the affine ``a*i+b`` matcher, so the oracle has to carry the
    ``Min`` through to the solver rather than give up on it."""
    hi = sp.Min(N - 2, I + 4095)
    assert smt_dependence.prove_disjoint_write_ranges(I, hi, 'i', 1, N - 2, 4096) is True


def test_the_same_chunk_shape_at_unit_step_is_refused():
    """Identical intervals, ``step = 1``: every chunk now overlaps its neighbour, so the verdict
    has to flip. This is what pins the step onto the solver's domain -- without that constraint
    the oracle sees the same formula for both loops and cannot answer them differently."""
    hi = sp.Min(N - 2, I + 4095)
    assert smt_dependence.prove_disjoint_write_ranges(I, hi, 'i', 1, N - 2, 1) is False


def test_overlapping_chunks_are_refused():
    """Same shape, one element too wide: chunk ``i`` reaches ``i + 4096`` while chunk ``i + 4096``
    starts there, so consecutive chunks share an element and the loop is NOT parallel. The oracle
    must refuse -- certifying this would parallelize a genuine write-write conflict."""
    assert smt_dependence.prove_disjoint_write_ranges(I, I + 4096, 'i', 1, N - 2, 4096) is False


def test_a_range_wider_than_the_step_is_refused():
    """A unit-step loop writing ``[i, i + 1]`` overlaps every neighbour. Refusing is the whole
    guard against reading interval disjointness off the lower bound alone."""
    assert smt_dependence.prove_disjoint_write_ranges(I, I + 1, 'i', 0, N, 1) is False


def test_an_empty_interval_never_collides():
    """``[i + 1, i]`` is empty, so no iteration writes anything and disjointness is vacuous. The
    intersection test has to yield that on its own rather than needing an emptiness special case."""
    assert smt_dependence.prove_disjoint_write_ranges(I + 1, I, 'i', 0, N, 1) is True


M = symbolic.pystr_to_symbolic('M')


def test_mirrored_triangular_boxes_meet_only_inside_one_iteration():
    """Polybench covariance: iteration ``i`` writes row tail ``cov[i, i:M]`` and column tail
    ``cov[i:M, i]``. An element in both needs ``i' >= i`` from the column and ``i >= i'`` from the
    row, so the two boxes can only meet when the iterations coincide. Neither box is a point and
    neither dimension is disjoint on its own, so this is the case no other test in the file covers
    and the one that decides whether covariance parallelizes at all."""
    row = [(I, I), (I, M - 1)]
    col = [(I, M - 1), (I, I)]
    assert smt_dependence.prove_disjoint_access_boxes(row, col, 'i', 0, M - 1, 1) is True


def test_full_row_against_full_column_is_refused():
    """Drop the triangular clamp and the same shape becomes a genuine conflict: ``cov[i, 0:M]``
    and ``cov[0:M, i]`` share element ``(i, i')`` for EVERY pair. Refusing this is what stops the
    certificate from being read off the transpose alone."""
    row = [(I, I), (sp.Integer(0), M - 1)]
    col = [(sp.Integer(0), M - 1), (I, I)]
    assert smt_dependence.prove_disjoint_access_boxes(row, col, 'i', 0, M - 1, 1) is False


def test_one_dimension_disjoint_settles_the_whole_box():
    """A box needs only ONE dimension to separate. ``a[2*i, 0:M]`` against ``a[2*i + 1, 0:M]``
    splits the rows even/odd, so the boxes never alias however total the second dimension is."""
    even = [(2 * I, 2 * I), (sp.Integer(0), M - 1)]
    odd = [(2 * I + 1, 2 * I + 1), (sp.Integer(0), M - 1)]
    assert smt_dependence.prove_disjoint_access_boxes(even, odd, 'i', 0, M - 1, 1) is True


def test_a_shifted_row_pair_is_refused():
    """The near miss of the case above: ``a[i, :]`` and ``a[i + 1, :]`` DO collide, because
    iteration ``i`` and iteration ``i + 1`` both land on row ``i + 1``. Parallelizing that is a
    write-write race, so the split must come from the indices, not from the pair being distinct."""
    here = [(I, I), (sp.Integer(0), M - 1)]
    next_ = [(I + 1, I + 1), (sp.Integer(0), M - 1)]
    assert smt_dependence.prove_disjoint_access_boxes(here, next_, 'i', 0, M - 1, 1) is False


def test_neighbouring_row_bands_are_refused():
    """``a[i:i+2, :]`` against itself: consecutive iterations share a row, so the boxes overlap
    and the oracle must refuse. Guards the intersection test against being read off the lower
    bounds alone, the box analogue of the interval case above."""
    band = [(I, I + 1), (sp.Integer(0), M - 1)]
    assert smt_dependence.prove_disjoint_access_boxes(band, band, 'i', 0, M - 1, 1) is False


def test_a_backward_loop_is_reasoned_about_in_its_own_direction():
    """A negative step counts DOWN, so the iteration domain is ``end .. start``.

    Emitting ``start <= i <= end`` for it makes the domain EMPTY, and an implication with an
    unsatisfiable antecedent is vacuously valid -- so every query on a backward loop came back
    "provably disjoint". adi's backward sweep is the shape: iteration ``j`` writes column ``j``
    and reads column ``j+1``, which iteration ``j+1`` wrote, and it was certified disjoint and
    parallelized. Both directions of the same access pair are asserted here because a fix that
    merely swaps the bounds unconditionally would break the forward one instead.
    """
    j = symbolic.pystr_to_symbolic('j')
    here = [(sp.Integer(1), M - 2), (j, j)]
    ahead = [(sp.Integer(1), M - 2), (j + 1, j + 1)]
    assert smt_dependence.prove_disjoint_access_boxes(here, ahead, 'j', M - 2, 1, -1) is False
    assert smt_dependence.prove_disjoint_access_boxes(here, ahead, 'j', 1, M - 2, 1) is False


def test_an_unknown_step_direction_is_inconclusive():
    """A symbolic step could travel either way, and the wrong guess is the vacuity above -- so
    the oracle abstains instead of picking one."""
    row = [(I, I), (sp.Integer(0), M - 1)]
    col = [(I, M - 1), (I, I)]
    assert smt_dependence.prove_disjoint_access_boxes(row, col, 'i', 0, M - 1, symbolic.pystr_to_symbolic('s')) is None


def test_boxes_of_different_rank_are_inconclusive():
    """Nothing to align dimension-wise, so the oracle abstains rather than guessing."""
    assert smt_dependence.prove_disjoint_access_boxes([(I, I)], [(I, I), (I, I)], 'i', 0, M - 1, 1) is None


def test_an_indirect_read_reaches_the_solver():
    """``A[IDX[i]]`` read against ``A[i]`` written: the subscript becomes a Select, not an exception.

    The rank of a cached z3 array is measured off its SORT. Classifying that sort with the
    expression-level ``z3.is_array_sort`` raises ``ast is not an expression`` and takes every
    subscripted read down with it, so this asserts the verdict, not merely that nothing raised.
    """
    assert smt_dependence.classify_read_write_pair(symbolic.pystr_to_symbolic('IDX[i]'), I, 'i', 0, N) == 'RAW'


def test_one_name_read_at_two_ranks_is_refused():
    """``A[IDX[i]]`` against ``A[IDX[i, i]]``: the array cache is keyed by NAME, so one rank's Select
    chain would be applied to the other's sort. z3 does not reject the ill-sorted term -- it
    segfaults inside the solver -- so the oracle has to abstain here."""
    assert smt_dependence.classify_read_write_pair(symbolic.pystr_to_symbolic('IDX[i]'),
                                                   symbolic.pystr_to_symbolic('IDX[i, i]'), 'i', 0, N) is None


if __name__ == '__main__':
    test_injective_writes_are_proven()
    test_colliding_write_is_refused()
    test_read_behind_the_write_is_a_raw()
    test_read_ahead_of_the_write_is_a_war()
    test_disjoint_accesses_carry_nothing()
    test_a_guard_can_break_the_raw_chain()
    test_an_untranslatable_expression_is_inconclusive()
    test_chunked_ranges_are_proven_disjoint()
    test_the_same_chunk_shape_at_unit_step_is_refused()
    test_overlapping_chunks_are_refused()
    test_a_range_wider_than_the_step_is_refused()
    test_an_empty_interval_never_collides()
    test_mirrored_triangular_boxes_meet_only_inside_one_iteration()
    test_full_row_against_full_column_is_refused()
    test_one_dimension_disjoint_settles_the_whole_box()
    test_a_shifted_row_pair_is_refused()
    test_neighbouring_row_bands_are_refused()
    test_a_backward_loop_is_reasoned_about_in_its_own_direction()
    test_an_unknown_step_direction_is_inconclusive()
    test_boxes_of_different_rank_are_inconclusive()
    test_an_indirect_read_reaches_the_solver()
    test_one_name_read_at_two_ranks_is_refused()
