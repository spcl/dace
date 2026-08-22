# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Soundness of the ``BestEffortLoopPeeling`` index-set split: where it may cut, and what decides.

Part 1 -- an OUT-OF-RANGE split point. ``_split_loop_at`` regroups ``loop`` into
[start, x-1] + {x} + [x+1, end]. That is a pure regrouping of the same iterations only while
``start <= x <= end``. A split point is merely LOOP-INVARIANT, though -- an ``if i == K`` guard with
a free symbol ``K`` is a perfectly ordinary candidate whose position relative to the loop bounds is
undecidable at transform time. When ``K`` lands OUTSIDE the range at runtime the split must
degenerate to the original range, not invent iterations the loop never ran.

Part 2 -- the DIRECTION-FLIP crossover, where the cut comes from. A loop whose read index and write
index move AGAINST each other (TSVC s281, ``x = a[N-1-i] + ...; a[i] = x - 1; b[i] = x``) aliases at
``i2 = N-1-i1``: below ``(N-1)/2`` the write is later (an anti-dependence), above it the write
already happened (a true dependence). It is neither one thing nor the other, so no single-direction
rewrite repairs it -- but cutting at the crossover leaves each side with read and write sets that no
longer meet. Pinned here: the candidates the body nominates, that MEASUREMENT (not the derivation)
is what accepts one, that the accepted cut needs no runtime guard, and that it preserves values at
BOTH parities of the trip count -- the odd case puts the loop's fixpoint iteration inside the split
and is the one an off-by-one would corrupt.
"""
import copy

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling

N = dace.symbol('N', nonnegative=True)
K = dace.symbol('K', nonnegative=True)
LEN_1D = dace.symbol('LEN_1D', nonnegative=True)


@dace.program
def guarded_special_case(a: dace.float64[N], b: dace.float64[N]):
    # `i == K` is an index-set-split candidate; the body's `a[N - 1]` broadcast read against the
    # `a[i]` write is what keeps the whole loop off LoopToMap, so the split gets applied.
    for i in range(1, N):
        if i == K:
            a[i] = a[i] + a[N - 1]
        else:
            a[i] = b[i]


def _reference(a, b, n, k):
    """The sequential meaning: the loop starts at 1, so `i` NEVER equals a `k` below 1."""
    for i in range(1, n):
        if i == k:
            a[i] = a[i] + a[n - 1]
        else:
            a[i] = b[i]
    return a


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def test_split_point_below_loop_start_does_not_invent_an_iteration():
    """``K == 0`` is below the loop's start of 1, so the guard never fires and ``a[0]`` is
    untouched. The split must not run the carved-out ``{K}`` iteration at ``i = 0``."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 1

    n, k = 16, 0
    rng = np.random.default_rng(0)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref), f'a[0] must stay {ref[0]!r} (the loop starts at 1), got {got[0]!r}'


def test_split_point_inside_the_range_is_still_value_preserving():
    """The in-range companion: ``K == 5`` genuinely selects an iteration, and the split must
    reproduce the sequential result exactly."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)

    n, k = 16, 5
    rng = np.random.default_rng(1)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref)


@dace.program
def reverse_read_same_write(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    # TSVC s281. Reads a[LEN_1D-1-i], writes a[i]: the two indices move against each other.
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x


@dace.program
def carried_recurrence(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    # Read and write indices move TOGETHER (slopes both 1), so the dependence keeps one direction
    # for the whole run and there is no crossover to cut at.
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + b[i]


def test_crossover_candidates_are_the_floor_and_ceil_of_the_direction_flip():
    """``write(x) == read(x)`` is ``x == N-1-x``, i.e. ``x = (N-1)/2``. Both brackets are offered
    because which one leaves BOTH halves provable depends on the parity, and the pass measures.

    Both are spelled ``int_floor`` (``ceil(a,b) == int_floor(a+b-1, b)``). A bare rational would be
    distributed by the C printer and each term truncated on its own, so the split point that ran
    would not be the one just proven safe."""
    sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    points = BestEffortLoopPeeling(peel_limit=4).direction_flip_split_points(loop)
    assert points == [
        symbolic.int_floor(symbolic.pystr_to_symbolic('LEN_1D') - 1, 2),
        symbolic.int_floor(symbolic.pystr_to_symbolic('LEN_1D'), 2)
    ], points


def test_measurement_picks_the_bracket_that_unblocks_both_halves():
    """The derivation only NOMINATES; the probe on an isolated copy is what accepts. Only the ceil
    bracket leaves both range segments provably conflict-free, so that is the one kept."""
    sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    peel = BestEffortLoopPeeling(peel_limit=4)
    found = peel._best_split_for(loop, sdfg)
    assert found is not None
    x, middle_singleton, _guarded = found
    # A direction-flip crossover is one iteration wide, so this is the carve-a-singleton family,
    # not the two-way range-guard split (715dfeb83).
    assert middle_singleton is True, 'crossover split must carve the single fixpoint iteration'
    assert x == symbolic.int_floor(symbolic.pystr_to_symbolic('LEN_1D'), 2), x
    assert peel._split_range_relations(loop, x) == frozenset(), 'the split must need no runtime guard'


def test_a_singleton_only_split_is_not_counted_as_a_win():
    """The measurement must not accept a cut that parallelizes nothing.

    Splitting at a free ``K`` carves a middle ``{K}``, which ``LoopToMap`` accepts purely because it
    provably runs at most once -- no dependence is proven about it. Counting that would score a cut
    that only ADDS residual sequential loops as an improvement, so such a segment must not count."""
    sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    peel = BestEffortLoopPeeling(peel_limit=4)
    mini, _ = peel._isolate_loop(_loops(sdfg)[0], sdfg)
    assert mini is not None
    cand = copy.deepcopy(mini)
    assert peel._split_loop_at(cand, _loops(cand)[0], symbolic.pystr_to_symbolic('K'))
    peel._clean_peeled_remainder(cand)
    assert peel._mappable_loop_count(cand) == 0, 'a provably-single-iteration segment is not parallelism'


def test_the_accepted_split_counts_both_range_segments():
    """Counterpart: the accepted cut scores 2, which is only reachable if the count SSA-renames the
    segment iterators first. Sharing one name makes ``LoopToMap`` refuse every segment but the last
    ("loop-defined symbol used after the loop") -- exactly the ones a split exists to unblock."""
    sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    peel = BestEffortLoopPeeling(peel_limit=4)
    mini, _ = peel._isolate_loop(_loops(sdfg)[0], sdfg)
    assert peel._mappable_loop_count(copy.deepcopy(mini)) == 0, 'the unsplit loop parallelizes nothing'
    cand = copy.deepcopy(mini)
    assert peel._split_loop_at(cand, _loops(cand)[0], symbolic.int_floor(symbolic.pystr_to_symbolic('LEN_1D'), 2))
    peel._clean_peeled_remainder(cand)
    assert peel._mappable_loop_count(cand) == 2


def test_no_crossover_means_the_pass_refuses_and_leaves_the_loop_alone():
    """NEGATIVE case. Equal slopes give no crossover, so a plain carried recurrence nominates
    nothing and the pass must not touch it -- a pass that does not apply must not mutate."""
    sdfg = carried_recurrence.to_sdfg(simplify=True)
    loop = _loops(sdfg)[0]
    peel = BestEffortLoopPeeling(peel_limit=4)
    assert peel.direction_flip_split_points(loop) == []
    assert peel._best_split_for(loop, sdfg) is None
    before = [(l.label, l.init_statement.as_string, l.loop_condition.as_string) for l in _loops(sdfg)]
    assert peel.apply_pass(sdfg, {}) is None
    assert [(l.label, l.init_statement.as_string, l.loop_condition.as_string) for l in _loops(sdfg)] == before


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 64, 65, 128, 129])
def test_crossover_split_is_bit_exact_at_both_parities(n):
    """The split is a pure REGROUPING of the same iterations in the same order, so it must
    reproduce the unsplit SDFG BIT-EXACTLY -- not merely closely. An odd ``n`` puts the loop's
    fixpoint iteration (where the read and the write hit the same element in the SAME iteration)
    inside the carved middle, which is where an off-by-one in the cut would corrupt values without
    crashing. Both arrays are checked: s281 writes ``a`` AND ``b``."""
    ref_sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    split_sdfg = reverse_read_same_write.to_sdfg(simplify=True)
    assert BestEffortLoopPeeling(peel_limit=4).apply_pass(split_sdfg, {}) == 1
    split_sdfg.validate()
    assert len(_loops(split_sdfg)) == 3, 'expected [start, x-1] + {x} + [x+1, end]'

    rng = np.random.default_rng(n)
    a0, b0, c0 = rng.random(max(n, 1))[:n], rng.random(max(n, 1))[:n], rng.random(max(n, 1))[:n]
    ra, rb = a0.copy(), b0.copy()
    ref_sdfg.compile()(a=ra, b=rb, c=c0.copy(), LEN_1D=n)
    ga, gb = a0.copy(), b0.copy()
    split_sdfg.compile()(a=ga, b=gb, c=c0.copy(), LEN_1D=n)
    assert np.array_equal(ga, ra), f'a differs at n={n}'
    assert np.array_equal(gb, rb), f'b differs at n={n}'


def test_split_point_past_loop_end_does_not_run_past_the_end():
    """``K >= N`` is past the loop's end, so the guard never fires. Neither the carved-out ``{K}``
    iteration nor the ``[start, K-1]`` segment may run beyond the original last iteration."""
    sdfg = guarded_special_case.to_sdfg(simplify=True)

    n, k = 16, 16
    rng = np.random.default_rng(2)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = _reference(a0.copy(), b.copy(), n, k)

    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), N=n, K=k)
    assert np.array_equal(got, ref)


def test_split_point_provably_past_the_end_is_refused():
    """``_split_loop_at``'s CONTRACT says the segments regroup the same iterations only while
    ``start <= x <= end``; a PROVABLY out-of-range point must be refused, not emitted -- the middle
    singleton would fabricate an iteration the loop never runs (cloudsc: a phantom ``{end + 1}``
    segment whose body reads past the array once TrivialLoopElimination splices it)."""
    sdfg = dace.SDFG('split_past_end')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('a', [dace.symbol('K')], dace.float64)
    loop = LoopRegion('l', 'i < K', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body')
    tasklet = body.add_tasklet('w', set(), {'o'}, 'o = 1.0')
    body.add_edge(tasklet, 'o', body.add_write('a'), None, dace.Memlet('a[i]'))
    sdfg.validate()

    x = symbolic.pystr_to_symbolic('K')  # end is K - 1: provably past the last iteration
    assert not BestEffortLoopPeeling()._split_loop_at(sdfg, loop, x)
    assert not BestEffortLoopPeeling()._split_loop_at(sdfg, loop, x, middle_singleton=False)
    assert not BestEffortLoopPeeling()._split_loop_at(sdfg, loop, symbolic.pystr_to_symbolic('-1'))
    sdfg.validate()
    assert _loops(sdfg) == [loop], 'a refused split must leave the loop alone'
