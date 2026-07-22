# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Closing the guard on the mid-array broadcast index-set split (tsvc ``s1113_d_single``).

``a[i] = a[LEN_1D // 2] + b[i]`` broadcasts one element across the whole loop, so the read
``a[N // 2]`` conflicts with the write ``a[i]`` at exactly one iteration. ``BestEffortLoopPeeling``
index-set-splits the loop at ``x = int_floor(N, 2)`` to carve that iteration out, which needs
:meth:`BestEffortLoopPeeling._split_loop_at`'s range-membership contract.

Historically both membership sides were left unproven -- ``_provably_nonneg`` only decides concrete
numbers and the large-trip-count assumption only reaches points AFFINE in the trip count, which
``N // 2`` is not -- so the split was emitted under a runtime guard
``if (0 <= N//2 and N//2 <= N-1) { map } else { pinned sequential loop }``.

Two things close it, both pinned here:

1. The ``before`` segment ``[start, x-1]`` is emitted as ``i < x``, so it overruns only when it runs
   an iteration ABOVE ``end`` (``x - 1 > end``). Its true safety bound is therefore ``x <= end + 1``,
   not the stricter ``x <= end`` -- at ``x == end + 1`` the ``before`` segment is exactly the whole
   loop and the clamped middle singleton and ``after`` are both empty.
2. :meth:`BestEffortLoopPeeling._provably_nonneg_symbolic` discharges the resulting two sides,
   ``0 <= int_floor(N, 2)`` and ``int_floor(N, 2) <= N``, from the canonicalization
   nonnegative-symbol contract plus the floor/ceil bounds. Both hold for EVERY ``N >= 0``.

The distinction matters exactly at ``N == 0``: ``int_floor(0,2) = 0 <= end = -1`` is FALSE, so the
stricter bound is genuinely not always-true and must not be "proven". The loosened ``x <= end + 1``
IS always-true, and the split is correct at ``N == 0`` because every segment is then empty. The
edge-``N`` value test below pins that, so the closure rests on a bound that actually holds rather
than on the empty-loop case being overlooked.

Companion of ``close_guarded_back_peel_split_test.py`` (the affine-in-trip back-peel split); the
guard-PRESERVING cases live in ``strengthen2_parallelization_prep_test.py``.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling

LEN_1D = dace.symbol('LEN_1D', nonnegative=True)


@dace.program
def broadcast_half(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[LEN_1D // 2] + b[i]


def _reference(a, b, n):
    """The sequential meaning of the kernel (the tsvc numpy oracle, inlined)."""
    for i in range(n):
        a[i] = a[n // 2] + b[i]
    return a


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _guarded_fallback_loops(sdfg) -> int:
    """LoopRegions that are the sequential fallback of an ``if cond: map else: seq`` guard
    (a Map-less branch of a ConditionalBlock whose sibling holds a Map) -- the corpus ``g``."""
    n = 0
    for cfr in sdfg.all_control_flow_regions():
        if not isinstance(cfr, ConditionalBlock):
            continue
        has_map = [any(isinstance(x, nd.MapEntry) for x, _ in br.all_nodes_recursive()) for _, br in cfr.branches]
        if not any(has_map):
            continue
        for hm, (_, br) in zip(has_map, cfr.branches):
            if not hm:
                n += sum(1 for s in br.all_control_flow_regions(recursive=True) if isinstance(s, LoopRegion))
    return n


def test_provably_nonneg_symbolic_discharges_the_floor_membership_sides():
    """Both sides of the ``x = int_floor(N, 2)`` membership hold for every ``N >= 0``."""
    peel = BestEffortLoopPeeling(peel_limit=4)
    n = dace.symbolic.pystr_to_symbolic('LEN_1D')
    x = dace.symbolic.pystr_to_symbolic('int_floor(LEN_1D, 2)')
    # after side: start <= x, i.e. int_floor(N, 2) - 0 >= 0.
    assert peel._provably_nonneg_symbolic(x) is True
    # before side against the TRUE bound: x <= end + 1, i.e. (N-1) + 1 - int_floor(N, 2) >= 0.
    assert peel._provably_nonneg_symbolic((n - 1) + 1 - x) is True


def test_provably_nonneg_symbolic_rejects_the_over_strict_bound_and_false_claims():
    """Soundness: the prover must NOT prove things that are false for some nonnegative ``N``."""
    peel = BestEffortLoopPeeling(peel_limit=4)
    n = dace.symbolic.pystr_to_symbolic('LEN_1D')
    x = dace.symbolic.pystr_to_symbolic('int_floor(LEN_1D, 2)')
    # The STRICTER (historical) bound ``x <= end`` is genuinely false at N == 0 (0 <= -1), so it
    # must stay unproven -- this is what forces the closure onto the correct ``x <= end + 1``.
    assert peel._provably_nonneg_symbolic((n - 1) - x) is False
    # Plainly false / undecidable claims stay unproven.
    assert peel._provably_nonneg_symbolic(-n) is False
    assert peel._provably_nonneg_symbolic(n - 5) is False
    assert peel._provably_nonneg_symbolic(x - (n - 1)) is False
    # A free (unbounded-above) offset must not be provable either.
    k = dace.symbolic.pystr_to_symbolic('KOFF')
    assert peel._provably_nonneg_symbolic(n - k) is False


def test_split_range_relations_are_fully_discharged():
    """The broadcast split emits NO runtime guard: an empty relation set."""
    sdfg = broadcast_half.to_sdfg(simplify=True)
    loops = _loops(sdfg)
    assert len(loops) == 1
    loop = loops[0]
    peel = BestEffortLoopPeeling(peel_limit=4)
    x = peel._best_split_for(loop, sdfg)
    assert x is not None, 'expected an index-set split point for the broadcast-conflict loop'
    # The split point is the broadcast index itself, int_floor(LEN_1D, 2).
    assert dace.symbolic.simplify(x - dace.symbolic.pystr_to_symbolic('int_floor(LEN_1D, 2)')) == 0, \
        f'split point should be int_floor(LEN_1D, 2), got {x}'
    # Compared through the same untagged parse path as ``x`` above, so the assumption-tagged
    # symbol copies cancel (the module-level ``LEN_1D`` carries ``nonnegative=True``).
    end = loop_analysis.get_loop_end(loop)
    assert dace.symbolic.simplify(end - dace.symbolic.pystr_to_symbolic('LEN_1D - 1')) == 0, \
        f'loop end should be LEN_1D - 1, got {end}'
    assert peel._split_range_relations(loop, x) == frozenset(), 'both membership sides must be provable'


def test_broadcast_split_drops_the_pinned_sequential_fallback():
    """The loop is UNCONDITIONALLY parallelized: g goes 1 -> 0 and no pinned fallback survives."""
    sdfg = broadcast_half.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 1
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()
    assert _guarded_fallback_loops(sdfg) == 0, 'the pinned sequential fallback must be dropped'
    assert not any(r.pinned_sequential for r in _loops(sdfg)), 'no pinned-sequential loop may survive'


@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 128])
def test_broadcast_split_is_bit_exact_including_the_edge_trip_counts(n):
    """The unconditionally-split form reproduces the sequential result BIT-EXACTLY.

    ``n`` in ``{0, 1}`` are the cases the dropped guard used to divert to the sequential fallback
    (``int_floor(n,2) <= n-1`` is false at 0 and only just holds at 1): the split must compute them
    correctly with no guard at all. There is no reduction here, so equality is exact, not just
    close -- any reassociation would be a real miscompile.
    """
    sdfg = broadcast_half.to_sdfg(simplify=True)
    BestEffortLoopPeeling(peel_limit=4).apply_pass(sdfg, {})
    sdfg.validate()

    rng = np.random.default_rng(1234)
    a0 = rng.random(max(n, 1))[:n]
    b = rng.random(max(n, 1))[:n]
    ref = _reference(a0.copy(), b.copy(), n)

    got = a0.copy()
    sdfg.compile()(a=got, b=b.copy(), LEN_1D=n)
    assert np.array_equal(got, ref), f'split form must match the sequential meaning bit-exactly at n={n}'


if __name__ == '__main__':
    test_provably_nonneg_symbolic_discharges_the_floor_membership_sides()
    test_provably_nonneg_symbolic_rejects_the_over_strict_bound_and_false_claims()
    test_split_range_relations_are_fully_discharged()
    test_broadcast_split_drops_the_pinned_sequential_fallback()
    for _n in [0, 1, 2, 3, 5, 128]:
        test_broadcast_split_is_bit_exact_including_the_edge_trip_counts(_n)
    print('OK')
