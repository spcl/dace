# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the single-shot ``parallelize`` pipeline (reduction-aware
loop-to-map). SDFGs are produced through the DaCe Python frontend."""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes import (ParallelizePipeline, parallelize, BestEffortLoopPeeling, ShortLoopUnroll)

M, N = (dace.symbol(s) for s in ('M', 'N'))


def _num_maps(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _num_loops(sdfg: dace.SDFG) -> int:
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def test_parallelize_independent_loop_maps_value_preserving():
    """An embarrassingly parallel loop is mapped and stays value-preserving."""

    @dace.program
    def scale(A: dace.float64[M, N], B: dace.float64[M, N]):
        for i in range(M):
            for j in range(N):
                B[i, j] = A[i, j] * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    parallelize(sdfg, validate=True)
    assert _num_maps(sdfg) > 0
    assert _num_loops(sdfg) < loops_before

    A = np.random.default_rng(0).random((6, 5))
    B = np.zeros((6, 5))
    sdfg(A=A, B=B, M=6, N=5)
    assert np.allclose(B, A * 2.0)


def test_parallelize_rowsum_reduction_value_preserving():
    """A per-row accumulator reduction parallelizes (the outer row loop becomes a
    map; the accumulator is privatized/WCR'd) and matches the numpy oracle."""

    @dace.program
    def rowsum(A: dace.float64[M, N], out: dace.float64[M]):
        for i in range(M):
            acc = 0.0
            for j in range(N):
                acc += A[i, j]
            out[i] = acc

    sdfg = rowsum.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _num_maps(sdfg) > 0

    A = np.random.default_rng(1).random((7, 4))
    out = np.zeros(7)
    sdfg(A=A, out=out, M=7, N=4)
    assert np.allclose(out, A.sum(axis=1))


def test_parallelize_runs_once_idempotent():
    """The pipeline is single-shot: a second run does not change a fixed SDFG's
    results, and reports the same number of stages."""

    @dace.program
    def add(A: dace.float64[M], B: dace.float64[M], C: dace.float64[M]):
        for i in range(M):
            C[i] = A[i] + B[i]

    sdfg = add.to_sdfg(simplify=True)
    # Compare against the pipeline's own composed-stage list rather than a hardcoded number: the
    # list is what apply_pass actually iterates, so this stays correct across stage additions or
    # removals instead of going stale (LoopToReduce's normalization split into an explicit
    # WCRToAugAssign stage, moving the count from 11 to 12, and this assertion missed it).
    expected_stages = len(ParallelizePipeline()._stages())
    assert ParallelizePipeline().apply_pass(sdfg, {}) == expected_stages
    # Re-running is harmless (nothing left to parallelize) and reports the same count.
    assert ParallelizePipeline().apply_pass(sdfg, {}) == expected_stages
    sdfg.validate()

    # Structural: the loop actually became a map, not just "some stages ran".
    assert _num_loops(sdfg) == 0
    assert _num_maps(sdfg) > 0

    A = np.random.default_rng(2).random(8)
    B = np.random.default_rng(3).random(8)
    C = np.zeros(8)
    sdfg(A=A, B=B, C=C, M=8)
    assert np.allclose(C, A + B)


def test_parallelize_unrolls_short_constant_loop():
    """A short constant-trip loop is fully unrolled (no loop, no atomics) below the
    limit, and left alone above it; the result is value-preserving either way."""

    @dace.program
    def short_reduce(A: dace.float64[5], B: dace.float64[1]):
        acc = 0.0
        for j in range(5):
            acc += A[j]
        B[0] = acc

    # The unroll gate fires only at/below the limit (probe the stage directly,
    # since downstream LoopToReduce would also eliminate the loop).
    below = short_reduce.to_sdfg(simplify=True)
    assert _num_loops(below) == 1
    ShortLoopUnroll(unroll_limit=8).apply_pass(below, {})
    assert _num_loops(below) == 0  # trip 5 <= 8 -> unrolled

    above = short_reduce.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=4).apply_pass(above, {})
    assert _num_loops(above) == 1  # trip 5 > 4 -> left alone

    off = short_reduce.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=0).apply_pass(off, {})
    assert _num_loops(off) == 1  # disabled

    # Full pipeline (default limit 8) is value-preserving end to end.
    sdfg = short_reduce.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)
    assert _num_loops(sdfg) == 0
    A = np.random.default_rng(4).random(5)
    B = np.zeros(1)
    sdfg(A=A, B=B)
    assert np.allclose(B[0], A.sum())


def test_short_loop_unroll_refuses_unfusable_branchy_body() -> None:
    """The unroll is declined exactly when it can neither specialize an index nor decide a guard.

    Unrolling substitutes the loop variable, so a body that never names it clones verbatim; if that
    body is also branchy, the clones are separated by conditionals and the pass's local state fusion
    re-merges none of them (npbench ``crc16``: 11 states -> 46, no map gained). Both halves of the
    conjunction are load-bearing, so both are probed from the other side as well.
    """

    @dace.program
    def bit_mix(seed: dace.int64[1], out: dace.int64[1]):
        c = seed[0]
        for _ in range(4):
            if c & 1:
                c = (c >> 1) ^ 7
            else:
                c = c >> 1
        out[0] = c

    @dace.program
    def branchy_indexed(A: dace.float64[4], B: dace.float64[1]):
        for m in range(4):
            if A[m] > 0.0:
                B[0] = B[0] + A[m]
            else:
                B[0] = B[0] - A[m]

    @dace.program
    def straight_line(A: dace.float64[1], B: dace.float64[1]):
        for _ in range(4):
            B[0] = A[0] * 0.5

    # Branchy AND the iterate is unused -> refused, and the refusal leaves the SDFG untouched.
    refused = bit_mix.to_sdfg(simplify=True)
    states_before = sum(1 for sd in refused.all_sdfgs_recursive() for _ in sd.all_states())
    ShortLoopUnroll(unroll_limit=8).apply_pass(refused, {})
    assert _num_loops(refused) == 1
    assert sum(1 for sd in refused.all_sdfgs_recursive() for _ in sd.all_states()) == states_before

    # Branchy but the iterate INDEXES the body: unrolling pins it, which is what folds the guards
    # and the constant subsets downstream (the CloudSC species loop). Still unrolled.
    indexed = branchy_indexed.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=8).apply_pass(indexed, {})
    assert _num_loops(indexed) == 0

    # Unused iterate but straight-line: the clones fuse back down, which is the pass's whole point.
    flat = straight_line.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=8).apply_pass(flat, {})
    assert _num_loops(flat) == 0

    # Value-preserving: the refused loop still computes the reference recurrence.
    seed = np.array([12345], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    refused(seed=seed, out=out)
    expected = 12345
    for _ in range(4):
        expected = ((expected >> 1) ^ 7) if expected & 1 else expected >> 1
    assert out[0] == expected


def test_parallelize_peel_mechanism_value_preserving():
    """``BestEffortLoopPeeling._peel_loops`` peels boundary iterations off a loop
    (front / back / both) and stays value-preserving, even for symbolic bounds."""

    @dace.program
    def scale(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            B[i] = A[i] * 3.0

    for direction in ('front', 'back', 'both'):
        sdfg = scale.to_sdfg(simplify=True)
        peeled = BestEffortLoopPeeling()._peel_loops(sdfg, count=2, direction=direction)
        assert peeled == 1, direction
        sdfg.validate()
        A = np.arange(10, dtype=np.float64)
        B = np.zeros(10)
        sdfg(A=A, B=B, N=10)
        assert np.allclose(B, A * 3.0), direction


def test_parallelize_peeling_reverts_on_recurrence():
    """A genuine loop-carried recurrence cannot be peeled into a map, so the
    best-effort search reverts (no spurious map) and the result is correct."""

    @dace.program
    def prefix_sum(A: dace.float64[N], B: dace.float64[N]):
        B[0] = A[0]
        for i in range(1, N):
            B[i] = B[i - 1] + A[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    parallelize(sdfg, validate=True)  # peeling cannot help -> reverts
    assert _num_loops(sdfg) >= 1  # recurrence stays a sequential loop

    A = np.arange(1, 9, dtype=np.float64)
    B = np.zeros(8)
    sdfg(A=A, B=B, N=8)
    assert np.allclose(B, np.cumsum(A))


def test_parallelize_peel_limit_zero_disables():
    """``peel_limit=0`` skips the peeling search entirely."""

    @dace.program
    def prefix_sum(A: dace.float64[N], B: dace.float64[N]):
        B[0] = A[0]
        for i in range(1, N):
            B[i] = B[i - 1] + A[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    before = _num_loops(sdfg)
    BestEffortLoopPeeling(peel_limit=0).apply_pass(sdfg, {})
    assert _num_loops(sdfg) == before  # untouched


# -----------------------------------------------------------------------------
# Unrolling a 3-level (double-tiled) nest: the shape ``UntileLoops`` hands over
# when it cannot collapse a tile, and the shape that decides whether the tile
# gets re-baked into the body as straight-line code.
# -----------------------------------------------------------------------------


def _num_tasklets(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)])


def test_short_loop_unroll_fully_unrolls_a_three_level_tile_nest():
    """Three constant-trip levels (2 x 2 x 2) unroll to eight straight-line body copies.

    Asserting the copy count is what separates a full unroll from a partial one: a residual loop
    with a smaller trip count also leaves zero *matching* structure behind if only values are
    checked."""

    @dace.program
    def tiled(a: dace.float64[64], b: dace.float64[64]):
        for i in range(0, 8, 4):
            for ii in range(i, i + 4, 2):
                for iii in range(ii, ii + 2):
                    a[iii] = b[iii] * 2.0

    sdfg = tiled.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 3
    per_body = _num_tasklets(sdfg)

    assert ShortLoopUnroll(unroll_limit=8).apply_pass(sdfg, {}) == 3, 'all three levels must unroll'
    sdfg.validate()
    assert _num_loops(sdfg) == 0, 'full unroll must leave no residual loop'
    assert _num_tasklets(sdfg) == 8 * per_body, f'expected 8 body copies, got {_num_tasklets(sdfg) / per_body}'

    rng = np.random.default_rng(9)
    b = rng.standard_normal(64)
    a = np.zeros(64)
    sdfg(a=a, b=b)
    assert np.allclose(a[:8], b[:8] * 2.0)


def test_short_loop_unroll_unrolls_the_constant_levels_under_a_symbolic_outer_tile():
    """A symbolic outer tile keeps its loop; the two constant inner levels unroll to 4 x 4 copies.

    This is the tile re-baking ``UntileLoops`` runs ahead of: with the tile still in place the
    unroll writes 16 copies of the body into the outer loop, and the canonical unit-stride nest is
    no longer recoverable from it."""

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 16):
            for ii in range(i, i + 16, 4):
                for iii in range(ii, ii + 4):
                    a[iii] = b[iii] * 2.0

    sdfg = tiled.to_sdfg(simplify=True)
    per_body = _num_tasklets(sdfg)

    assert ShortLoopUnroll(unroll_limit=8).apply_pass(sdfg, {}) == 2, 'both constant levels must unroll'
    sdfg.validate()
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(loops) == 1 and loops[0].loop_variable == 'i', 'only the symbolic outer tile may survive'
    assert _num_tasklets(sdfg) == 16 * per_body, f'expected 16 body copies, got {_num_tasklets(sdfg) / per_body}'

    rng = np.random.default_rng(10)
    b = rng.standard_normal(64)
    a = np.zeros(64)
    sdfg(a=a, b=b, N=64)
    assert np.allclose(a, b * 2.0)


def test_untile_before_unroll_leaves_the_unroll_nothing_to_re_bake():
    """The pipeline order contract: ``UntileLoops`` first, so the tile never reaches the unroll.

    Same kernel as above. Once untiled, the nest is a single symbolic-trip unit-stride loop --
    ``ShortLoopUnroll`` finds no constant-trip loop, reports "untouched", and the canonical form
    survives to the parallelization stage instead of being 16x straight-lined."""
    from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

    @dace.program
    def tiled(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, 16):
            for ii in range(i, i + 16, 4):
                for iii in range(ii, ii + 4):
                    a[iii] = b[iii] * 2.0

    sdfg = tiled.to_sdfg(simplify=True)
    per_body = _num_tasklets(sdfg)
    assert UntileLoops().apply_pass(sdfg, {}) == 2, 'the 3-level cascade must collapse twice'
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(loops) == 1 and loops[0].loop_variable.startswith('_untile_k_')

    assert ShortLoopUnroll(unroll_limit=8).apply_pass(sdfg, {}) is None, 'nothing left to unroll'
    assert _num_tasklets(sdfg) == per_body, 'the body must not be duplicated'
    assert _num_loops(sdfg) == 1

    rng = np.random.default_rng(11)
    b = rng.standard_normal(64)
    a = np.zeros(64)
    sdfg(a=a, b=b, N=64)
    assert np.allclose(a, b * 2.0)


if __name__ == '__main__':
    test_parallelize_independent_loop_maps_value_preserving()
    test_parallelize_rowsum_reduction_value_preserving()
    test_parallelize_runs_once_idempotent()
    test_parallelize_unrolls_short_constant_loop()
    test_short_loop_unroll_refuses_unfusable_branchy_body()
    test_parallelize_peel_mechanism_value_preserving()
    test_parallelize_peeling_reverts_on_recurrence()
    test_parallelize_peel_limit_zero_disables()
    test_short_loop_unroll_fully_unrolls_a_three_level_tile_nest()
    test_short_loop_unroll_unrolls_the_constant_levels_under_a_symbolic_outer_tile()
    test_untile_before_unroll_leaves_the_unroll_nothing_to_re_bake()
