# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll`.
SDFGs are produced through the DaCe Python frontend."""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll


def _num_loops(sdfg: dace.SDFG) -> int:
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def test_short_loop_unroll_gate_on_trip_count():
    """A short constant-trip loop is fully unrolled at/below the limit and left alone above it or when
    disabled; the unrolled SDFG stays value-preserving."""

    @dace.program
    def short_reduce(A: dace.float64[5], B: dace.float64[1]):
        acc = 0.0
        for j in range(5):
            acc += A[j]
        B[0] = acc

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

    A = np.random.default_rng(4).random(5)
    B = np.zeros(1)
    below(A=A, B=B)
    assert np.allclose(B[0], A.sum())


def test_short_loop_unroll_branchy_and_straightline_value_preserving() -> None:
    """Constant-trip branchy / straight-line short loops unroll and stay value-preserving.

    ShortLoopUnroll only unrolls what ``LoopUnroll.can_be_applied_to`` accepts; whether a branchy
    recurrence (loop variable unused in the body) is accepted depends on the SDFG shape the frontend
    produces (on the extended branch that shape declined; here it is accepted). Either outcome is
    correct, so this asserts the invariant that holds regardless -- the result computes the reference
    -- plus the unconditional unroll of the two shapes ``LoopUnroll`` always takes.
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
    def straight_line(x: dace.float64[1]):
        for _ in range(4):
            x[0] = x[0] * 0.5

    # Branchy but the iterate INDEXES the body: unrolling pins it, which folds the guards downstream
    # (the CloudSC species loop). Unrolled.
    indexed = branchy_indexed.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=8).apply_pass(indexed, {})
    assert _num_loops(indexed) == 0

    # Straight-line: the clones fuse back down, which is the pass's whole point.
    flat = straight_line.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=8).apply_pass(flat, {})
    assert _num_loops(flat) == 0

    # Branchy recurrence with an unused iterate: unroll it (allowed here) and check it still computes
    # the reference -- correctness holds whether or not the shape is unrolled.
    recur = bit_mix.to_sdfg(simplify=True)
    ShortLoopUnroll(unroll_limit=8).apply_pass(recur, {})
    seed = np.array([12345], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    recur(seed=seed, out=out)
    expected = 12345
    for _ in range(4):
        expected = ((expected >> 1) ^ 7) if expected & 1 else expected >> 1
    assert out[0] == expected


if __name__ == "__main__":
    test_short_loop_unroll_gate_on_trip_count()
    test_short_loop_unroll_branchy_and_straightline_value_preserving()
