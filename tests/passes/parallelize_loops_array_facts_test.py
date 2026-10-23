# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ParallelizeLoops`` maps a loop that stores through an index array when the array's producer proves the stores
disjoint: a counting-sort permutation (``INJECTIVE``) and prefix-sum row pointers (``NONDECREASING`` over
nonnegative counts). Each negative breaks one premise of a proof and keeps the loop sequential."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import array_facts
from dace.transformation.passes.parallelize_loops import ParallelizeLoops

M = dace.symbol("M")
NNZ = dace.symbol("NNZ")
BINS = 4


def loop_variables(sdfg: dace.SDFG) -> set[str]:
    return {
        r.loop_variable
        for sd in sdfg.all_sdfgs_recursive()
        for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion)
    }


def fact_notes(sdfg: dace.SDFG) -> list[str]:
    return [
        n.specialization_hint
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.specialization_hint and "array facts" in n.specialization_hint
    ]


def parallelized(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    return sdfg


@dace.program
def gather_by_bin(key: dace.int64[M], val: dace.float64[M], out: dace.float64[M]):
    perm = np.zeros((M,), dtype=np.int64)
    count = np.zeros((BINS,), dtype=np.int64)
    offset = np.zeros((BINS,), dtype=np.int64)
    for i in range(M):
        b = key[i]
        if b >= 0:
            count[b] = count[b] + 1
    running = 0
    for b in range(BINS):
        offset[b] = running
        running = running + count[b]
        count[b] = 0
    for r in range(M):
        perm[r] = -1
    for i in range(M):
        b = key[i]
        if b >= 0:
            perm[offset[b] + count[b]] = i
            count[b] = count[b] + 1
    for r in range(M):
        row = perm[r]
        if row >= 0:
            out[row] = val[row] * 2.0


def test_a_store_through_a_counting_sort_permutation_maps():
    """``perm`` is filled with -1, then each iteration of the scatter stores its own index once, so its entries
    other than -1 are distinct and ``out[perm[r]]`` under ``perm[r] >= 0`` never collides."""
    sdfg = parallelized(gather_by_bin)
    assert "r" not in loop_variables(sdfg)
    assert any("perm injective apart from -1 with values in [0, M - 1]" in n for n in fact_notes(sdfg))
    rng = np.random.default_rng(0)
    key = rng.integers(-1, BINS, 50)
    val = rng.random(50)
    out = np.zeros(50)
    sdfg(key=key, val=val, out=out, M=50)
    assert np.array_equal(out, np.where(key >= 0, val * 2.0, 0.0))


@dace.program
def segments_by_bin(key: dace.int64[M], ptr: dace.int64[M + 1], seg: dace.int64[NNZ]):
    perm = np.zeros((M,), dtype=np.int64)
    width = np.zeros((M,), dtype=np.int64)
    for i in range(M):
        perm[i] = -1
    for i in range(M):
        width[i] = 0
        if key[i] >= 0:
            width[i] = width[i] + 2
    running = 0
    for i in range(M):
        ptr[i] = running
        running = running + width[i]
    ptr[M] = running
    for i in range(M):
        if key[i] >= 0:
            perm[M - 1 - i] = i
    for r in range(M):
        row = perm[r]
        if row >= 0:
            base = ptr[row]
            for t in range(ptr[row + 1] - base):
                seg[base + t] = row


def test_a_segment_store_through_running_sum_row_pointers_maps():
    """``ptr`` is a running sum of nonnegative widths, so the segments ``[ptr[row], ptr[row + 1])`` of distinct rows
    are disjoint, and ``perm`` gives each row to one iteration."""
    sdfg = parallelized(segments_by_bin)
    assert "r" not in loop_variables(sdfg)
    notes = "\n".join(fact_notes(sdfg))
    assert "ptr nondecreasing on [0, M]" in notes and "perm injective" in notes
    rng = np.random.default_rng(1)
    key = rng.integers(-1, 2, 40)
    nnz = int(2 * np.count_nonzero(key >= 0))
    ptr, seg = np.zeros(41, dtype=np.int64), np.full(nnz, -7, dtype=np.int64)
    sdfg(key=key, ptr=ptr, seg=seg, M=40, NNZ=nnz)
    ref_ptr = np.concatenate([[0], np.cumsum(np.where(key >= 0, 2, 0))])
    ref_seg = np.repeat(np.flatnonzero(key >= 0), 2)
    assert np.array_equal(ptr, ref_ptr) and np.array_equal(seg, ref_seg)


@dace.program
def two_scatters_after_one_fill(key: dace.int64[M], out: dace.float64[M]):
    perm = np.zeros((M,), dtype=np.int64)
    for r in range(M):
        perm[r] = -1
    for i in range(M):
        if key[i] >= 0:
            perm[key[i]] = i
    for i in range(M):
        if key[i] < 0:
            perm[M - 1 - i] = i
    for r in range(M):
        row = perm[r]
        if row >= 0:
            out[row] = 1.0


@dace.program
def unguarded_sentinel(key: dace.int64[M], out: dace.float64[M]):
    perm = np.zeros((M,), dtype=np.int64)
    for r in range(M):
        perm[r] = -1
    for i in range(M):
        if key[i] >= 0:
            perm[key[i]] = i
    for r in range(M):
        row = perm[r]
        out[row + 1] = 1.0


@dace.program
def index_array_rewritten_in_the_loop(key: dace.int64[M], out: dace.float64[M]):
    perm = np.zeros((M,), dtype=np.int64)
    for r in range(M):
        perm[r] = -1
    for i in range(M):
        if key[i] >= 0:
            perm[key[i]] = i
    for r in range(M):
        row = perm[r]
        if row >= 0:
            out[row] = 1.0
            perm[row] = 0


@dace.program
def caller_supplied_index_array(perm: dace.int64[M], out: dace.float64[M]):
    for r in range(M):
        row = perm[r]
        if row >= 0:
            out[row] = 1.0


@dace.program
def widths_of_unknown_sign(width: dace.int64[M], ptr: dace.int64[M + 1], seg: dace.int64[NNZ]):
    running = 0
    for i in range(M):
        ptr[i] = running
        running = running + width[i]
    ptr[M] = running
    for r in range(M):
        base = ptr[r]
        for t in range(ptr[r + 1] - base):
            seg[base + t] = r


@pytest.mark.parametrize(
    "program",
    [
        # Two scatters each store every index once, so together they can store one index twice.
        two_scatters_after_one_fill,
        # Without ``row >= 0`` two iterations can both read the sentinel and store into ``out[0]``.
        unguarded_sentinel,
        # The loop writes the array its stores are indexed through.
        index_array_rewritten_in_the_loop,
        # No producer: nothing is known about a caller's array.
        caller_supplied_index_array,
    ],
)
def test_a_store_through_an_unproven_permutation_stays_sequential(program):
    sdfg = parallelized(program)
    assert "r" in loop_variables(sdfg)
    assert not fact_notes(sdfg)


def test_segments_of_a_running_sum_over_caller_widths_stay_sequential():
    """A caller's width can be negative, so the running sum need not be nondecreasing and segments can overlap."""
    sdfg = parallelized(widths_of_unknown_sign)
    assert "r" in loop_variables(sdfg)


def test_a_difference_is_not_nonnegative():
    """A transient stored as ``a - b`` of two nonnegative transients is not nonnegative; a sum of them is."""
    sdfg = dace.SDFG("nonneg")
    for name in ("a", "b", "s", "d"):
        sdfg.add_scalar(name, dace.int64, transient=True)
    state = sdfg.add_state()
    for name in ("a", "b"):
        t = state.add_tasklet(f"set_{name}", {}, {"o"}, "o = 3")
        state.add_edge(t, "o", state.add_write(name), None, dace.Memlet(f"{name}[0]"))
    for name, op in (("s", "+"), ("d", "-")):
        t = state.add_tasklet(f"set_{name}", {"x", "y"}, {"o"}, f"o = x {op} y")
        state.add_edge(state.add_read("a"), None, t, "x", dace.Memlet("a[0]"))
        state.add_edge(state.add_read("b"), None, t, "y", dace.Memlet("b[0]"))
        state.add_edge(t, "o", state.add_write(name), None, dace.Memlet(f"{name}[0]"))
    known = array_facts.nonnegative_containers(sdfg)
    assert {"a", "b", "s"} <= known and "d" not in known


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
