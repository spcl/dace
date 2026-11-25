# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LiftCounterScan: a loop-carried counter ``c = c + s(i)`` becomes an exclusive scan of its increments.

The lifts run through the whole canonicalize pipeline and are checked against NumPy; each refusal applies the pass alone
to the frontend's graph, where it runs in the pipeline, and checks that nothing changed."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.lift_counter_scan import INCREMENT_PREFIX, LiftCounterScan
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import MASK_PREFIX

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


@dace.program
def offsets(k: dace.int64[N], start: dace.int64[N], cnt: dace.int64[1]):
    n = 0
    for i in range(N):
        start[i] = n
        n = n + k[i]
    cnt[0] = n


@dace.program
def guarded_offsets(k: dace.int64[N], m: dace.float64[N], start: dace.int64[N], cnt: dace.int64[1]):
    # QE vexx's ``nij_type(j) = nij; if (tvanp(j)) nij = nij + nh(j) * (nh(j) + 1) / 2``
    n = 5
    for i in range(N):
        start[i] = n
        if m[i] > 0.0:
            n = n + k[i] * (k[i] + 1) // 2
    cnt[0] = n


@dace.program
def inclusive_offsets(k: dace.int64[N], end: dace.int64[N], cnt: dace.int64[1]):
    n = 0
    for i in range(N):
        if k[i] > 1:
            n = n + k[i]
        end[i] = n * 2
    cnt[0] = n


@dace.program
def counter_gathers(k: dace.int64[N], table: dace.float64[M], out: dace.float64[N]):
    n = 0
    for i in range(N):
        out[i] = table[n]
        n = n + k[i]


def offsets_in_numpy(increments: np.ndarray, base: int = 0) -> tuple[np.ndarray, int]:
    """The counter at the start of each iteration, and after the loop."""
    running = base + np.cumsum(increments)
    return np.concatenate([[base], running[:-1]]) if increments.size else running, int(base + increments.sum())


def scanned(sdfg: dace.SDFG) -> bool:
    return any(name.startswith(INCREMENT_PREFIX) for name in sdfg.arrays)


def scans(sdfg: dace.SDFG) -> int:
    return sum(1 for node, parent in sdfg.all_nodes_recursive() if isinstance(node, Scan))


def loops(sdfg: dace.SDFG) -> int:
    return sum(1 for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion))


def maps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, parent in sdfg.all_nodes_recursive() if isinstance(node, nd.MapEntry))


def canonical(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    return sdfg


def lifted_alone(program) -> int | None:
    """The pass alone on the frontend's graph; validates what it leaves."""
    sdfg = program.to_sdfg(simplify=True)
    result = LiftCounterScan().apply_pass(sdfg, {})
    sdfg.validate()
    return result


@pytest.mark.parametrize("size", [1, 2, 37, 1000])
def test_a_counter_bumped_every_iteration_becomes_a_scan_and_two_parallel_loops(size):
    sdfg = canonical(offsets)
    assert scanned(sdfg) and scans(sdfg) == 1 and loops(sdfg) == 0 and maps(sdfg) >= 2
    k = np.random.default_rng(size).integers(0, 9, size)
    start, cnt = np.zeros(size, np.int64), np.zeros(1, np.int64)
    sdfg(k=k, start=start, cnt=cnt, N=size)
    want, total = offsets_in_numpy(k)
    assert np.array_equal(start, want) and cnt[0] == total


@pytest.mark.parametrize("size", [1, 64, 1000])
def test_a_guarded_bump_read_before_the_guard_adds_nothing_where_the_guard_fails(size):
    sdfg = canonical(guarded_offsets)
    assert scanned(sdfg) and scans(sdfg) == 1 and loops(sdfg) == 0
    rng = np.random.default_rng(size)
    k, m = rng.integers(0, 9, size), rng.uniform(-1.0, 1.0, size)
    start, cnt = np.zeros(size, np.int64), np.zeros(1, np.int64)
    sdfg(k=k, m=m, start=start, cnt=cnt, N=size)
    want, total = offsets_in_numpy(np.where(m > 0.0, k * (k + 1) // 2, 0), base=5)
    assert np.array_equal(start, want) and cnt[0] == total


def test_a_read_after_the_bump_sees_the_bumped_value():
    sdfg = canonical(inclusive_offsets)
    assert scanned(sdfg) and loops(sdfg) == 0
    k = np.random.default_rng(3).integers(0, 5, 500)
    end, cnt = np.zeros(500, np.int64), np.zeros(1, np.int64)
    sdfg(k=k, end=end, cnt=cnt, N=500)
    running = np.cumsum(np.where(k > 1, k, 0))
    assert np.array_equal(end, 2 * running) and cnt[0] == running[-1]


def test_a_zero_trip_loop_leaves_the_counter_at_its_value_before_the_loop():
    sdfg = canonical(guarded_offsets)
    cnt = np.zeros(1, np.int64)
    sdfg(k=np.zeros(0, np.int64), m=np.zeros(0), start=np.zeros(0, np.int64), cnt=cnt, N=0)
    assert cnt[0] == 5


def test_a_counter_that_gathers_reads_the_table_at_each_offset():
    sdfg = canonical(counter_gathers)
    assert scanned(sdfg) and loops(sdfg) == 0
    rng = np.random.default_rng(4)
    k = rng.integers(0, 3, 300)
    table = rng.random(int(k.sum()) + 1)
    out = np.zeros(300)
    sdfg(k=k, table=table, out=out, N=300, M=table.size)
    assert np.array_equal(out, table[offsets_in_numpy(k)[0]])


# Refusals: firing on any of these computes the wrong counter, or takes another lift's loop.


@dace.program
def compaction(a: dace.float64[N], out: dace.float64[N], cnt: dace.int64[1]):
    n = 0
    for i in range(N):
        if a[i] > 0.0:
            out[n] = a[i]
            n = n + 1
    cnt[0] = n


@dace.program
def count_only(a: dace.float64[N], cnt: dace.int64[1]):
    n = 0
    for i in range(N):
        if a[i] > 0.0:
            n = n + 1
    cnt[0] = n


@dace.program
def bump_in_inner_loop(k: dace.int64[N], start: dace.int64[N]):
    n = 0
    for i in range(N):
        start[i] = n
        for j in range(k[i]):
            n = n + 1


@dace.program
def guard_reads_counter(k: dace.int64[N], start: dace.int64[N]):
    n = 0
    for i in range(N):
        start[i] = n
        if n < 100:
            n = n + k[i]


@dace.program
def step_reads_derived(k: dace.int64[N], start: dace.int64[N]):
    n = 1
    for i in range(N):
        t = n * 2
        start[i] = t
        n = n + t


@dace.program
def step_reads_body_write(k: dace.int64[N], start: dace.int64[N]):
    n = 0
    for i in range(N):
        k[i] = k[i] + 1
        start[i] = n
        n = n + k[i]


def test_a_cursor_that_indexes_a_write_is_left_to_the_compaction_lift():
    assert lifted_alone(compaction) is None
    sdfg = canonical(compaction)
    assert not scanned(sdfg) and any(name.startswith(MASK_PREFIX) for name in sdfg.arrays)


def test_a_counter_read_only_after_the_loop_is_left_to_the_reductions():
    assert lifted_alone(count_only) is None


def test_a_bump_inside_an_inner_loop_is_refused():
    assert lifted_alone(bump_in_inner_loop) is None


def test_a_guard_that_reads_the_counter_is_refused():
    assert lifted_alone(guard_reads_counter) is None


def test_a_step_that_reads_the_counter_is_refused():
    assert lifted_alone(step_reads_derived) is None


def test_a_step_that_reads_what_the_body_writes_is_refused():
    assert lifted_alone(step_reads_body_write) is None


if __name__ == "__main__":
    for size in (1, 2, 37, 1000):
        test_a_counter_bumped_every_iteration_becomes_a_scan_and_two_parallel_loops(size)
    for size in (1, 64, 1000):
        test_a_guarded_bump_read_before_the_guard_adds_nothing_where_the_guard_fails(size)
    test_a_read_after_the_bump_sees_the_bumped_value()
    test_a_zero_trip_loop_leaves_the_counter_at_its_value_before_the_loop()
    test_a_counter_that_gathers_reads_the_table_at_each_offset()
    test_a_cursor_that_indexes_a_write_is_left_to_the_compaction_lift()
    test_a_counter_read_only_after_the_loop_is_left_to_the_reductions()
    test_a_bump_inside_an_inner_loop_is_refused()
    test_a_guard_that_reads_the_counter_is_refused()
    test_a_step_that_reads_the_counter_is_refused()
    test_a_step_that_reads_what_the_body_writes_is_refused()
