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
from dace.transformation.passes.canonicalize.lift_counter_scan import INCREMENT_PREFIX, MIN_PREFIX, LiftCounterScan
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import MASK_PREFIX
from tests.corpus import corpus_suite as suite
from tests.corpus.gpu_offload_corpus_test import canon_gpu
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

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


def test_a_bump_inside_an_inner_loop_adds_its_trip_count_per_iteration():
    sdfg = canonical(bump_in_inner_loop)
    assert scanned(sdfg)
    k = np.random.default_rng(4).integers(-2, 6, 300)
    start = np.zeros(300, np.int64)
    sdfg(k=k, start=start, N=300)
    assert np.array_equal(start, offsets_in_numpy(np.maximum(k, 0))[0])


def test_a_guard_that_reads_the_counter_is_refused():
    assert lifted_alone(guard_reads_counter) is None


def test_a_step_that_reads_the_counter_is_refused():
    assert lifted_alone(step_reads_derived) is None


def test_a_step_that_reads_what_the_body_writes_is_refused():
    assert lifted_alone(step_reads_body_write) is None


# Counters that index their own segment ``[c, c + s)`` of an array: the reproducers of QE vexx and cp2k.

L = dace.symbol("L", dtype=dace.int64)
NAT = dace.symbol("NAT", dtype=dace.int64)
NH = dace.symbol("NH", dtype=dace.int64)
P = dace.symbol("P", dtype=dace.int64)


@dace.program
def harmonics_segments(ylm: dace.float64[(L + 1) * (L + 1)], scale: dace.float64[L + 1]):
    # QE ylmr2's ``sym51 = sym51 + 1 + 2 * max(0, i)``: degree i owns the 2i + 1 entries from sym51 on
    s = 1
    for i in range(1, L + 1):
        ylm[s] = scale[i] * ylm[s]
        for k in range(1, i + 1):
            ylm[2 * k + s - 1] = ylm[2 * k + s - 1] + scale[i]
        for k in range(1, i + 1):
            ylm[2 * k + s] = ylm[2 * k + s] - scale[i]
        s = s + 1 + 2 * max(0, i)


def harmonics_in_numpy(ylm: np.ndarray, scale: np.ndarray, degree: int) -> np.ndarray:
    ylm = ylm.copy()
    s = 1
    for i in range(1, degree + 1):
        ylm[s] *= scale[i]
        for k in range(1, i + 1):
            ylm[2 * k + s - 1] = ylm[2 * k + s - 1] + scale[i]
        for k in range(1, i + 1):
            ylm[2 * k + s] = ylm[2 * k + s] - scale[i]
        s = s + 1 + 2 * max(0, i)
    return ylm


@dace.program
def triangle_segments(lx: dace.int64[L * L], ly: dace.int64[L * L], j: dace.int64):
    # cp2k grid_integrate's ``inl1 = inl1 + max(0, j - k + 1)``: pair k owns the j - k + 1 entries from inl1 on
    n = 0
    for k in range(j + 1):
        for m in range(j - k + 1):
            lx[n + m] = k
            ly[n + m] = m
        n = n + max(0, j - k + 1)


def triangle_in_numpy(size: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    lx, ly = np.zeros(size, np.int64), np.zeros(size, np.int64)
    n = 0
    for k in range(j + 1):
        for m in range(j - k + 1):
            lx[n + m], ly[n + m] = k, m
        n += max(0, j - k + 1)
    return lx, ly


@dace.program
def projector_segments(
    ityp: dace.int64[NAT], nh: dace.int64[1], sk: dace.float64[NAT, P], vkb: dace.float64[NAT * NH, P], j: dace.int64
):
    # QE vexx's ``sym1 = sym1 + nhnt`` over the atoms of type j: atom k owns nhnt rows of vkb from sym1 on
    s = 0
    nhnt = nh[0]
    for k in range(NAT):
        if ityp[k] == j:
            for m in range(nhnt):
                for p in range(P):
                    vkb[s + m, p] = sk[k, p] * (m + 1)
            s = s + nhnt


def projector_in_numpy(ityp: np.ndarray, nhnt: int, sk: np.ndarray, rows: int, j: int) -> np.ndarray:
    vkb = np.zeros((rows, sk.shape[1]))
    s = 0
    for k in range(ityp.size):
        if ityp[k] == j:
            for m in range(nhnt):
                vkb[s + m] = sk[k] * (m + 1)
            s += nhnt
    return vkb


def guarded(sdfg: dace.SDFG) -> bool:
    return any(name.startswith(MIN_PREFIX) for name in sdfg.arrays)


@pytest.mark.parametrize("degree", [0, 1, 6])
def test_a_counter_that_indexes_its_own_segment_becomes_a_scan_and_a_parallel_map(degree):
    sdfg = canonical(harmonics_segments)
    assert scanned(sdfg) and loops(sdfg) == 0 and not guarded(sdfg)  # the step 1 + 2 * max(0, i) is never negative
    rng = np.random.default_rng(degree)
    ylm, scale = rng.random((degree + 1) ** 2), rng.random(degree + 1)
    want = harmonics_in_numpy(ylm, scale, degree)
    sdfg(ylm=ylm, scale=scale, L=degree)
    assert np.array_equal(ylm, want)


@pytest.mark.parametrize("j", [0, 3, 9])
def test_a_triangular_segment_counter_fills_each_pair_once(j):
    sdfg = canonical(triangle_segments)
    assert scanned(sdfg) and not guarded(sdfg)
    side = j + 2
    lx, ly = np.full(side * side, -1, np.int64), np.full(side * side, -1, np.int64)
    sdfg(lx=lx, ly=ly, j=j, L=side)
    want_x, want_y = triangle_in_numpy(side * side, j)
    filled = (j + 1) * (j + 2) // 2
    assert np.array_equal(lx[:filled], want_x[:filled]) and np.array_equal(ly[:filled], want_y[:filled])
    assert (lx[filled:] == -1).all() and (ly[filled:] == -1).all()


@pytest.mark.parametrize("nhnt", [3, 0, -2])
def test_a_data_sized_segment_runs_in_parallel_only_when_no_step_is_negative(nhnt):
    """``nhnt`` is data: phase 3 runs in parallel under ``min(inc) >= 0``, in order otherwise."""
    sdfg = canonical(projector_segments)
    assert scanned(sdfg) and guarded(sdfg)
    rng = np.random.default_rng(5)
    nat, p, j = 40, 7, 2
    ityp, sk = rng.integers(0, 4, nat), rng.random((nat, p))
    vkb = np.zeros((nat * 3, p))
    sdfg(ityp=ityp, nh=np.array([nhnt]), sk=sk, vkb=vkb, j=j, NAT=nat, NH=3, P=p)
    assert np.array_equal(vkb, projector_in_numpy(ityp, nhnt, sk, nat * 3, j))


NSP = dace.symbol("NSP", dtype=dace.int64)


@dace.program
def pair_segments(nh: dace.int64[NSP], tvanp: dace.int64[NSP], qgm: dace.float64[NSP * NH * NH, P]):
    # QE vexx's ``ijh = ijh + 1`` inside the k and l loops: pair (k, l), l >= k, of a species with tvanp gets a column
    ijh = 0
    for j in range(NSP):
        if tvanp[j] != 0:
            for k in range(nh[j]):
                for m in range(k, nh[j]):
                    ijh = ijh + 1
                    for p in range(P):
                        qgm[ijh - 1, p] = j * 100.0 + k * 10.0 + m + p * 0.5


def pairs_in_numpy(nh: np.ndarray, tvanp: np.ndarray, rows: int, columns: int) -> np.ndarray:
    qgm, ijh = np.zeros((rows, columns)), 0
    for j in range(nh.size):
        if tvanp[j] != 0:
            for k in range(nh[j]):
                for m in range(k, nh[j]):
                    qgm[ijh] = j * 100.0 + k * 10.0 + m + np.arange(columns) * 0.5
                    ijh += 1
    return qgm


def test_a_counter_bumped_in_inner_loops_scans_the_sum_of_each_iterations_bumps():
    sdfg = canonical(pair_segments)
    assert scanned(sdfg) and not guarded(sdfg)
    rng = np.random.default_rng(6)
    nsp, nh_max, p = 9, 4, 5
    nh, tvanp = rng.integers(0, nh_max + 1, nsp), rng.integers(0, 2, nsp)
    qgm = np.zeros((nsp * nh_max * nh_max, p))
    sdfg(nh=nh, tvanp=tvanp, qgm=qgm, NSP=nsp, NH=nh_max, P=p)
    assert np.array_equal(qgm, pairs_in_numpy(nh, tvanp, qgm.shape[0], p))


@dace.program
def rectangular_nest_counter(a: dace.float64[N, M], out: dace.float64[N * M]):
    n = 0
    for i in range(N):
        for m in range(M):
            out[n] = a[i, m]
            n = n + 1


def test_a_constant_bump_in_a_rectangular_nest_is_left_to_the_induction_variables():
    assert lifted_alone(rectangular_nest_counter) is None


@dace.program
def segment_overlaps(k: dace.int64[N], out: dace.float64[2 * N]):
    n = 0
    for i in range(N):
        for m in range(k[i] + 1):
            out[n + m] = 1.0
        n = n + k[i]


def test_a_write_past_its_segment_is_refused():
    """``out[n + k]`` is the next iteration's first element."""
    assert lifted_alone(segment_overlaps) is None


# Reproducer: s123 stayed sequential, CARRIED_SYMBOL j: two bump sites of one counter (LLR tsvc_2_s123)
@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_a_counter_bumped_at_two_sites_scans_the_sum_of_each_iterations_bumps(target: str):
    kernel = tsvc.collect(name="s123_d_single")[0]
    sdfg = tsvc.to_sdfg(kernel, f"two_bumps_{target}", simplify=True)
    canonicalize(sdfg, validate=True, target=target)
    assert scanned(sdfg) and loops(sdfg) == 0, f"{loops(sdfg)} loops left"
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    arrays["c"] -= 0.5  # both branches of the guard
    want = {name: value.copy() for name, value in arrays.items()}
    REFERENCES[kernel.name](**want, **call_kwargs)
    sdfg(**arrays, **call_kwargs)
    for name, value in arrays.items():
        assert np.allclose(value, want[name]), name


@pytest.mark.gpu
def test_a_counter_bumped_at_two_sites_runs_on_the_device():
    ctx = suite.make("tsvc", "s123_d_single")
    sdfg = suite.build(ctx, canon_gpu, "two_bumps_device")
    assert scanned(sdfg) and loops(sdfg) == 0, f"{loops(sdfg)} loops left"
    assert suite.run_matches(ctx, sdfg)


@dace.program
def two_bumps(a: dace.float64[2 * N], b: dace.float64[N], c: dace.float64[N]):
    j = -1
    for i in range(N):
        j = j + 1
        a[j] = b[i]
        if c[i] > 0.0:
            j = j + 1
            a[j] = c[i]


@dace.program
def two_bumps_reaching_ahead(a: dace.float64[2 * N + 1], b: dace.float64[N], c: dace.float64[N]):
    j = -1
    for i in range(N):
        j = j + 1
        a[j + 1] = b[i]
        if c[i] > 0.0:
            j = j + 1
            a[j] = c[i]


def test_two_bump_sites_lift_alone():
    assert lifted_alone(two_bumps) == 1


def test_two_bump_sites_writing_past_the_bumped_value_are_refused():
    """``a[j + 1]`` after the first bump is the next iteration's first element when the guard fails."""
    assert lifted_alone(two_bumps_reaching_ahead) is None


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
    test_a_bump_inside_an_inner_loop_adds_its_trip_count_per_iteration()
    test_a_guard_that_reads_the_counter_is_refused()
    test_a_step_that_reads_the_counter_is_refused()
    test_a_step_that_reads_what_the_body_writes_is_refused()
    for degree in (0, 1, 6):
        test_a_counter_that_indexes_its_own_segment_becomes_a_scan_and_a_parallel_map(degree)
    for j in (0, 3, 9):
        test_a_triangular_segment_counter_fills_each_pair_once(j)
    for nhnt in (3, 0, -2):
        test_a_data_sized_segment_runs_in_parallel_only_when_no_step_is_negative(nhnt)
    test_a_write_past_its_segment_is_refused()
    test_a_counter_bumped_in_inner_loops_scans_the_sum_of_each_iterations_bumps()
    test_a_constant_bump_in_a_rectangular_nest_is_left_to_the_induction_variables()
    for target in ("cpu", "gpu"):
        test_a_counter_bumped_at_two_sites_scans_the_sum_of_each_iterations_bumps(target)
    test_a_counter_bumped_at_two_sites_runs_on_the_device()
    test_two_bump_sites_lift_alone()
    test_two_bump_sites_writing_past_the_bumped_value_are_refused()
