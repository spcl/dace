# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP cost analysis read directly off an SDFG loop nest: latency and bandwidth per iteration, local
memory free, and a layout change visible in the predicted time. Pure/symbolic -- no measurement."""
import dace
import sympy as sp

from dace.transformation.layout.cost_model.logp_analysis import analyze_loop_nest
from dace.transformation.layout.cost_model.loggp import LogGP, gap_from_bandwidth

N = dace.symbol("N")
P = LogGP(L=95e-9, o=0.0, g=4e-9, G=gap_from_bandwidth(100e9), line_bytes=64, bw_saturated=100e9, bw_core=40e9)


def _build(a_strides, a_storage=dace.dtypes.StorageType.Default):
    sdfg = dace.SDFG("logp")
    sdfg.add_array("A", [N, N], dace.float64, strides=a_strides, storage=a_storage)
    sdfg.add_array("B", [N, N], dace.float64)
    sdfg.add_array("C", [N, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"})
    t = st.add_tasklet("t", {"a", "b"}, {"c"}, "c = a + b")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(st.add_read("B"), me, t, dst_conn="b", memlet=dace.Memlet("B[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=dace.Memlet("C[i,j]"))
    return sdfg, st, me


def _at(expr, n=4096):
    return float(sp.simplify(expr).subs(N, n))


def test_latency_and_bandwidth_per_iteration_are_both_reported():
    """The analysis produces a latency term (L per message) AND a bandwidth term (G per byte) per
    iteration, for every global array."""
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert set(cost.arrays) == {"A", "B", "C"}
    assert _at(cost.latency_per_iter()) > 0
    assert _at(cost.bandwidth_per_iter()) > 0
    # three contiguous arrays: ~1/8 block each per iter, so latency ~ 3 * (1/8) * L
    assert _at(cost.latency_per_iter()) == \
        __import__("pytest").approx(3 * (1.0 / 8.0) * P.L, rel=0.02)


def test_time_per_iter_is_latency_plus_bandwidth():
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert _at(cost.time_per_iter()) == __import__("pytest").approx(
        _at(cost.latency_per_iter()) + _at(cost.bandwidth_per_iter()))


def test_layout_change_moves_the_predicted_cost():
    """The point: a Permute of A (contiguous inner -> strided) must raise the predicted LogP time.
    A is now touched a whole block per iteration instead of 1/8, so both latency and total rise."""
    _, st_row, me_row = _build((N, 1))
    _, st_col, me_col = _build((1, N))
    row = analyze_loop_nest(st_row, me_row, P, block_bytes=64)
    col = analyze_loop_nest(st_col, me_col, P, block_bytes=64)
    assert _at(col.latency_per_iter()) > 2 * _at(row.latency_per_iter())
    assert _at(col.total_time()) > 2 * _at(row.total_time())


def test_local_memory_is_free():
    """Local storage (or an explicit override) contributes no messages: local access is free for
    now, per the framing."""
    # storage-based: A in registers
    _, st, me = _build((N, 1), a_storage=dace.dtypes.StorageType.Register)
    by_storage = analyze_loop_nest(st, me, P, block_bytes=64)
    assert by_storage.arrays["A"].is_local
    assert sp.simplify(by_storage.arrays["A"].messages_per_iter) == 0

    # override-based
    _, st2, me2 = _build((N, 1))
    by_override = analyze_loop_nest(st2, me2, P, block_bytes=64, local_arrays=frozenset({"A"}))
    assert by_override.arrays["A"].is_local
    # A no longer contributes; latency drops from 3 arrays to 2
    full = analyze_loop_nest(st2, me2, P, block_bytes=64)
    assert _at(by_override.latency_per_iter()) < _at(full.latency_per_iter())


def test_serialized_is_a_diagnostic_ceiling_not_the_answer():
    """The serialized sum charges every message its full latency with no overlap. It is retained as
    an upper bound only -- the MLP sweep refuted it as a predictor (rate scales linearly with
    independent chains). total_time must sit far below it whenever the nest overlaps requests."""
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert _at(cost.total_time_serialized()) > 10 * _at(cost.total_time())
    # and the ceiling relation holds at ANY concurrency, including the dependent chain:
    chase = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=1.0)
    assert _at(chase.total_time_serialized()) >= _at(chase.total_time())


def test_regime_from_the_schedule():
    """A parallel map saturates the channels (bandwidth-bound); a Sequential map is serialized to one
    request at a time (latency-bound). This is how the analysis reports which bound a nest hits."""
    import dace as _dace
    sdfg = _dace.SDFG("reg")
    sdfg.add_array("A", [N, N], _dace.float64)
    sdfg.add_array("C", [N, N], _dace.float64)

    def cost_for(schedule):
        st = sdfg.add_state(f"s_{schedule}", is_start_block=(schedule == _dace.ScheduleType.GPU_Device))
        me, mx = st.add_map(f"m_{schedule}", {"i": "0:N", "j": "0:N"}, schedule=schedule)
        t = st.add_tasklet(f"t_{schedule}", {"a"}, {"c"}, "c = a * 2.0")
        st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=_dace.Memlet("A[i,j]"))
        st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=_dace.Memlet("C[i,j]"))
        return analyze_loop_nest(st, me, P, block_bytes=64)

    assert cost_for(_dace.ScheduleType.GPU_Device).regime() == "bandwidth"
    assert cost_for(_dace.ScheduleType.CPU_Multicore).regime() == "bandwidth"
    assert cost_for(_dace.ScheduleType.Sequential).regime() == "latency"


def test_concurrency_override_controls_the_regime():
    """A caller who knows the true MLP overrides the estimate: a parallel schedule with a
    dependency-limited concurrency of 1 (a pointer chase) is latency-bound; a single CPU core cannot
    saturate DRAM either."""
    _, st, me = _build((N, 1))
    assert analyze_loop_nest(st, me, P, block_bytes=64, concurrency=1.0).regime() == "latency"
    assert analyze_loop_nest(st, me, P, block_bytes=64, concurrency=P.concurrency).regime() == "latency"
    assert analyze_loop_nest(st, me, P, block_bytes=64, concurrency=float("inf")).regime() == "bandwidth"


def test_sequential_affine_nest_gets_prefetch_inclusive_concurrency():
    """Sequential does NOT mean one request at a time -- and for a prefetch-friendly affine stream
    it does not even mean the demand-miss knee: the prefetcher keeps line fills in flight beyond the
    miss queue, and the single-core streaming bandwidth already embodies that. Little's Law inverts
    it: C_stream = bw_core * L / line (~59 here), several times the chase knee (~8), which the chase
    DEFEATS by design. The knee (core_mlp) is the right C only for prefetch-hostile patterns, whose
    callers pass concurrency explicitly."""
    import dace as _dace
    sdfg = _dace.SDFG("seq_mlp")
    sdfg.add_array("A", [N, N], _dace.float64)
    sdfg.add_array("C", [N, N], _dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"}, schedule=_dace.ScheduleType.Sequential)
    t = st.add_tasklet("t", {"a"}, {"c"}, "c = a * 2.0")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=_dace.Memlet("A[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=_dace.Memlet("C[i,j]"))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert cost.concurrency == P.core_stream_mlp == P.bw_core * P.L / P.line_bytes
    assert cost.concurrency > P.core_mlp > 1.0  # stream > demand-miss knee > chain
    # the demand-miss knee stays available for prefetch-hostile callers, measured or defaulted
    p_measured = LogGP(L=P.L, o=0.0, g=P.g, G=P.G, line_bytes=64, bw_saturated=P.bw_saturated,
                       bw_core=P.bw_core, c_core=8.0)
    assert p_measured.core_mlp == 8.0
    assert analyze_loop_nest(st, me, p_measured, block_bytes=64,
                             concurrency=p_measured.core_mlp).concurrency == 8.0


def test_total_time_is_one_continuous_formula():
    """total_time = max(B*G, M*L/C) -- ONE expression, no regime branch. C=inf lands on the
    bandwidth term, C=1 recovers the dependent chain (every miss waits out L), and between them the
    time falls monotonically as 1/C until the crossover. The old switch returned the serialized SUM
    below the BDP -- a ~26x cliff at the threshold; the max is continuous there."""
    pytest = __import__("pytest")
    _, st, me = _build((N, 1))
    bw = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=float("inf"))
    lat = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=1.0)
    # endpoints
    assert _at(bw.total_time()) == pytest.approx(_at(bw.total_time_bandwidth()))
    assert _at(lat.total_time()) == pytest.approx(_at(lat.total_messages()) * P.L)  # the chase: M*L
    assert _at(lat.total_time()) > _at(bw.total_time())
    # monotone in C, and continuous at the crossover C* = M*L/(B*G)
    times = [_at(analyze_loop_nest(st, me, P, block_bytes=64, concurrency=c).total_time())
             for c in (1.0, 4.0, 8.0, 64.0, 148.0, 1e6)]
    assert times == sorted(times, reverse=True)
    m, b = _at(lat.total_messages()), _at(lat.total_bytes())
    c_star = m * P.L / (b * P.G)
    just_below = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=c_star * 0.999)
    just_above = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=c_star * 1.001)
    assert _at(just_below.total_time()) == pytest.approx(_at(just_above.total_time()), rel=2e-3)


def test_latency_term_quantifies_exposed_latency():
    """Where LogP explains load/store performance wrt latency: at fixed traffic, the latency term
    scales as 1/C -- doubling the exposed MLP halves the exposed-latency time, exactly the measured
    MLP-sweep behavior (1.97x at 2 chains, 3.99x at 4)."""
    pytest = __import__("pytest")
    _, st, me = _build((N, 1))
    t1 = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=1.0)
    t2 = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=2.0)
    t4 = analyze_loop_nest(st, me, P, block_bytes=64, concurrency=4.0)
    assert _at(t1.total_time_latency()) == pytest.approx(2 * _at(t2.total_time_latency()))
    assert _at(t1.total_time_latency()) == pytest.approx(4 * _at(t4.total_time_latency()))


def test_gpu_sector_granularity_changes_the_message_count():
    """block_bytes is the granularity: a 32B GPU sector (4 fp64) shares fewer elements than a 64B
    line (8 fp64), so contiguous access issues more messages -- higher latency term."""
    _, st, me = _build((N, 1))
    line = analyze_loop_nest(st, me, P, block_bytes=64)
    sector = analyze_loop_nest(st, me, P, block_bytes=32)
    assert _at(sector.latency_per_iter()) > _at(line.latency_per_iter())


GPU = LogGP(L=500e-9, o=0.0, g=1e-9, G=gap_from_bandwidth(1000e9), line_bytes=128, bw_saturated=1000e9,
            bw_core=1000e9, sector_bytes=32)


def test_request_and_transfer_granularities_are_split_when_p_carries_both():
    """With block_bytes omitted, messages count at the REQUEST granularity (line, 128B) and bytes at
    the TRANSFER granularity (sector, 32B). A contiguous fp64 stream: 1/16 requests per element but
    1/4 sectors -> bytes/iter = (1/4)*32 = 8 = the useful bytes (no waste), while one collapsed
    128B granularity would claim (1/16)*128 = 8 too BUT only 1/16 latency events where a collapsed
    32B one would claim 1/4 -- 4x apart. The split keeps both terms right at once."""
    pytest = __import__("pytest")
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, GPU)
    a = cost.arrays["A"]
    assert _at(a.messages_per_iter) == pytest.approx(1.0 / 16, rel=0.02)  # 128B / 8B = 16 elems/request
    assert _at(a.sectors_per_iter) == pytest.approx(1.0 / 4, rel=0.02)  # 32B / 8B = 4 elems/sector
    assert _at(a.bytes_moved_per_iter) == pytest.approx(8.0, rel=0.02)  # contiguous: no wasted bytes


def test_scattered_access_needs_line_over_sector_more_concurrency_to_saturate():
    """The latency insight the split buys: saturation needs C >= L/(k*sector*G) with k the sectors
    used per request. A scattered access (k=1) needs line/sector = 4x the concurrency of a coalesced
    one (k=4) to reach the same channels. Computed from the nest terms: C* = M*L/(B*G)."""
    pytest = __import__("pytest")
    _, st_contig, me_contig = _build((N, 1))
    _, st_scat, me_scat = _build((1, N))  # strided inner: every element a fresh request AND sector
    contig = analyze_loop_nest(st_contig, me_contig, GPU)
    scat = analyze_loop_nest(st_scat, me_scat, GPU)

    def c_star(cost, name):
        a = cost.arrays[name]
        return _at(a.messages_per_iter) * GPU.L / (_at(a.bytes_moved_per_iter) * GPU.G)

    # contiguous A: k = 4 sectors per 128B request; scattered A: k = 1 (one 32B sector per request)
    assert c_star(scat, "A") / c_star(contig, "A") == pytest.approx(GPU.line_bytes / GPU.sector_bytes,
                                                                    rel=0.05)


def test_line_granular_analysis_not_element_granular():
    """A[:] += B[:] analyzed per ELEMENT would report one message latency + bandwidth per iteration
    with low line utilization -- wrong twice: the hardware requests a FULL line, and the next 7
    iterations hit that same line. Counting NEW blocks per iteration gets both right at once: 1/8
    messages per fp64 element and every fetched byte used (utilization 1)."""
    pytest = __import__("pytest")
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    a = cost.arrays["A"]
    assert _at(a.messages_per_iter) == pytest.approx(1.0 / 8, rel=0.02)  # NOT 1 per element
    # full utilization: bytes moved == bytes useful (8 per fp64 element)
    assert _at(a.bytes_moved_per_iter) == pytest.approx(8.0, rel=0.02)


def test_populated_cores_refine_the_saturated_assumption():
    """Distributing the iterations over n pinned cores in contiguous chunks: aggregate concurrency
    n_cores * core_mlp instead of blanket inf. The honest number on this box: 16 cores x 8
    outstanding = 128 < BDP ~148 -- even all cores do not QUITE saturate, which inf hides."""
    import dace as _dace
    sdfg = _dace.SDFG("cores")
    sdfg.add_array("A", [N, N], _dace.float64)
    sdfg.add_array("C", [N, N], _dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"}, schedule=_dace.ScheduleType.CPU_Multicore)
    t = st.add_tasklet("t", {"a"}, {"c"}, "c = a * 2.0")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=_dace.Memlet("A[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=_dace.Memlet("C[i,j]"))

    p_measured = LogGP(L=P.L, o=0.0, g=P.g, G=P.G, line_bytes=64, bw_saturated=P.bw_saturated,
                       bw_core=P.bw_core, c_core=8.0)
    saturated = analyze_loop_nest(st, me, p_measured, block_bytes=64)
    populated = analyze_loop_nest(st, me, p_measured, block_bytes=64, n_cores=16)
    assert saturated.concurrency == float("inf")
    # STREAMING nest: prefetch-inclusive units saturate easily (the triad saturates at ~2 cores)
    assert populated.concurrency == 16 * p_measured.core_stream_mlp
    assert populated.regime() == "bandwidth"
    # SCATTERED nest (prefetch-hostile): the demand-miss budget is the honest unit -- and 16 x 8 =
    # 128 < BDP ~148: a scattered parallel nest does NOT quite saturate, which blanket inf hides.
    scattered = analyze_loop_nest(st, me, p_measured, block_bytes=64,
                                  concurrency=16 * p_measured.core_mlp)
    assert scattered.concurrency == 128.0
    assert scattered.regime() == "latency"
    # explicit concurrency overrides n_cores
    assert analyze_loop_nest(st, me, p_measured, block_bytes=64, concurrency=4.0,
                             n_cores=16).concurrency == 4.0


def test_gpu_thread_per_cell_lane_axis_carries_the_coalescing():
    """One GPU thread per cell: no thread streams anything -- adjacency moves to the LANES of a
    warp, and the coalescer's merge of 32 lane addresses IS the new-blocks count over the innermost
    (lane-mapped) axis. Contiguous fp64: 256B per warp = 2 requests of 128B = 1/16 per element.
    Permute the layout so lanes are strided and every lane pays its own request: 16x the messages.
    The layout and the lane mapping jointly set M -- the layout/schedule interaction in one term."""
    pytest = __import__("pytest")
    _, st_c, me_c = _build((N, 1))  # innermost axis (lane axis) contiguous
    _, st_s, me_s = _build((1, N))  # innermost axis strided: one sector per lane
    contig = analyze_loop_nest(st_c, me_c, GPU)
    strided = analyze_loop_nest(st_s, me_s, GPU)
    assert _at(contig.arrays["A"].messages_per_iter) == pytest.approx(1.0 / 16, rel=0.02)
    assert _at(strided.arrays["A"].messages_per_iter) == pytest.approx(1.0, rel=0.02)


def test_gpu_occupancy_is_the_unit_count():
    """GPU latency hiding is warp SWITCHING, not per-lane OoO: units = resident warps (grid size x
    occupancy), unit MLP = outstanding loads per warp. An under-occupied kernel misses the ~3900-
    request GPU BDP and lands latency-regime -- the model catches an occupancy shortfall that the
    blanket saturated assumption hides. n_cores generalizes to n_units."""
    import dace as _dace
    sdfg = _dace.SDFG("gpu_occ")
    sdfg.add_array("A", [N, N], _dace.float64)
    sdfg.add_array("C", [N, N], _dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    me, mx = st.add_map("m", {"i": "0:N", "j": "0:N"}, schedule=_dace.ScheduleType.GPU_Device)
    t = st.add_tasklet("t", {"a"}, {"c"}, "c = a * 2.0")
    st.add_memlet_path(st.add_read("A"), me, t, dst_conn="a", memlet=_dace.Memlet("A[i,j]"))
    st.add_memlet_path(t, mx, st.add_write("C"), src_conn="c", memlet=_dace.Memlet("C[i,j]"))

    # per-warp outstanding ~4 (in-order lanes; scoreboard slots), measured by a multi-warp P-chase
    gpu = LogGP(L=GPU.L, o=0.0, g=GPU.g, G=GPU.G, line_bytes=128, bw_saturated=GPU.bw_saturated,
                bw_core=GPU.bw_core, sector_bytes=32, c_core=4.0)
    bdp = analyze_loop_nest(st, me, gpu).bandwidth_delay_product()
    assert bdp == __import__("pytest").approx(500e-9 * 1000e9 / 128)  # ~3906 requests

    # a tiny grid: 100 resident warps x 4 outstanding = 400 << 3906 -> latency-regime
    small_grid = analyze_loop_nest(st, me, gpu, n_cores=100)
    assert small_grid.concurrency == 400.0
    assert small_grid.regime() == "latency"
    # a full grid: 2048 resident warps x 4 = 8192 >= 3906 -> saturated
    full_grid = analyze_loop_nest(st, me, gpu, n_cores=2048)
    assert full_grid.regime() == "bandwidth"


if __name__ == "__main__":
    test_latency_and_bandwidth_per_iteration_are_both_reported()
    test_time_per_iter_is_latency_plus_bandwidth()
    test_layout_change_moves_the_predicted_cost()
    test_local_memory_is_free()
    test_serialized_is_a_diagnostic_ceiling_not_the_answer()
    test_regime_from_the_schedule()
    test_concurrency_override_controls_the_regime()
    test_sequential_affine_nest_gets_the_core_mlp_not_one()
    test_total_time_is_one_continuous_formula()
    test_latency_term_quantifies_exposed_latency()
    test_gpu_sector_granularity_changes_the_message_count()
    test_request_and_transfer_granularities_are_split_when_p_carries_both()
    test_scattered_access_needs_line_over_sector_more_concurrency_to_saturate()
    test_line_granular_analysis_not_element_granular()
    test_populated_cores_refine_the_saturated_assumption()
    test_gpu_thread_per_cell_lane_axis_carries_the_coalescing()
    test_gpu_occupancy_is_the_unit_count()
    print("logp_analysis tests PASS")
