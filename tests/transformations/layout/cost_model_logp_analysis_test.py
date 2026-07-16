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
    assert _at(col.total_time_overlapped()) > 2 * _at(row.total_time_overlapped())


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


def test_serialized_exceeds_overlapped():
    """The serialized per-iteration sum charges every message its full latency; the overlapped total
    lets requests hide under the channel bandwidth and is far lower (the ~40x LogP gap)."""
    _, st, me = _build((N, 1))
    cost = analyze_loop_nest(st, me, P, block_bytes=64)
    assert _at(cost.total_time_serialized()) > 10 * _at(cost.total_time_overlapped())


def test_gpu_sector_granularity_changes_the_message_count():
    """block_bytes is the granularity: a 32B GPU sector (4 fp64) shares fewer elements than a 64B
    line (8 fp64), so contiguous access issues more messages -- higher latency term."""
    _, st, me = _build((N, 1))
    line = analyze_loop_nest(st, me, P, block_bytes=64)
    sector = analyze_loop_nest(st, me, P, block_bytes=32)
    assert _at(sector.latency_per_iter()) > _at(line.latency_per_iter())


if __name__ == "__main__":
    test_latency_and_bandwidth_per_iteration_are_both_reported()
    test_time_per_iter_is_latency_plus_bandwidth()
    test_layout_change_moves_the_predicted_cost()
    test_local_memory_is_free()
    test_serialized_exceeds_overlapped()
    test_gpu_sector_granularity_changes_the_message_count()
    print("logp_analysis tests PASS")
