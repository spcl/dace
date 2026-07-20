# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""When an intermediate layout change pays for itself, and the two granularities LogP needs.

A relayout is not free: it reads the array once and writes it once. It pays when the delta the new
layout buys, times the number of nests that use it, exceeds that. Two model facts drive the answer:

* A WRITE to a partially-covered block moves it TWICE (read-for-ownership, then writeback); a write
  covering the whole block streams and moves it once. So a bad layout costs a written array roughly
  double what it costs a read-only one -- and relayout pays back sooner there.
* Latency is paid per REQUEST (128 B on NVIDIA) but bandwidth per SECTOR (32 B). One granularity
  cannot serve both without a 4x error on GPU.
"""
import sympy as sp
import pytest

import dace
from dace.transformation.layout.cost_model.loggp import LogGP, achievable_rate
from dace.transformation.layout.cost_model.relayout import (array_bytes, bandwidth_efficiency, block_traffic,
                                                            break_even_uses, cache_efficiency, max_layout_delta,
                                                            nest_time_by_efficiency, relayout_pays, relayout_pays_by_efficiency,
                                                            relayout_time, single_pass_efficiency_threshold,
                                                            streaming_relayout_time)

# A DRAM-ish CPU level: 95 ns latency, 100 GB/s, 64-byte lines, no sectoring.
CPU = LogGP(L=95e-9, o=0.0, g=4e-9, G=1.0 / 100e9, line_bytes=64, bw_saturated=100e9, bw_core=20e9)
# A GPU level: a 128-byte request granularity, but 32-byte sectors actually cross the channels.
GPU = LogGP(L=500e-9, o=0.0, g=1e-9, G=1.0 / 1000e9, line_bytes=128, bw_saturated=1000e9, bw_core=1000e9,
            sector_bytes=32)


def _arr(n=1024):
    sdfg = dace.SDFG("t")
    sdfg.add_array("A", [n], dace.float64)
    return sdfg.arrays["A"]


def test_sector_defaults_to_line_when_unsectored():
    """x86 has no sectoring: one number, and nothing about the CPU model changes."""
    assert CPU.sector_bytes == CPU.line_bytes == 64


def test_gpu_keeps_request_and_transfer_granularity_apart():
    """The whole point: latency granularity != bandwidth granularity. Collapsing them is the 4x."""
    assert GPU.line_bytes == 128  # what one request covers -> the L term
    assert GPU.sector_bytes == 32  # what actually crosses the channels -> the G term
    assert GPU.line_bytes // GPU.sector_bytes == 4


def test_partial_write_costs_two_transfers_and_a_covering_write_costs_one():
    """A partial write must fetch the block before merging into it; a covering write streams."""
    partial = block_traffic(10, CPU, written=True, covers_full_block=False)
    covering = block_traffic(10, CPU, written=True, covers_full_block=True)
    read = block_traffic(10, CPU, written=False, covers_full_block=False)
    assert partial == 2 * covering  # read-for-ownership doubles it
    assert covering == read == 10 * 64


def test_written_array_has_a_wider_layout_gap_than_a_read_array():
    """The consequence for placement: layout matters MORE where the nest writes, because a bad
    layout there forces a fetch a good layout would not need at all."""
    a = _arr()
    read_gap = max_layout_delta(a, CPU, written=False)
    write_gap = max_layout_delta(a, CPU, written=True)
    r = CPU.sector_bytes / 8  # elements per block, fp64
    rate = achievable_rate(CPU, float("inf"))  # relayout break-evens live in the saturated scope
    assert float(read_gap) == pytest.approx(float(array_bytes(a) * (r - 1) / rate))
    assert float(write_gap) == pytest.approx(float(array_bytes(a) * (2 * r - 1) / rate))
    assert float(write_gap) > 2 * float(read_gap)


def test_relayout_can_pay_on_a_single_use():
    """The claim that kills "mid-program relayout is never good": a relayout costs 2*S, while the gap
    it can recover is 7*S (read) or 15*S (written). One use suffices once it recovers enough."""
    a = _arr()
    t_relayout = streaming_relayout_time(a, CPU)
    gap = max_layout_delta(a, CPU, written=False)
    assert float(t_relayout) < float(gap)  # a single full pass CAN redeem it

    # A nest that recovers the whole gap: one use is enough.
    t_before, t_after = float(gap), 0.0
    assert relayout_pays(t_before, t_after, t_relayout, uses=1)
    # A nest that recovers only a tenth of it: one use is not.
    assert not relayout_pays(float(gap) * 0.1, 0.0, t_relayout, uses=1)


def test_break_even_uses_counts_the_nests_needed():
    a = _arr()
    t_relayout = streaming_relayout_time(a, CPU)
    delta = float(t_relayout) / 4  # each use recovers a quarter of the relayout cost
    assert break_even_uses(delta, 0.0, t_relayout) == 4
    assert not relayout_pays(delta, 0.0, t_relayout, uses=3)
    assert relayout_pays(delta, 0.0, t_relayout, uses=4)


def test_a_slower_layout_is_never_redeemed_by_more_uses():
    """No number of uses redeems a layout that is not faster -- break_even_uses says so with None
    rather than returning a huge number that a caller might treat as a threshold."""
    assert break_even_uses(1.0, 2.0, 0.5) is None
    assert not relayout_pays(1.0, 2.0, 0.5, uses=10**9)


def test_break_even_refuses_a_symbolic_delta():
    """Placement needs a number. A symbolic delta means the caller forgot to substitute the nest's
    symbols, and silently returning something would hide that."""
    N = dace.symbol("N")
    with pytest.raises(ValueError, match="concrete times"):
        break_even_uses(sp.sympify(N), 0.0, 1.0)


def test_break_even_refuses_a_symbolic_relayout_time():
    """t_relayout gets the same concreteness check as the delta -- a symbolic value in EITHER
    argument is the same caller mistake and must surface as the same ValueError. Unchecked, a
    symbolic t_relayout reached float() and raised a bare TypeError that named neither the argument
    at fault nor the fix (substitute the nest's symbols)."""
    N = dace.symbol("N")
    with pytest.raises(ValueError, match="concrete times"):
        break_even_uses(2.0, 1.0, sp.sympify(N))
    with pytest.raises(ValueError, match="concrete times"):
        break_even_uses(sp.sympify(N), 0.0, 1.0)
    assert break_even_uses(4.0, 1.0, 10.0) == 4  # both concrete: ceil(10 / (4 - 1)) == 4


def test_cache_efficiency_is_a_ratio_of_useful_to_moved():
    """One fp64 element used out of every 64-byte block = 1/8."""
    assert float(cache_efficiency(8, block_traffic(1, CPU, written=False, covers_full_block=False))) == 1.0 / 8
    assert float(cache_efficiency(64, block_traffic(1, CPU, written=False, covers_full_block=False))) == 1.0


def test_relayout_time_takes_traffic_rather_than_assuming_it_streams():
    """The default `pure` expansion is one flat mapped-tasklet copy, so it does NOT move whole blocks
    on both sides. The caller passes measured traffic; the model must not override it with an
    optimistic streaming assumption."""
    a = _arr()
    streamed = float(streaming_relayout_time(a, CPU))
    naive = float(relayout_time(array_bytes(a) * 9, CPU))  # read S + write 8*S of partial blocks
    assert naive > streamed
    assert float(relayout_time(2 * array_bytes(a), CPU)) == pytest.approx(streamed)


# --------------------------------------------------------------------------------------------- #
#  Combining cache efficiency with LogP
# --------------------------------------------------------------------------------------------- #
def test_efficiency_spans_one_sixteenth_to_one():
    """eps = eps_spatial * eps_write. Worst: one fp64 element per 64-byte block AND a partial write
    (so it is fetched before being merged) -> 8 / (2*64) = 1/16. Best: a fully covered block -> 1."""
    worst = bandwidth_efficiency(8, 1, CPU, written=True, covers_full_block=False)
    best = bandwidth_efficiency(64, 1, CPU, written=False, covers_full_block=False)
    read_worst = bandwidth_efficiency(8, 1, CPU, written=False, covers_full_block=False)
    assert float(worst) == 1.0 / 16
    assert float(read_worst) == 1.0 / 8
    assert float(best) == 1.0


def test_efficiency_framing_reproduces_the_independent_delta_bounds():
    """The check that the factorization is RIGHT: 1/eps - 1 must reproduce max_layout_delta, which
    was derived separately. Read-only worst eps=1/8 -> 7*S; written worst eps=1/16 -> 15*S."""
    a = _arr()
    S = float(array_bytes(a))
    for written, eps in ((False, 1.0 / 8), (True, 1.0 / 16)):
        via_eps = S * (1.0 / eps - 1.0) / achievable_rate(CPU, float("inf"))  # saturated scope
        via_delta = float(max_layout_delta(a, CPU, written=written))
        assert via_eps == pytest.approx(via_delta)


def test_efficiency_cannot_determine_the_latency_term():
    """The formal reason not to collapse LogP into elementwise x efficiency: two GPU access patterns
    can move the SAME bytes (identical eps) while costing 1 vs 4 messages. Efficiency is blind to
    it, so a model that keeps only eps cannot tell these apart -- and they differ in latency."""
    # 4 sectors of 32 B, all useful. Same traffic, same epsilon...
    packed_eps = bandwidth_efficiency(4 * 32, 4, GPU)
    spread_eps = bandwidth_efficiency(4 * 32, 4, GPU)
    assert float(packed_eps) == float(spread_eps) == 1.0
    # ...but 4 sectors inside ONE 128-byte request vs spread over FOUR requests differ 4x in messages.
    packed_msgs, spread_msgs = 1, 4
    useful, conc = 4 * 32, 1.0
    t_packed = nest_time_by_efficiency(useful, packed_eps, packed_msgs, GPU, conc)
    t_spread = nest_time_by_efficiency(useful, spread_eps, spread_msgs, GPU, conc)
    assert float(t_spread) > float(t_packed)  # identical epsilon, different time


def test_nest_time_takes_the_binding_term():
    """max(), not sum: the terms overlap, and whichever binds is the regime."""
    # Latency-bound: many messages, few useful bytes.
    assert float(nest_time_by_efficiency(64, 1.0, 10_000, CPU, 1.0)) == pytest.approx(10_000 * CPU.L)
    # Bandwidth-bound: huge traffic, one message.
    big = 10**9
    assert float(nest_time_by_efficiency(big, 1.0, 1, CPU, 1.0)) == pytest.approx(big * CPU.G)


def test_a_single_pass_redeems_a_relayout_below_one_third_efficiency():
    """The headline rule, and it is hardware-free: eps < 1/3 -> one pass already pays."""
    assert single_pass_efficiency_threshold(1.0) == pytest.approx(1.0 / 3)
    assert relayout_pays_by_efficiency(0.32, 1.0, passes=1)  # just under -> pays
    assert not relayout_pays_by_efficiency(0.34, 1.0, passes=1)  # just over -> does not
    # The worst layouts are far below the threshold, so they pay trivially.
    assert relayout_pays_by_efficiency(1.0 / 8, 1.0, passes=1)  # read-only worst
    assert relayout_pays_by_efficiency(1.0 / 16, 1.0, passes=1)  # partial-write worst


def test_more_passes_redeem_a_milder_layout_gap():
    """A gap too small for one pass is redeemed by reuse -- the amortization k16 measured (1.4)."""
    assert not relayout_pays_by_efficiency(0.5, 1.0, passes=1)  # gain 1 per pass, cost 2 -> short
    assert relayout_pays_by_efficiency(0.5, 1.0, passes=2)  # 2*1 == 2 -> exactly paid for itself
    assert relayout_pays_by_efficiency(0.5, 1.0, passes=3)  # 3*1 > 2 -> a clear win


def test_relayout_to_an_equal_layout_never_pays():
    for eps in (0.1, 0.5, 1.0):
        assert not relayout_pays_by_efficiency(eps, eps, passes=10**6)


def test_efficiency_outside_the_unit_interval_is_rejected():
    """eps > 1 would mean using more bytes than were moved; eps <= 0 is nonsense. Either means the
    caller mixed up useful and traffic -- fail rather than return a confident wrong answer."""
    with pytest.raises(ValueError):
        relayout_pays_by_efficiency(1.5, 1.0)
    with pytest.raises(ValueError):
        relayout_pays_by_efficiency(0.5, 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("relayout cost-model tests PASS")
