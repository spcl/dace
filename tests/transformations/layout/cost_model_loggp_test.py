# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP/LogGP parameters, the message-size fit, and the regime model. Pure -- no measurement, so
these run anywhere and pin the algebra the measured parameters flow through."""
import math

import pytest

from dace.transformation.layout.cost_model.loggp import (LogGP, Fit, gap_from_bandwidth, lines_touched,
                                                         message_time, achievable_rate, fit_message_size, validate)


def _dram(L=95e-9, g=4e-9, bw_sat=100e9, bw_core=40e9, line=64):
    return LogGP(L=L, o=0.0, g=g, G=gap_from_bandwidth(bw_sat), line_bytes=line, bw_saturated=bw_sat,
                 bw_core=bw_core)


def test_lines_touched_rounds_up_to_granularity():
    assert lines_touched(1, 64) == 1  # a 1-byte request still moves a whole line
    assert lines_touched(64, 64) == 1
    assert lines_touched(65, 64) == 2
    assert lines_touched(0, 64) == 0


def test_message_time_first_line_is_latency_rest_streams():
    p = _dram()
    assert message_time(p, 0) == 0.0
    assert message_time(p, 64) == pytest.approx(p.L)  # one line: pure latency, no per-byte term
    # two lines: latency + one line streamed at the channel gap
    assert message_time(p, 128) == pytest.approx(p.L + 64 * p.G)


def test_gap_is_the_saturated_bandwidth_not_the_core_bandwidth():
    """G is 1/BW_saturated; a core cannot saturate the channels, so bw_core < bw_saturated and using
    it as G would understate the transfer time."""
    p = _dram(bw_sat=100e9, bw_core=40e9)
    assert p.G == pytest.approx(1.0 / 100e9)
    assert p.bw_core < p.bw_saturated


def test_achievable_rate_is_the_min_of_the_two_regimes():
    """Literal LogP charges L per line and sums, which is ~40x too low; the real rate overlaps the
    latency and is bounded by min(bandwidth ceiling, concurrency/latency)."""
    p = _dram()
    naive = p.line_bytes / p.L  # one-outstanding-miss rate: what summing L per line gives
    expected = min(1.0 / p.G, p.concurrency * p.line_bytes / p.L)
    assert achievable_rate(p) == pytest.approx(expected)
    assert achievable_rate(p) > 10 * naive  # overlap lifts it far above the naive sum
    assert achievable_rate(p) <= 1.0 / p.G  # never beats the bandwidth ceiling

    # Bandwidth-bound: enough concurrency (small g) that the channels, not latency, are the limit.
    bandwidth_bound = _dram(L=95e-9, g=0.5e-9, bw_sat=100e9)  # concurrency 190 -> lat_bound >> 1/G
    assert achievable_rate(bandwidth_bound) == pytest.approx(1.0 / bandwidth_bound.G)  # == 1/G ceiling

    # Latency-bound: few requests outstanding (large g), capped by concurrency/L below the ceiling.
    latency_bound = _dram(L=95e-9, g=90e-9, bw_sat=100e9)  # concurrency ~1.06
    assert achievable_rate(latency_bound) == pytest.approx(latency_bound.concurrency * 64 / 95e-9)
    assert achievable_rate(latency_bound) < 1.0 / latency_bound.G


def test_fit_recovers_alpha_and_beta_from_synthetic_loggp_data():
    """T(n) = L + n*G by construction; the fit must recover L as alpha and G as beta."""
    L, G = 95e-9, 1.0 / 40e9
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    times = [L + n * G for n in sizes]
    fit = fit_message_size(sizes, times)
    assert fit.alpha == pytest.approx(L, rel=1e-6)
    assert fit.beta == pytest.approx(G, rel=1e-6)
    assert fit.residual < 1e-9  # a clean line


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_message_size([64], [1e-9])
    with pytest.raises(ValueError):
        fit_message_size([64, 64, 64], [1e-9, 1e-9, 1e-9])  # all sizes equal


def test_validate_accepts_a_consistent_parametrization():
    """Independent measurements that agree: alpha==L, beta==1/bw_core, L/g on the miss knee, BW<=peak."""
    p = _dram(L=95e-9, g=4e-9, bw_sat=100e9, bw_core=40e9)
    fit = Fit(alpha=95e-9, beta=1.0 / 40e9, residual=0.01)
    knee = p.concurrency  # exactly on the knee
    assert validate(p, fit, peak_bytes_per_s=102.4e9, knee_concurrency=knee) == []


def test_validate_flags_each_inconsistency():
    p = _dram(L=95e-9, g=4e-9, bw_sat=100e9, bw_core=40e9)

    # fit intercept disagreeing with measured L
    bad_alpha = Fit(alpha=60e-9, beta=1.0 / 40e9, residual=0.01)
    assert any("intercept" in r for r in validate(p, bad_alpha, 102.4e9, p.concurrency))

    # fit slope disagreeing with single-core 1/bw
    bad_beta = Fit(alpha=95e-9, beta=1.0 / 80e9, residual=0.01)
    assert any("slope" in r for r in validate(p, bad_beta, 102.4e9, p.concurrency))

    # L/g far from the measured outstanding-miss knee
    good_fit = Fit(alpha=95e-9, beta=1.0 / 40e9, residual=0.01)
    assert any("knee" in r for r in validate(p, good_fit, 102.4e9, knee_concurrency=100.0))

    # saturated bandwidth exceeding the hardware peak == a cache-resident array
    over_peak = _dram(bw_sat=250e9)
    over_fit = Fit(alpha=95e-9, beta=1.0 / over_peak.bw_core, residual=0.01)
    assert any("exceeds hardware peak" in r for r in validate(over_peak, over_fit, 102.4e9, over_peak.concurrency))


if __name__ == "__main__":
    test_lines_touched_rounds_up_to_granularity()
    test_message_time_first_line_is_latency_rest_streams()
    test_gap_is_the_saturated_bandwidth_not_the_core_bandwidth()
    test_achievable_rate_is_the_min_of_the_two_regimes()
    test_fit_recovers_alpha_and_beta_from_synthetic_loggp_data()
    test_fit_rejects_degenerate_input()
    test_validate_accepts_a_consistent_parametrization()
    test_validate_flags_each_inconsistency()
    print("loggp tests PASS")
