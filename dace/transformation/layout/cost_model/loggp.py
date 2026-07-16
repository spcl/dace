# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP/LogGP parameters of a memory level, and the fit that extracts them from the microbenchmarks.

The parameters (all SI: seconds, seconds/byte):

  * ``L`` -- unloaded round-trip latency of one line. A pointer-chase step already contains BOTH legs
    (the request and the reply), so ``L`` is the whole round trip and is NOT halved the way a network
    ping-pong's RTT is; Culler's ``2L + 4o`` must not be layered on top.
  * ``o`` -- per-message overhead. 0 in v1: local access is free by the model's framing, and ``o`` is
    exactly the local issue occupancy.
  * ``g`` -- the minimum interval between requests issued by ONE core (the reciprocal of a core's peak
    request rate). From the concurrency sweep.
  * ``G`` -- the per-byte gap, ``1 / bandwidth``. Bandwidth is a property of the memory CHANNELS, not
    of a core: a single core cannot saturate the channels (its outstanding-miss budget runs out
    first), so ``G`` is ``1 / BW_saturated`` measured with ALL cores. The single-core bandwidth is a
    diagnostic (``bw_core``), never the model's ``G`` -- using it would make every transfer look ~2.5x
    faster than the channels can actually deliver.

Why both ``L`` and ``G`` and not just a sum: charging ``L`` per line and summing gives ~0.8 GB/s here
against a measured 39.5 GB/s single thread -- ~40x low -- because outstanding requests OVERLAP their
latency. ``L`` bounds a single dependent access; ``G`` bounds a stream. The achievable rate is
``min(1/G, concurrency * line / L)``: latency-bound when few requests are in flight, bandwidth-bound
once the channels saturate.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class LogGP:
    """LogP/LogGP parameters of ONE memory level (e.g. global DRAM), in seconds and seconds/byte."""
    L: float  # s -- unloaded round-trip latency of one line
    o: float  # s -- per-message overhead (0 in v1)
    g: float  # s -- minimum interval between requests from one core
    G: float  # s/byte -- per-byte gap = 1 / BW_saturated (channel-limited, all cores)
    line_bytes: int  # the memory-transfer granularity (64 on x86, 32 per GPU sector)
    bw_saturated: float  # bytes/s -- all-core channel bandwidth; G == 1 / bw_saturated
    bw_core: float  # bytes/s -- single-core bandwidth (< bw_saturated); DIAGNOSTIC, not the model G

    @property
    def concurrency(self) -> float:
        """LogP's ``L/g`` -- the number of requests one core keeps in flight. Its agreement with the
        measured outstanding-miss knee is a consistency check, not a free parameter."""
        return self.L / self.g


def gap_from_bandwidth(bytes_per_s: float) -> float:
    """Per-byte gap ``G = 1 / bandwidth``."""
    return 1.0 / bytes_per_s


def lines_touched(nbytes: int, line_bytes: int) -> int:
    """Number of whole lines a contiguous ``nbytes`` request moves. Memory moves whole lines, so a
    4-byte request still costs one line -- this is the granularity that makes the model SEE a layout
    change (Permute/Pad/Block move bytes between lines without changing ``nbytes``)."""
    if nbytes <= 0:
        return 0
    return (nbytes + line_bytes - 1) // line_bytes


def message_time(p: LogGP, nbytes: int) -> float:
    """Time of ONE contiguous request of ``nbytes``: the first line exposes the full latency, the rest
    stream at the channel gap. ``o + L + (lines - 1) * line_bytes * G``. Rounds up to line
    granularity, since a partial line still moves a whole line."""
    lines = lines_touched(nbytes, p.line_bytes)
    if lines == 0:
        return 0.0
    return p.o + p.L + (lines - 1) * p.line_bytes * p.G


def achievable_rate(p: LogGP) -> float:
    """Sustained bytes/s: ``min(1/G, concurrency * line / L)``. Bandwidth-bound once the channels
    saturate, latency-bound while few requests are outstanding. NOT ``line / L`` (that is the
    single-outstanding-miss rate, ~40x below reality)."""
    latency_bound = p.concurrency * p.line_bytes / p.L
    bandwidth_bound = 1.0 / p.G
    return min(bandwidth_bound, latency_bound)


def memory_time(blocks_per_iter: float, total_iters: float, p: LogGP) -> float:
    """Predicted memory time (seconds) of a kernel from its block-transaction count.

    This is where the latency model MEETS the layout: ``blocks_per_iter`` (from
    :func:`~dace.transformation.layout.cost_model.blocks_touched.average_blocks_touched`) is the only
    term a layout transformation moves, and the total time is the block-message traffic divided by
    the rate the hardware sustains.

        time = blocks_per_iter * total_iters * line_bytes / achievable_rate

    The consequence -- the answer to "can a latency model rank layouts": ``total_iters``,
    ``line_bytes`` and ``achievable_rate`` are the SAME for every layout of one kernel on one device,
    so the predicted time is proportional to ``blocks_per_iter``. The layout ranking is therefore
    carried entirely by the block count and is INVARIANT to whether the kernel runs latency-bound or
    bandwidth-bound -- both regimes scale the block count by the same per-block constant. A latency
    model explains layout performance precisely because a layout change is a change in the number of
    block messages. (This assumes the achievable rate itself does not depend on the layout; a badly
    scattered access that cannot sustain concurrency drops to the latency-bound branch and is penalised
    a second time -- a refinement, noted, not yet modelled.)
    """
    total_bytes = blocks_per_iter * total_iters * p.line_bytes
    return total_bytes / achievable_rate(p)


@dataclass(frozen=True)
class Fit:
    """A least-squares ``T(n) = alpha + beta * n`` over the message-size sweep."""
    alpha: float  # s -- intercept; identifies with L
    beta: float  # s/byte -- slope; identifies with the single-core per-byte gap
    residual: float  # relative RMS residual, so a bad fit cannot be silently trusted


def fit_message_size(sizes: Sequence[int], times: Sequence[float]) -> Fit:
    """Fit ``T(n) = alpha + beta * n`` over the message-size sweep at ONE memory level.

    LogGP says ``T(n) = o + L + (n - 1) * G``, so ``alpha ~= L`` and ``beta ~= G`` -- and those two
    identities are how the whole parametrization is validated: ``alpha`` must reproduce ``L`` measured
    independently by the working-set sweep, and ``beta`` must reproduce the single-core ``1/BW``
    measured independently by the bandwidth sweep.

    Weighted by ``1/T`` because timing error is MULTIPLICATIVE: unweighted, the large-``n`` points
    dominate and ``alpha`` -- the term we actually want to check -- is left undetermined. The sweep
    must stay within one memory level (fixed arena, all misses to DRAM); a fit that crosses a cache
    boundary has a high R2 and predicts badly inside every regime.
    """
    if len(sizes) != len(times) or len(sizes) < 2:
        raise ValueError("need at least two (size, time) points of equal length")
    weight = [1.0 / t if t > 0 else 0.0 for t in times]
    sw = sum(weight)
    swx = sum(w * x for w, x in zip(weight, sizes))
    swy = sum(w * y for w, y in zip(weight, times))
    swxx = sum(w * x * x for w, x in zip(weight, sizes))
    swxy = sum(w * x * y for w, x, y in zip(weight, sizes, times))
    denom = sw * swxx - swx * swx
    if denom == 0:
        raise ValueError("degenerate message-size sweep (all sizes equal)")
    beta = (sw * swxy - swx * swy) / denom
    alpha = (swy - beta * swx) / sw
    predicted = [alpha + beta * x for x in sizes]
    num = sum(w * (y - p) ** 2 for w, y, p in zip(weight, times, predicted))
    mean = swy / sw
    den = sum(w * (y - mean) ** 2 for w, y in zip(weight, times))
    residual = math.sqrt(num / den) if den > 0 else 0.0
    return Fit(alpha=alpha, beta=beta, residual=residual)


def validate(p: LogGP, fit: Fit, peak_bytes_per_s: float, knee_concurrency: float,
             latency_tol: float = 0.10, gap_tol: float = 0.20) -> List[str]:
    """Reasons to REJECT the parametrization; empty means accept.

    There is no external oracle here (Intel MLC is Intel-only, no counters at this paranoia level), so
    the checks are internal CONSISTENCY -- each parameter measured two independent ways must agree,
    which is strictly stronger than agreeing with one tool:

      * ``fit.alpha`` (message-size fit) vs ``L`` (working-set sweep) -- same quantity, two experiments.
      * ``fit.beta`` (message-size fit) vs single-core ``1/bw_core`` (bandwidth sweep) -- ditto.
      * ``concurrency = L/g`` must land on the measured outstanding-miss knee -- LogP's capacity bound
        IS Little's Law IS the miss-buffer count; over-determined, so a mismatch means a broken run.
      * saturated bandwidth must not EXCEED the hardware peak -- exceeding it proves a cached array.
    """
    reasons: List[str] = []
    if p.L <= 0 or p.G <= 0 or p.g <= 0:
        reasons.append("non-positive L, G, or g")
        return reasons
    if _relative_gap(fit.alpha, p.L) > latency_tol:
        reasons.append(f"fit intercept {fit.alpha * 1e9:.1f} ns disagrees with measured L "
                       f"{p.L * 1e9:.1f} ns by > {latency_tol:.0%}")
    core_gap = gap_from_bandwidth(p.bw_core)
    if _relative_gap(fit.beta, core_gap) > gap_tol:
        reasons.append(f"fit slope {fit.beta * 1e9:.4f} ns/B disagrees with single-core 1/bw "
                       f"{core_gap * 1e9:.4f} ns/B by > {gap_tol:.0%}")
    if _relative_gap(p.concurrency, knee_concurrency) > 0.35:
        reasons.append(f"L/g = {p.concurrency:.1f} does not match the outstanding-miss knee "
                       f"{knee_concurrency:.1f}")
    if p.bw_saturated > peak_bytes_per_s:
        reasons.append(f"saturated bandwidth {p.bw_saturated / 1e9:.1f} GB/s exceeds hardware peak "
                       f"{peak_bytes_per_s / 1e9:.1f} GB/s (a cache-resident array)")
    return reasons


def _relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0
