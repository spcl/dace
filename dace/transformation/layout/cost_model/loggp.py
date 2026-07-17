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
from typing import List, Sequence

import sympy


@dataclass(frozen=True)
class LogGP:
    """LogP/LogGP parameters of ONE memory level (e.g. global DRAM), in seconds and seconds/byte."""
    L: float  # s -- unloaded round-trip latency of one line
    o: float  # s -- per-message overhead (0 in v1)
    g: float  # s -- minimum interval between requests from one core
    G: float  # s/byte -- per-byte gap = 1 / BW_saturated (channel-limited, all cores)
    line_bytes: int  # REQUEST granularity: what one message covers (64 x86, 128 NVIDIA) -> the L term
    bw_saturated: float  # bytes/s -- all-core channel bandwidth; G == 1 / bw_saturated
    bw_core: float  # bytes/s -- single-core bandwidth (< bw_saturated); DIAGNOSTIC, not the model G
    #: TRANSFER granularity: the smallest unit that actually crosses the channels -> the G term.
    #: x86 has no sectoring, so it equals ``line_bytes`` (the default). NVIDIA moves 32-byte SECTORS
    #: between L2 and DRAM inside a 128-byte line, so a scattered access fetches 32 bytes, not 128.
    #:
    #: These are TWO DIFFERENT granularities and collapsing them into one is a 4x error on GPU in
    #: whichever direction you pick: with one 128-byte number a scattered access overcounts bytes 4x;
    #: with one 32-byte number a coalesced access overcounts messages 4x. Latency is paid per
    #: REQUEST, bandwidth is paid per SECTOR.
    sector_bytes: int = None  # defaults to line_bytes (no sectoring)

    #: MEASURED outstanding-miss budget of one core: the knee of the concurrency sweep (latency vs
    #: number of independent pointer chains). ``None`` falls back to LogP's ``L/g`` cap. The two are
    #: not the same thing: ``L/g`` is the issue-rate ceiling, the knee is ``min(L/g, miss-queue
    #: depth)`` -- and the hardware miss queue usually binds first (measured knee ~8 on a Zen 4 core
    #: against ``L/g`` = 23.75 from the default ``g``). ``validate()`` cross-checks them.
    c_core: float = None

    def __post_init__(self):
        if self.sector_bytes is None:
            object.__setattr__(self, "sector_bytes", self.line_bytes)

    @property
    def concurrency(self) -> float:
        """LogP's ``L/g`` -- the number of requests one core keeps in flight. Its agreement with the
        measured outstanding-miss knee is a consistency check, not a free parameter."""
        return self.L / self.g

    @property
    def core_mlp(self) -> float:
        """The DEMAND-MISS budget of one core: the measured knee when available, else LogP's ``L/g``
        cap. This is the ``C`` of a PREFETCH-HOSTILE pattern -- scattered, large-stride, or
        replayed-indirect access, where every block arrives via an explicit outstanding miss. The
        chase sweep that measures it defeats the prefetcher BY DESIGN (random Hamiltonian cycle), so
        this number deliberately excludes prefetch. A DATA-DEPENDENT chain drops further, to
        ``C = 1``."""
        return self.c_core if self.c_core is not None else self.concurrency

    @property
    def core_stream_mlp(self) -> float:
        """The EFFECTIVE concurrency of one core on a prefetch-friendly (contiguous/streaming
        affine) pattern, by Little's Law on the measured single-core streaming bandwidth:
        ``bw_core * L / line_bytes``. The prefetcher is a concurrency engine beyond the miss queue --
        it keeps line fills in flight that never occupy a demand-miss slot -- and ``bw_core`` is
        prefetch-INCLUSIVE by construction (a streamed triad), so this derivation prices it without
        any new measurement. Typically several times ``core_mlp`` (e.g. 40 GB/s * 95 ns / 64 B ~ 59
        vs a demand-miss knee of ~8).

        The pair makes per-core concurrency PATTERN-DEPENDENT, which is itself a layout statement:
        a layout that scatters an access does not only touch more blocks -- it also demotes the nest
        from ``core_stream_mlp`` to ``core_mlp``, paying a second time in the latency term."""
        return self.bw_core * self.L / self.line_bytes


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


def achievable_rate(p: LogGP, concurrency: float = None) -> float:
    """Sustained bytes/s at ``concurrency`` outstanding requests: ``min(1/G, concurrency*line/L)``.
    Bandwidth-bound once the channels saturate, latency-bound while few requests are outstanding.
    NOT ``line / L`` (that is the single-outstanding-miss rate, ~40x below reality).

    ``concurrency`` is the CALLER's -- the nest's exposed MLP, ``float('inf')`` for the saturated
    all-core scope. Defaults to ``p.core_mlp`` (one core's demand-miss budget, measured when
    available); every analysis path passes it explicitly, because crediting a dependency-limited
    nest with the core's full budget was exactly the bug that made a serialized fallback look
    necessary.

    Valid when bytes and requests are line-coupled (``B = M * line``, the coalesced case). Under
    sectoring they decouple; use :func:`nest_memory_time`, the general form.
    """
    if concurrency is None:
        concurrency = p.core_mlp
    latency_bound = concurrency * p.line_bytes / p.L
    bandwidth_bound = 1.0 / p.G
    return min(bandwidth_bound, latency_bound)


def nest_memory_time(p: LogGP, bytes_moved, messages, concurrency):
    """THE nest formula -- the single source every total-time path routes through:

        T  =  max( bytes_moved * G ,  messages * L / concurrency )  +  o * messages

    One continuous expression, no regime switch:

      * ``concurrency = 1`` recovers the dependent chain exactly: ``messages * L`` (the pointer
        chase -- each miss waits out the full round trip).
      * ``concurrency >= L / (k * sector * G)``, with ``k`` the sectors used per request, makes the
        bandwidth term bind: the saturated ``B * G``.
      * In between, the latency term IS Little's Law: sustained rate ``concurrency / L`` requests/s.
        Measured (MLP sweep, 2026-07-17): rate scales linearly with independent chains -- 1.97x at
        2, 3.99x at 4, knee ~8 -- refuting the serialized ``messages * L`` sum by the measured ~8x.

    The two arguments are DIFFERENT granularities and must come from different counts: ``messages``
    at the REQUEST granularity (``line_bytes`` -- what pays ``L``), ``bytes_moved`` at the TRANSFER
    granularity (``sector_bytes`` -- what pays ``G``). Collapsing them is a 4x GPU error.

    The max is continuous at the crossover (both terms equal ``messages*L/C*`` there), unlike the
    old regime switch, whose cliff at the BDP was a factor ``1 + BDP`` (~149x at the example DRAM:
    the serialized sum ``M*L + B*G`` against ``B*G`` with ``B = M*line``).

    Works on floats and sympy expressions alike; symbolic inputs yield a symbolic ``Max``.
    """
    g_term = bytes_moved * p.G
    l_term = messages * p.L / concurrency
    o_term = p.o * messages
    if isinstance(bytes_moved, sympy.Basic) or isinstance(messages, sympy.Basic):
        return sympy.Max(g_term, l_term) + o_term
    return max(g_term, l_term) + o_term


def bandwidth_delay_product(p: LogGP) -> float:
    """Messages that must be OUTSTANDING to saturate the channels: ``L / (line * G)`` (= ``L * BW /
    line``). Little's Law -- to keep a pipe of latency ``L`` and bandwidth ``1/G`` full you need this
    many block requests in flight at once. It is the threshold that separates the two regimes.

    Example (DRAM L=95ns, BW=100GB/s, 64B line): ~148 requests. A single CPU core tracks ~24
    outstanding misses, so one core CANNOT saturate DRAM -- it is latency-bound, which is exactly why
    the model assumes multiple cores. A GPU keeps thousands of accesses in flight, so it clears the
    threshold easily and is bandwidth-bound.
    """
    return p.L / (p.line_bytes * p.G)


def regime(p: LogGP, available_concurrency: float) -> str:
    """``"bandwidth"`` if the available outstanding-request concurrency reaches the bandwidth-delay
    product (latency is fully hidden, the channels are the limit), else ``"latency"``.

    ``available_concurrency`` is how many INDEPENDENT block requests the workload can keep in flight:
    the hardware's outstanding-miss budget for a parallel nest, or ~1 for a dependency-serialized loop
    (a pointer chase, a scan). Equivalent to asking which term of :func:`achievable_rate` binds --
    ``concurrency * line / L < 1/G`` is the latency-bound branch.
    """
    return "bandwidth" if available_concurrency >= bandwidth_delay_product(p) else "latency"


def memory_time(blocks_per_iter: float, total_iters: float, p: LogGP, concurrency: float = None) -> float:
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
    block messages.

    This is the coalesced quick path (``B = blocks * line``); the full nest analysis, with the
    request/transfer granularities split and the nest's own concurrency, is
    :func:`nest_memory_time` via ``logp_analysis.LoopNestLogP.total_time``.
    """
    total_bytes = blocks_per_iter * total_iters * p.line_bytes
    return total_bytes / achievable_rate(p, concurrency)  # default: p.core_mlp single-core view


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
      * the measured outstanding-miss knee must not EXCEED ``L/g`` -- the knee is
        ``min(L/g, miss-queue depth)``, so the issue cap is an upper bound on it; a knee above the
        cap is impossible and means a broken run. A knee far BELOW ``L/g`` is the normal case (the
        miss queue binds first) and is not an error.
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
    # One-sided: knee = min(L/g, miss-queue depth) <= L/g always; only knee > cap is impossible.
    if knee_concurrency > p.concurrency * 1.35:
        reasons.append(f"outstanding-miss knee {knee_concurrency:.1f} exceeds the issue cap "
                       f"L/g = {p.concurrency:.1f} -- impossible under min(), so the run is broken")
    if p.bw_saturated > peak_bytes_per_s:
        reasons.append(f"saturated bandwidth {p.bw_saturated / 1e9:.1f} GB/s exceeds hardware peak "
                       f"{peak_bytes_per_s / 1e9:.1f} GB/s (a cache-resident array)")
    return reasons


def _relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0
