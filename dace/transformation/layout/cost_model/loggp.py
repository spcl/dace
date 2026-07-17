# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP/LogGP parameters (L, o, g, G) of a memory level, and the fit extracting them from microbenchmarks."""
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
    #: TRANSFER granularity crossing the channels (the G term). x86 == line_bytes; NVIDIA sectors 32B within a 128B line.
    sector_bytes: int = None  # defaults to line_bytes (no sectoring)

    #: measured outstanding-miss knee of one core; None falls back to L/g. validate() cross-checks them.
    c_core: float = None

    def __post_init__(self):
        if self.sector_bytes is None:
            object.__setattr__(self, "sector_bytes", self.line_bytes)

    @property
    def concurrency(self) -> float:
        """LogP's L/g: requests one core keeps in flight."""
        return self.L / self.g

    @property
    def core_mlp(self) -> float:
        """Demand-miss budget of one core: measured knee when available, else L/g."""
        return self.c_core if self.c_core is not None else self.concurrency

    @property
    def core_stream_mlp(self) -> float:
        """Effective concurrency on a prefetch-friendly pattern, by Little's Law: bw_core * L / line_bytes."""
        return self.bw_core * self.L / self.line_bytes


def gap_from_bandwidth(bytes_per_s: float) -> float:
    """Per-byte gap ``G = 1 / bandwidth``."""
    return 1.0 / bytes_per_s


def lines_touched(nbytes: int, line_bytes: int) -> int:
    """Number of whole lines a contiguous nbytes request moves."""
    if nbytes <= 0:
        return 0
    return (nbytes + line_bytes - 1) // line_bytes


def message_time(p: LogGP, nbytes: int) -> float:
    """Time of one contiguous nbytes request: o + L + (lines - 1) * line_bytes * G."""
    lines = lines_touched(nbytes, p.line_bytes)
    if lines == 0:
        return 0.0
    return p.o + p.L + (lines - 1) * p.line_bytes * p.G


def achievable_rate(p: LogGP, concurrency: float = None) -> float:
    """Sustained bytes/s at concurrency outstanding requests: min(1/G, concurrency*line/L).
    Valid only when line-coupled (B = M*line); else use nest_memory_time."""
    if concurrency is None:
        concurrency = p.core_mlp
    latency_bound = concurrency * p.line_bytes / p.L
    bandwidth_bound = 1.0 / p.G
    return min(bandwidth_bound, latency_bound)


def nest_memory_time(p: LogGP, bytes_moved, messages, concurrency):
    """The nest formula every total-time path routes through: T = max(bytes_moved*G, messages*L/concurrency) + o*messages.
    messages must be at REQUEST granularity (line_bytes), bytes_moved at TRANSFER granularity (sector_bytes); collapsing them is a 4x GPU error."""
    g_term = bytes_moved * p.G
    l_term = messages * p.L / concurrency
    o_term = p.o * messages
    if isinstance(bytes_moved, sympy.Basic) or isinstance(messages, sympy.Basic):
        return sympy.Max(g_term, l_term) + o_term
    return max(g_term, l_term) + o_term


def bandwidth_delay_product(p: LogGP) -> float:
    """Messages outstanding to saturate the channels: L / (line * G). Little's Law threshold separating the two regimes."""
    return p.L / (p.line_bytes * p.G)


def regime(p: LogGP, available_concurrency: float) -> str:
    """"bandwidth" if available_concurrency reaches the bandwidth-delay product, else "latency"."""
    return "bandwidth" if available_concurrency >= bandwidth_delay_product(p) else "latency"


def memory_time(blocks_per_iter: float, total_iters: float, p: LogGP, concurrency: float = None) -> float:
    """Predicted memory time: blocks_per_iter * total_iters * line_bytes / achievable_rate.
    Coalesced quick path only (B = blocks * line); see nest_memory_time for the general form."""
    total_bytes = blocks_per_iter * total_iters * p.line_bytes
    return total_bytes / achievable_rate(p, concurrency)  # default: p.core_mlp single-core view


@dataclass(frozen=True)
class Fit:
    """A least-squares ``T(n) = alpha + beta * n`` over the message-size sweep."""
    alpha: float  # s -- intercept; identifies with L
    beta: float  # s/byte -- slope; identifies with the single-core per-byte gap
    residual: float  # relative RMS residual, so a bad fit cannot be silently trusted


def fit_message_size(sizes: Sequence[int], times: Sequence[float]) -> Fit:
    """Fit T(n) = alpha + beta * n over the message-size sweep at one memory level, weighted by 1/T since timing error is multiplicative."""
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
    """Reasons to reject the parametrization; empty means accept. Cross-checks each parameter two independent ways."""
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
    # one-sided: knee <= L/g always; only knee > cap is impossible
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
