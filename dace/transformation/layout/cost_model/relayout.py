# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What a layout change costs, and when it pays for itself: insert iff uses * (t_nest(L0) - t_nest(L1)) > t_relayout."""
import math
from typing import Optional

import sympy as sp

from dace import data
from dace.transformation.layout.cost_model.loggp import LogGP, achievable_rate, nest_memory_time


def array_bytes(desc: data.Array) -> sp.Basic:
    """Bytes the descriptor occupies: ``total_size * itemsize``."""
    return sp.sympify(desc.total_size) * desc.dtype.bytes


def streaming_relayout_time(desc: data.Array, p: LogGP) -> sp.Basic:
    """Time for a relayout moving whole blocks on both sides: 2 * S at the saturated rate.
    Lower bound (TILED transpose); the default pure expansion does not achieve it."""
    return 2 * array_bytes(desc) / achievable_rate(p, float("inf"))


def relayout_time(traffic_bytes: sp.Basic, p: LogGP) -> sp.Basic:
    """Time to move traffic_bytes at the sustained rate; takes measured traffic rather than assuming it streams."""
    return sp.sympify(traffic_bytes) / achievable_rate(p, float("inf"))  # saturated scope


def block_traffic(blocks_touched: sp.Basic, p: LogGP, written: bool, covers_full_block: bool) -> sp.Basic:
    """Bytes crossing the channels for blocks_touched transfer blocks.
    A write moves a block twice unless it covers the whole block (read-for-ownership vs streaming store)."""
    per_block = 2 * p.sector_bytes if (written and not covers_full_block) else p.sector_bytes
    return sp.sympify(blocks_touched) * per_block


def cache_efficiency(useful_bytes: sp.Basic, traffic_bytes: sp.Basic) -> sp.Basic:
    """useful / transferred: a derived diagnostic, never a model input. Only fixes the G term, not L."""
    return sp.sympify(useful_bytes) / sp.sympify(traffic_bytes)


def max_layout_delta(desc: data.Array, p: LogGP, written: bool = False) -> sp.Basic:
    """Max saving any layout change can buy on one full pass over desc: worst layout (one element per block) vs best (contiguous)."""
    per_block = p.sector_bytes / desc.dtype.bytes
    worst = (2 * per_block - 1) if written else (per_block - 1)
    return array_bytes(desc) * worst / achievable_rate(p, float("inf"))  # saturated scope


def break_even_uses(t_nest_before: sp.Basic, t_nest_after: sp.Basic, t_relayout: sp.Basic) -> Optional[int]:
    """Uses needed before the relayout pays for itself; None if the new layout is not faster."""
    delta = sp.simplify(sp.sympify(t_nest_before) - sp.sympify(t_nest_after))
    if not delta.is_number:
        raise ValueError(f"break_even_uses needs concrete times; got a symbolic delta {delta}. "
                         "Substitute the nest's symbols first.")
    if float(delta) <= 0.0:
        return None
    return int(math.ceil(float(sp.sympify(t_relayout)) / float(delta)))


def relayout_pays(t_nest_before: sp.Basic, t_nest_after: sp.Basic, t_relayout: sp.Basic, uses: int = 1) -> bool:
    """Whether inserting the layout change is a win over ``uses`` consumer nests."""
    n = break_even_uses(t_nest_before, t_nest_after, t_relayout)
    return n is not None and uses >= n


# Combining cache efficiency with LogP: efficiency enters only the G (bandwidth) term, never L
# (latency is paid per message, not per byte).


def bandwidth_efficiency(useful_bytes: sp.Basic, sectors_touched: sp.Basic, p: LogGP, written: bool = False,
                         covers_full_block: bool = False) -> sp.Basic:
    """epsilon in (0, 1]: fraction of moved bytes the computation actually uses."""
    return sp.sympify(useful_bytes) / block_traffic(sectors_touched, p, written, covers_full_block)


def nest_time_by_efficiency(useful_bytes: sp.Basic, epsilon: sp.Basic, messages: sp.Basic, p: LogGP,
                            concurrency: float) -> sp.Basic:
    """nest_memory_time in efficiency form: useful_bytes / epsilon is the bytes moved."""
    return nest_memory_time(p, sp.sympify(useful_bytes) / sp.sympify(epsilon), sp.sympify(messages), concurrency)


def relayout_pays_by_efficiency(eps_before: float, eps_after: float, passes: int = 1,
                                overhead_passes: float = 0.0) -> bool:
    """Whether a relayout pays from efficiencies alone: passes * (1/eps_before - 1/eps_after) >= 2 + overhead_passes.
    Bandwidth regime only; use nest_time / relayout_pays for a latency-bound nest."""
    if not (0 < eps_before <= 1) or not (0 < eps_after <= 1):
        raise ValueError(f"efficiencies must lie in (0, 1]; got before={eps_before}, after={eps_after}")
    if overhead_passes < 0:
        raise ValueError(f"overhead_passes must be >= 0; got {overhead_passes}")
    # >=, not >: must agree with break_even_passes' ceil() at exact equality
    return passes * (1.0 / eps_before - 1.0 / eps_after) >= 2.0 + overhead_passes


def break_even_passes(eps_before: float, eps_after: float, overhead_passes: float = 0.0) -> Optional[int]:
    """Passes over the array before a relayout pays for itself; None if the new layout is not better."""
    if not (0 < eps_before <= 1) or not (0 < eps_after <= 1):
        raise ValueError(f"efficiencies must lie in (0, 1]; got before={eps_before}, after={eps_after}")
    gain = 1.0 / eps_before - 1.0 / eps_after
    if gain <= 0:
        return None
    return int(math.ceil((2.0 + overhead_passes) / gain))


def single_pass_efficiency_threshold(eps_after: float = 1.0) -> float:
    """Efficiency below which a single pass already redeems a relayout to eps_after: 1 / (2 + 1/eps_after)."""
    if not 0 < eps_after <= 1:
        raise ValueError(f"eps_after must lie in (0, 1]; got {eps_after}")
    return 1.0 / (2.0 + 1.0 / eps_after)
