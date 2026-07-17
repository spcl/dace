# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What a layout change COSTS, and when it pays for itself.

A layout change is not free: it reads the array once and writes it once. Deciding whether to insert
one -- after a map, or on the edge between two nests that disagree about the layout they want -- is a
break-even question, and the thing it breaks even against is NOT the consumer nest's absolute cost.
It is the DELTA the new layout buys, times the number of nests that get to use it:

    insert iff   uses * (t_nest(L0) - t_nest(L1))  >  t_relayout  [+ t_relayout_back if L0 must be restored]

Why a relayout can pay on a SINGLE use
--------------------------------------
For an array of ``S`` bytes with ``line/elem`` elements per cache line (8 for fp64 on 64-byte lines),
one full pass moves at most ``S * line/elem`` bytes in the worst layout (a new line per element) and
``S`` bytes in the best. So the most any layout change can save on one pass is::

    delta_max = S * (line/elem - 1) / rate          # 7*S/rate for fp64

while a STREAMING relayout costs ``2*S/rate`` (read S, write S). A single use therefore pays off once
the layout recovers more than ``2 / (line/elem - 1)`` -- about 29% for fp64 -- of the maximum gap.
Mid-program layout change is not "never good"; it is good whenever the layout gap is wide.

Do not assume the relayout streams
----------------------------------
``2*S/rate`` holds only for a TILED transpose (``HPTT``, ``cuTENSOR``), which moves whole lines on
both sides. ``LayoutChange``'s default ``pure`` expansion is a single flat mapped-tasklet copy, so
one side is strided and touches a line per element -- up to ``(1 + line/elem) * S`` of traffic, ~4x
the streaming cost. That is why :func:`relayout_time` takes the traffic as an argument instead of
assuming it, and why the honest way to cost a ``pure`` relayout is to expand it and hand the copy
nest to :func:`~dace.transformation.layout.cost_model.logp_analysis.analyze_loop_nest` like any other
nest -- a relayout IS a loop nest. Then "does it stream?" is an OUTPUT of the model, not an input.
"""
import math
from typing import Optional

import sympy as sp

from dace import data
from dace.transformation.layout.cost_model.loggp import LogGP, achievable_rate, nest_memory_time


def array_bytes(desc: data.Array) -> sp.Basic:
    """Bytes the descriptor occupies: ``total_size * itemsize``."""
    return sp.sympify(desc.total_size) * desc.dtype.bytes


def streaming_relayout_time(desc: data.Array, p: LogGP) -> sp.Basic:
    """Time for a relayout that moves whole blocks on BOTH sides: read every byte, write every byte.

    ``2 * S``, and note what it is NOT paying: the read side is contiguous (one transfer per block,
    no waste) and the write side covers whole blocks (streaming stores, no read-for-ownership). A
    relayout is the BEST case on both sides simultaneously -- which is exactly why it can beat a
    single badly-laid-out pass that pays ``2 * r`` per block.

    This is the TILED-transpose cost (``HPTT``/``cuTENSOR``), and a LOWER BOUND for any relayout.
    The default ``pure`` expansion does not achieve it -- see the module docstring.

    Rates here are the SATURATED channel rate (``concurrency=inf``): a relayout is a streaming copy
    over all cores, the bandwidth-regime scope this module's break-evens are stated in.
    """
    return 2 * array_bytes(desc) / achievable_rate(p, float("inf"))


def relayout_time(traffic_bytes: sp.Basic, p: LogGP) -> sp.Basic:
    """Time to move ``traffic_bytes`` at the sustained rate.

    Takes the traffic rather than deriving it, so a caller who measured it (via ``blocks_touched``
    on the relayout's own copy nest) is not overridden by an optimistic streaming assumption.
    """
    return sp.sympify(traffic_bytes) / achievable_rate(p, float("inf"))  # saturated scope


def block_traffic(blocks_touched: sp.Basic, p: LogGP, written: bool, covers_full_block: bool) -> sp.Basic:
    """Bytes crossing the channels for a pass that touches ``blocks_touched`` transfer blocks.

    A READ of a block moves it once. A WRITE moves it TWICE when the pass does not cover the whole
    block -- the hardware must fetch the block to merge the partial update (read-for-ownership /
    write-allocate) and then write it back. A write that DOES cover its block can skip the fetch
    (a streaming / non-temporal store), so it moves once.

    This is why layout matters MORE for written arrays than for read arrays: a bad layout does not
    merely waste a fetch, it forces a fetch that a good layout would not need at all.
    """
    per_block = 2 * p.sector_bytes if (written and not covers_full_block) else p.sector_bytes
    return sp.sympify(blocks_touched) * per_block


def cache_efficiency(useful_bytes: sp.Basic, traffic_bytes: sp.Basic) -> sp.Basic:
    """``useful / transferred`` -- a DERIVED diagnostic, never a model input.

    Do not "simplify" the model to an elementwise cost multiplied by this ratio. Efficiency is a
    BYTES ratio, so it repairs the ``G`` term and silently corrupts the ``L`` term: latency is paid
    per MESSAGE (a distinct block touched), not per element scaled by a ratio. Collapsing the two
    loses :func:`~dace.transformation.layout.cost_model.loggp.regime` -- i.e. the ability to say
    whether a nest is latency- or bandwidth-bound, which is the reason to use LogP at all.
    """
    return sp.sympify(useful_bytes) / sp.sympify(traffic_bytes)


def max_layout_delta(desc: data.Array, p: LogGP, written: bool = False) -> sp.Basic:
    """The most any layout change can save on ONE full pass over ``desc``.

    Worst layout: one element per block. Best: fully contiguous.
      * read-only  -- worst ``S * sector/elem``, best ``S``            -> gap ``S * (r - 1)``
      * written    -- worst ``2 * S * sector/elem`` (every partial write fetches first), best ``S``
                      (full-block streaming stores)                    -> gap ``S * (2r - 1)``
    For fp64 / 64-byte blocks that is 7*S read-only but 15*S written, against a relayout costing
    only 2*S -- so an intermediate relayout of a WRITTEN array pays back after recovering ~13% of
    its gap, versus ~29% for a read-only one.
    """
    per_block = p.sector_bytes / desc.dtype.bytes
    worst = (2 * per_block - 1) if written else (per_block - 1)
    return array_bytes(desc) * worst / achievable_rate(p, float("inf"))  # saturated scope


def break_even_uses(t_nest_before: sp.Basic, t_nest_after: sp.Basic, t_relayout: sp.Basic) -> Optional[int]:
    """How many uses of the new layout are needed before the relayout pays for itself.

    ``None`` when the new layout is not faster -- no number of uses redeems it.
    """
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


# --------------------------------------------------------------------------------------------- #
#  Combining cache efficiency with LogP
# --------------------------------------------------------------------------------------------- #
#  Factor each array's cost into a layout-INDEPENDENT part (the algorithm) and a layout-DEPENDENT
#  part (the layout), then let efficiency enter EXACTLY ONE LogP term:
#
#      useful_bytes  = elements_touched * itemsize        -- algorithm; no layout changes it
#      epsilon       = useful_bytes / traffic_bytes       -- layout; the only thing layout moves
#      messages      = distinct request-blocks touched     -- layout
#
#      G-term (bandwidth) = useful_bytes / epsilon * G     <- efficiency lives HERE, and only here
#      L-term (latency)   = messages * L / concurrency     <- efficiency CANNOT express this
#      t_nest             = max(G-term, L-term)            <- regime() picks the binding one
#
#  Efficiency provably underdetermines the latency term: on a sectored GPU, four sectors inside ONE
#  128-byte line and four sectors spread over FOUR lines move the same bytes (same epsilon) but cost
#  1 vs 4 messages. Same efficiency, different latency. That is why the model keeps both and why
#  "elementwise LogP times a cache-efficiency factor" is not a simplification but a loss.


def bandwidth_efficiency(useful_bytes: sp.Basic, sectors_touched: sp.Basic, p: LogGP, written: bool = False,
                         covers_full_block: bool = False) -> sp.Basic:
    """``epsilon`` in (0, 1]: the fraction of moved bytes the computation actually uses.

    Decomposes as ``epsilon_spatial * epsilon_write``: how much of each fetched sector is used, times
    ``1/2`` when a partial write forces a read-for-ownership. So the write penalty is not a special
    case bolted on -- it is just another efficiency factor. For fp64 on 64-byte blocks, ``epsilon``
    ranges over ``[1/16, 1]``: 1/16 for a one-element-per-block partial write, 1 for a perfect
    contiguous pass.
    """
    return sp.sympify(useful_bytes) / block_traffic(sectors_touched, p, written, covers_full_block)


def nest_time_by_efficiency(useful_bytes: sp.Basic, epsilon: sp.Basic, messages: sp.Basic, p: LogGP,
                            concurrency: float) -> sp.Basic:
    """The nest formula in efficiency form: ``useful_bytes / epsilon`` IS the bytes moved, so this
    delegates straight to :func:`~dace.transformation.layout.cost_model.loggp.nest_memory_time` --
    the ONE formula (``max(bytes*G, messages*L/C)``); no second derivation lives here. The open
    disagreement this function used to carry (``total_time`` returning the serialized branch in the
    latency regime) is resolved: the MLP sweep refuted the serialized branch (~8x) and
    ``LoopNestLogP.total_time`` now routes through the same ``nest_memory_time``."""
    return nest_memory_time(p, sp.sympify(useful_bytes) / sp.sympify(epsilon), sp.sympify(messages), concurrency)


def relayout_pays_by_efficiency(eps_before: float, eps_after: float, passes: int = 1,
                                overhead_passes: float = 0.0) -> bool:
    """Whether a relayout pays, from EFFICIENCIES ALONE -- no bandwidth, no latency, no hardware.

    In the bandwidth regime the break-even is ``passes * useful * (1/eps0 - 1/eps1) > 2 * S``, and
    for a full pass (``useful == S``) the array size and ``G`` both CANCEL:

        passes * (1/eps_before - 1/eps_after) >= 2 + overhead_passes

    So "is this relayout worth it?" is a pure traffic ratio -- a faster machine does not change the
    answer, because it speeds the relayout and the nest by the same factor. The ``2`` is the
    relayout itself: one clean read pass plus one clean write pass.

    ``overhead_passes`` is any extra traffic the relayout needs, in units of one array pass, and it
    is what separates the two ways of removing an indirection ``A[sigma[i]]``:

    * STATIC REPLACE (``overhead_passes = 0``) -- ``sigma`` is known without running the program, so
      the permutation is a compile-time fact. Reorder once into ``A'`` and rewrite consumers to read
      ``A'[i]`` directly. Nothing is inspected at runtime; if the reorder happens offline the cost
      is not even 2.
    * INSPECTOR-EXECUTOR (``overhead_passes > 0``) -- ``sigma`` is only known at runtime, so an
      inspector must read the index array and sort/bucket it before the executor can use the
      reordered data. That inspection is real traffic, and it must be re-paid whenever ``sigma``
      changes.

    Static replace is therefore the ``overhead_passes -> 0`` limit of inspector-executor, and is
    strictly better whenever it applies. Only valid in the bandwidth regime; a latency-bound nest is
    not paying for bytes, so use :func:`nest_time` and :func:`relayout_pays` there.
    """
    if not (0 < eps_before <= 1) or not (0 < eps_after <= 1):
        raise ValueError(f"efficiencies must lie in (0, 1]; got before={eps_before}, after={eps_after}")
    if overhead_passes < 0:
        raise ValueError(f"overhead_passes must be >= 0; got {overhead_passes}")
    # >=, not >: at exact break-even the relayout HAS paid for itself, and this is the same
    # convention as break_even_passes' ceil() -- the two must not disagree at equality.
    return passes * (1.0 / eps_before - 1.0 / eps_after) >= 2.0 + overhead_passes


def break_even_passes(eps_before: float, eps_after: float, overhead_passes: float = 0.0) -> Optional[int]:
    """Passes over the array before a relayout (or an inspector-executor reorder) has paid for itself.

    ``None`` if the new layout is not better -- no amount of reuse redeems it. This is the number the
    inspector-executor literature amortizes: an unstructured code reorders ONCE and then reuses the
    ordering across many timesteps, so the question is never "is the inspector cheap" but "how many
    executor passes follow it".
    """
    if not (0 < eps_before <= 1) or not (0 < eps_after <= 1):
        raise ValueError(f"efficiencies must lie in (0, 1]; got before={eps_before}, after={eps_after}")
    gain = 1.0 / eps_before - 1.0 / eps_after
    if gain <= 0:
        return None
    return int(math.ceil((2.0 + overhead_passes) / gain))


def single_pass_efficiency_threshold(eps_after: float = 1.0) -> float:
    """The efficiency BELOW which one single pass already redeems a relayout to ``eps_after``.

    Solving ``1/eps0 - 1/eps1 >= 2`` gives ``eps0 <= 1 / (2 + 1/eps1)``; against a perfect target
    layout that is **1/3**. Read it as: a nest using less than a third of every block it moves is
    already worth relaying out FOR THAT NEST ALONE -- no reuse argument required. This is the
    quantitative refutation of "a mid-program relayout is never good".
    """
    if not 0 < eps_after <= 1:
        raise ValueError(f"eps_after must lie in (0, 1]; got {eps_after}")
    return 1.0 / (2.0 + 1.0 / eps_after)
