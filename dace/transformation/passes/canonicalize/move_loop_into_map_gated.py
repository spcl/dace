# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Target-gated loop<->map interchange (``for(seq) { map }`` -> ``map { for(seq) }``)."""

import math
from typing import Any

from dace import SDFG, properties, symbolic
from dace.sdfg import propagation
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap, lane_maps
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.minimize_stride_permutation import score_indexed_strides, score_to_float
from dace.transformation.passes.parallelization_prep import DEFAULT_UNROLL_LIMIT, _constant_trip_count


def stride_costs(loop: LoopRegion, sdfg: SDFG) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Contiguity scores ``(min_home_stride, total_home_stride)`` of the loop variable and of the best parameter of
    the maps in ``loop``'s body (smaller is more contiguous), or ``None`` without a map."""
    lanes = lane_maps(loop)
    if not lanes:
        return None
    itervar = loop.loop_variable
    mparams = [p for _, entry in lanes for p in entry.map.params]
    edges = [
        e for state, entry in lanes for e in state.scope_subgraph(entry, include_entry=True, include_exit=True).edges()
    ]
    scores = score_indexed_strides(edges, sdfg, [itervar] + mparams)
    loop_cost = (score_to_float(scores[itervar][0]), score_to_float(scores[itervar][1]))
    map_cost = min((score_to_float(scores[p][0]), score_to_float(scores[p][1])) for p in mparams)
    return loop_cost, map_cost


def interchange_lowers_stride(loop: LoopRegion, sdfg: SDFG) -> bool:
    """True if moving ``loop`` into its inner map lowers the innermost stride."""
    loop_cost, map_cost = stride_costs(loop, sdfg)
    return loop_cost < map_cost


def short_constant_trip(loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``loop`` runs a constant number of trips no larger than the short-loop unroll limit."""
    trips = _constant_trip_count(loop, sdfg)
    return trips is not None and trips <= DEFAULT_UNROLL_LIMIT


def launches_saved(loop: LoopRegion) -> float:
    """Kernel launches the GPU interchange saves, ``trips * maps - 1``; ``inf`` for a symbolic trip count."""
    maps = len(lane_maps(loop))
    start, end, step = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    if maps == 0 or start is None or end is None or step is None:
        return math.inf if maps else 0
    trips = symbolic.simplify((end - start) / step + 1)
    if not as_basic(trips).is_Number:
        return math.inf
    return max(int(as_expr(trips)), 0) * maps - 1


def single_iteration_lanes(loop: LoopRegion) -> bool:
    """Whether ``loop``'s body has maps and every one of them runs exactly one iteration."""
    lanes = lane_maps(loop)
    return bool(lanes) and all(symbolic.simplify(entry.map.range.num_elements()) == 1 for _, entry in lanes)


def interchange_pays_on_gpu(loop: LoopRegion, sdfg: SDFG) -> bool:
    """The GPU rule of the module docstring: launches saved, unless the loop axis is the contiguous one."""
    costs = stride_costs(loop, sdfg)
    if costs is None or launches_saved(loop) <= 0:
        return False
    if single_iteration_lanes(loop):
        return True
    # SMALL lanes are not a thread axis: the loop would run inside a handful of threads (velocity_tendencies'
    # level loop around maps over its one block, with the contiguous plane axis nested and serial in each).
    facts = sdfg.facts()
    extents = [end - begin for _, entry in lane_maps(loop) for begin, end, _ in entry.map.range]
    if extents and all(symbolic.is_small(extent, facts) for extent in extents):
        return False
    return not costs[0] < costs[1]


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopIntoMapGated(ppl.Pass):
    """Apply :class:`MoveLoopIntoMap` only where the cost model approves."""

    CATEGORY: str = "Optimization Preparation"

    target = properties.Property(
        dtype=str,
        default="cpu",
        choices=["cpu", "gpu"],
        desc="Per-target interchange policy ('gpu' when launches drop; 'cpu' when stride drops).",
    )

    single_iteration_only = properties.Property(
        dtype=bool,
        default=False,
        desc="Only consider loops whose maps all run one iteration (the offload wraps host-side work in "
        "such maps; the pass then runs on the offloaded graph)",
    )

    def __init__(self, target: str = "cpu", single_iteration_only: bool = False) -> None:
        super().__init__()
        self.target = target
        self.single_iteration_only = single_iteration_only

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def try_interchange(self, loop: LoopRegion, gpu: bool) -> bool:
        """Interchange ``loop`` with its map when the policy approves; whether it did."""
        if self.single_iteration_only and not single_iteration_lanes(loop):
            return False
        if gpu and not interchange_pays_on_gpu(loop, loop.sdfg):
            return False
        # A short constant loop moves in whole on CPU too: its body's maps of one range (velocity's gather and
        # accumulation per neighbour) become one map whose unrolled loop keeps the accumulator in a register.
        short = not gpu and short_constant_trip(loop, loop.sdfg)
        options = {"cfg_body": gpu or short}
        if not MoveLoopIntoMap.can_be_applied_to(loop.sdfg, options=options, loop=loop):
            return False
        if not gpu and not (short or interchange_lowers_stride(loop, loop.sdfg)):
            return False
        MoveLoopIntoMap.apply_to(loop.sdfg, options=options, loop=loop, verify=False, annotate=False)
        return True

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Interchange every approved loop<->map pair in ``sdfg``."""
        gpu = self.target == "gpu"
        applied = 0

        # Sweep until a sweep applies nothing: an apply removes its loop and nests a new one, which the next sweep sees.
        def live_loops() -> list[LoopRegion]:
            return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]

        refused: set[LoopRegion] = set()
        changed = True
        while changed:
            changed = False
            stale: set[LoopRegion] = set()
            live: set[LoopRegion] | None = None
            for loop in live_loops():
                if loop in refused or loop in stale or (live is not None and loop not in live):
                    continue
                around = loop_analysis.enclosing_loops(loop)
                if not self.try_interchange(loop, gpu):
                    refused.add(loop)
                    continue
                applied += 1
                changed = True
                refused -= around
                stale |= around
                live = set(live_loops())
            if changed:
                propagation.propagate_memlets_sdfg(sdfg)
        return applied or None
