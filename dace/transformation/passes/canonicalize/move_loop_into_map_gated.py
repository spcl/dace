# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Target-gated loop<->map interchange (``for(seq) { map }`` -> ``map { for(seq) }``).

:class:`~dace.transformation.interstate.move_loop_into_map.MoveLoopIntoMap`
moves a sequential loop into the parallel map it wraps, making the map the
outer (parallel) axis and the loop the inner sequential axis. The bare
transformation applies whenever it structurally can; this pass adds the cost
model the interchange needs to be a net win.

The rule
--------
**Contiguity is a property of the INNERMOST iterated axis alone.** Hoisting the
map out of the loop makes the LOOP variable innermost, so the interchange is
worth taking only when the loop variable is the more contiguous of the two
(:func:`interchange_lowers_stride`). Then it both lowers the innermost stride
and turns the loop's per-trip fork/join of the parallel region into one region
for the whole nest.

When the map's parallel axis is the more contiguous one, the interchange is
DECLINED. Hoisting it would put a strided loop variable innermost -- one cache
line and one TLB entry per element, and nothing left for the vectorizer -- and
outer parallelism does not pay for that. Parallelism is not lost by declining:
it stays on the map, which is now the innermost axis, and the emitter's own
region hoisting
(:class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`)
lifts the parallel region above the sequential loop, so the fork/join is paid
once for the nest either way. What the two orders do NOT share is the innermost
stride, and that is what this rule decides.

This is the ``for(seq) { map }`` recurrence-sweep shape (TSVC ``s231`` /
``s235`` / ``s233``: ``aa[j, i] = aa[j-1, i] + ...`` with ``i`` the contiguous
parallel column and ``j`` the sequential carry). Declining yields::

    #pragma omp parallel                 <- hoisted region, one fork/join
    for (j = 1; j < N; ++j)              <- the carry, sequential, outer
        #pragma omp for
        for (i = 0; i < N; ++i)          <- contiguous, vectorizable
            aa[j][i] = aa[j - 1][i] + ...

against the hoisted form's ``j`` innermost at ``stride = N``.

An earlier revision took the opposite view, hoisting whenever the map's axis
was unit-stride on the grounds that the vectorizer would recover the contiguity.
It does not: once hoisted the axis is no longer innermost, so nothing
strip-mines it into lanes. That rule is gone.

**GPU** decides by fork/join count (:func:`interchange_pays_on_gpu`), for every loop shape --
a single-map body and, through ``MoveLoopIntoMap(cfg_body=True)``, a body of control flow
over several maps of one range. Every top-level map is one kernel launch: a loop of ``K``
trips over ``M`` maps pays ``K * M`` launches, the interchanged ``map { for }`` pays one.
Threads keep the same lane axis and the same total work, so work and parallelism cancel and
the launches decide: interchange when ``K * M > 1``, with every symbol taken as the same
large size (a numeric trip count is exact, a symbolic one is large). Coalescing vetoes: when
the LOOP axis is strictly more contiguous than every map axis, the unit-stride access belongs
on the threads, and moving the loop inside would fix it as a per-thread serial walk over
strided threads.

The stride ranking reuses
:func:`~dace.transformation.passes.minimize_stride_permutation.score_indexed_strides`
-- the same scorer that orders map nests -- now applied across the loop<->map
boundary that the map-only and loop-only stride passes cannot cross.
"""
import math
from typing import Any, Dict, Optional

from dace import SDFG, properties, symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap, lane_maps
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.minimize_stride_permutation import _to_float, score_indexed_strides


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
    loop_cost = (_to_float(scores[itervar][0]), _to_float(scores[itervar][1]))
    map_cost = min((_to_float(scores[p][0]), _to_float(scores[p][1])) for p in mparams)
    return loop_cost, map_cost


def interchange_lowers_stride(loop: LoopRegion, sdfg: SDFG) -> bool:
    """True if moving ``loop`` into its inner map lowers the innermost stride.

    The post-interchange innermost iterated axis is the loop variable; the
    current innermost is the map's smallest-stride parameter. Interchanging is
    worthwhile (on CPU) only when the loop variable's contiguity score
    ``(min_home_stride, total_home_stride)`` is strictly better (smaller) than
    that of every map parameter, i.e. the loop reads more contiguous memory and
    belongs in the inner sequential position.

    :param loop: A loop whose single-state body holds exactly one map (already
                 validated by ``MoveLoopIntoMap.can_be_applied``).
    :param sdfg: The owning SDFG (for array strides).
    :returns: True if the interchange decreases the innermost stride.
    """
    loop_cost, map_cost = stride_costs(loop, sdfg)
    return loop_cost < map_cost


def launches_saved(loop: LoopRegion) -> float:
    """Kernel launches the GPU interchange saves, ``trips * maps - 1``; ``inf`` for a symbolic trip count.

    Every map in the body counts, branches included (an upper bound on a trip's launches).
    """
    maps = len(lane_maps(loop))
    start, end, step = (loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop),
                        loop_analysis.get_loop_stride(loop))
    if maps == 0 or start is None or end is None or step is None:
        return math.inf if maps else 0
    trips = symbolic.simplify((end - start) / step + 1)
    if not trips.is_Number:
        return math.inf
    return max(int(trips), 0) * maps - 1


def interchange_pays_on_gpu(loop: LoopRegion, sdfg: SDFG) -> bool:
    """The GPU rule of the module docstring: launches saved, unless the loop axis is the contiguous one."""
    costs = stride_costs(loop, sdfg)
    return costs is not None and launches_saved(loop) > 0 and not costs[0] < costs[1]


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopIntoMapGated(ppl.Pass):
    """Apply :class:`MoveLoopIntoMap` only where the cost model approves.

    See the module docstring for the rules. ``target='gpu'`` interchanges any legal loop, control-flow
    bodies included, that saves kernel launches without giving up a contiguous thread axis;
    ``target='cpu'`` interchanges a single-map body only when doing so lowers the innermost stride.
    """

    CATEGORY: str = 'Optimization Preparation'

    target = properties.Property(
        dtype=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        desc="Per-target interchange policy ('gpu' when launches drop; 'cpu' when stride drops).")

    def __init__(self, target: str = 'cpu'):
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Interchange every approved loop<->map pair in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :returns: The number of interchanges applied, or ``None`` if none.
        """
        gpu = self.target == 'gpu'
        options = {'cfg_body': gpu}
        applied = 0
        # Re-scan after each apply: MoveLoopIntoMap rewrites the CFG (removes the
        # loop, nests a new one), invalidating the iterator.
        changed = True
        while changed:
            changed = False
            for loop in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
                if gpu and not interchange_pays_on_gpu(loop, loop.sdfg):
                    continue
                if not MoveLoopIntoMap.can_be_applied_to(loop.sdfg, options=options, loop=loop):
                    continue
                if not gpu and not interchange_lowers_stride(loop, loop.sdfg):
                    continue
                MoveLoopIntoMap.apply_to(loop.sdfg, options=options, loop=loop, verify=False)
                applied += 1
                changed = True
                break
        return applied or None
