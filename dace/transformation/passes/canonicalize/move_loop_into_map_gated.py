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

**GPU** ignores all of this and always interchanges: one outer parallel map
launches a single kernel whose threads each run the sequential loop in
registers, instead of the loop re-launching a fresh kernel every iteration --
the kernel-launch saving dominates (``tests/ab_perf`` interchange A/B).

The stride ranking reuses
:func:`~dace.transformation.passes.minimize_stride_permutation.score_indexed_strides`
-- the same scorer that orders map nests -- now applied across the loop<->map
boundary that the map-only and loop-only stride passes cannot cross.
"""
from typing import Any, Dict, Optional

from dace import SDFG, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.passes.minimize_stride_permutation import _to_float, score_indexed_strides


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
    body = loop.nodes()[0]
    map_entry = next(n for n in body.nodes() if isinstance(n, nodes.MapEntry))
    itervar = loop.loop_variable
    mparams = list(map_entry.map.params)
    subgraph = body.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    scores = score_indexed_strides(subgraph.edges(), sdfg, [itervar] + mparams)
    loop_cost = (_to_float(scores[itervar][0]), _to_float(scores[itervar][1]))
    map_cost = min((_to_float(scores[p][0]), _to_float(scores[p][1])) for p in mparams)
    return loop_cost < map_cost


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveLoopIntoMapGated(ppl.Pass):
    """Apply :class:`MoveLoopIntoMap` only where the cost model approves.

    See the module docstring for the rule. ``target='gpu'`` always interchanges
    an applicable loop; ``target='cpu'`` interchanges only when doing so lowers
    the innermost iterated stride.
    """

    CATEGORY: str = 'Optimization Preparation'

    target = properties.Property(dtype=str,
                                 default='cpu',
                                 choices=['cpu', 'gpu'],
                                 desc="Per-target interchange policy ('gpu' always; 'cpu' only when stride drops).")

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
        applied = 0
        # Re-scan after each apply: MoveLoopIntoMap rewrites the CFG (removes the
        # loop, nests a new one), invalidating the iterator.
        changed = True
        while changed:
            changed = False
            for loop in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
                if not MoveLoopIntoMap.can_be_applied_to(loop.sdfg, loop=loop):
                    continue
                if self.target != 'gpu' and not interchange_lowers_stride(loop, loop.sdfg):
                    continue
                MoveLoopIntoMap.apply_to(loop.sdfg, loop=loop, verify=False)
                applied += 1
                changed = True
                break
        return applied or None
