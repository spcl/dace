# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Target-gated loop<->map interchange (``for(seq) { map }`` -> ``map { for(seq) }``).

:class:`~dace.transformation.interstate.move_loop_into_map.MoveLoopIntoMap`
moves a sequential loop into the parallel map it wraps, making the map the
outer (parallel) axis and the loop the inner sequential axis. The bare
transformation applies whenever it structurally can; this pass adds the cost
model the interchange needs to be a net win:

* **GPU**: always interchange. One outer parallel map launches a single kernel
  whose threads each run the sequential loop in registers, instead of the loop
  re-launching a fresh kernel every iteration -- the kernel-launch saving
  dominates (``tests/ab_perf`` interchange A/B).
* **CPU**: interchange when EITHER

  1. it lowers the innermost iterated stride (the loop variable indexes
     more-contiguous memory than the map's innermost parameter, so the loop
     belongs in the inner sequential position), OR
  2. the map's parallel axis is itself unit-stride
     (:func:`_map_axis_is_unit_stride`). Hoisting a unit-stride parallel map out
     of the sequential loop keeps its access contiguous once vectorized (the
     vectorizer strip-mines the unit-stride axis into SIMD lanes) while replacing
     the loop's per-iteration fork/join of the parallel region with a single
     outer parallel region. This is the ``for(seq) { map }`` recurrence sweep
     (TSVC ``s231`` / ``s235`` / ``s233``): the scalar stride ranking alone
     refuses it -- it sees only that moving the loop inward raises the innermost
     scalar stride -- leaving a top-level sequential loop where ``LoopToMap``
     alone parallelizes with the column outermost. Rule (2) restores that
     maximal-parallelism form.

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


def _map_axis_is_unit_stride(loop: LoopRegion, sdfg: SDFG) -> bool:
    """True if the wrapped map's parallel axis homes on a unit-stride array axis.

    When it does, hoisting the map OUT of the enclosing sequential ``loop`` (making
    the map the outer parallel axis and the loop its inner sequential body) is a net
    win even though it raises the *scalar* innermost stride: the vectorizer
    strip-mines the map's unit-stride parallel axis into SIMD lanes, so the hoisted
    map's loads/stores stay contiguous -- recovering exactly the locality
    :func:`_interchange_lowers_stride` assumes is lost -- while the interchange
    eliminates the sequential loop's per-iteration fork/join of the parallel region
    (one outer parallel region instead of one re-entered every iteration).

    This is the ``for(seq) { map }`` recurrence-sweep shape (TSVC ``s231`` / ``s235``
    / ``s233``: ``aa[j, i] = aa[j-1, i] + ...`` with ``i`` the unit-stride parallel
    column and ``j`` the sequential carry). Leaving the map nested inside the loop
    keeps a top-level sequential loop (``LoopToMap`` alone parallelizes these with
    the parallel column outermost); hoisting matches that maximal-parallelism form.
    The scalar-only :func:`_interchange_lowers_stride` mis-ranks it because it models
    the map axis as one scalar step, blind to the vectorization that recovers the
    contiguity.

    :param loop: A loop whose single-state body holds exactly one map (already
                 validated by ``MoveLoopIntoMap.can_be_applied``).
    :param sdfg: The owning SDFG (for array strides).
    :returns: True if any map parameter homes on a unit-stride (stride-1) axis.
    """
    body = loop.nodes()[0]
    map_entry = next(n for n in body.nodes() if isinstance(n, nodes.MapEntry))
    mparams = list(map_entry.map.params)
    subgraph = body.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    scores = score_indexed_strides(subgraph.edges(), sdfg, mparams)
    return any(_to_float(scores[p][0]) <= 1.0 for p in mparams)


def _interchange_lowers_stride(loop: LoopRegion, sdfg: SDFG) -> bool:
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

    See the module docstring for the per-target rule. ``target='gpu'`` always
    interchanges an applicable loop; ``target='cpu'`` interchanges when it lowers
    the innermost iterated stride OR when the map's parallel axis is unit-stride
    (a hoistable, vectorizable parallel region).
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
        return set()

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
                if self.target != 'gpu' and not (_interchange_lowers_stride(loop, loop.sdfg)
                                                 or _map_axis_is_unit_stride(loop, loop.sdfg)):
                    continue
                MoveLoopIntoMap.apply_to(loop.sdfg, loop=loop, verify=False)
                applied += 1
                changed = True
                break
        return applied or None
