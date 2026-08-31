# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hand a map whose iterations do UNEQUAL work an OpenMP schedule that balances them.

A triangular nest -- ``for i: for j in range(i, N)`` -- gives its first iteration ``N`` units of work
and its last one none. OpenMP's default splits the range into equal CONTIGUOUS blocks, so one thread
gets the long rows and another the short ones and the region costs its slowest thread. Handing that
map ``schedule(dynamic, 1)`` instead is worth 3.0x on TSVC ``s141`` and 1.6x on ``s1232`` (measured,
below).

The reason this pass is narrow rather than a blanket setting is the control that measures the cost:

    L, min of 7, same SDFG / data / process / flags, only the schedule differing:

    kernel          shape                       default  static_8  guided_16  dynamic_1
    tsvc_2_s141     triangular, forks once         1.00      1.45       1.09       3.01
    tsvc_2_s1232    triangular, forks once         1.00      1.21       1.81       1.63
    wf_triangular   equal tiles, re-forks          1.00      1.33       0.62       0.68
    fuse_diamond    rectangular, memory-bound      1.00      0.47       1.11       0.03

``fuse_diamond`` is the whole argument. It is a balanced elementwise stream, and ANY chunk clause
costs it between 2x and 33x: contiguous blocks are what let each thread prefetch linearly and keep
its own pages, and chunking shreds both. A schedule chosen for imbalance must therefore never reach
a map that does not have any.

So two conditions, and both are necessary:

* the work per iteration must actually VARY with the map parameter -- an inner trip count that
  mentions it (:func:`work_ramps_with_parameter`);
* the map must fork ONCE. ``wf_triangular``'s skewed tile-column map has a parametric range but
  equal-sized tiles, and it sits inside the sequential diagonal loop, so it re-forks per diagonal
  and its threads want the same tiles each step for cache reuse. Chunking it costs 32%.

Runs in the CPU specialization stage for the reason the whole stage exists: the verdict is read off
the FINAL map shapes, and it is a target decision, not a canonical one -- a GPU ignores
``omp_schedule`` entirely.
"""
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis


def forks_once(state: SDFGState) -> bool:
    """``True`` iff nothing re-enters ``state``'s scope per iteration of an enclosing loop.

    A parallel region inside a sequential loop pays its fork/join every time round, and its threads
    keep whatever they cached only if they are handed the same work again -- which a dynamic
    schedule is precisely a promise not to do.
    """
    region: Optional[ControlFlowRegion] = state.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            return False
        region = region.parent_graph
    return True


def outer_names_in(expr: Any, outer: Set[str]) -> bool:
    """``True`` iff the symbolic ``expr`` mentions any name in ``outer``."""
    if expr is None:
        return False
    try:
        free = symbolic.pystr_to_symbolic(expr).free_symbols
    except Exception:  # noqa: BLE001 -- an unparsable bound proves nothing either way
        return False
    return any(str(s) in outer for s in free)


def nested_names_for(nsdfg: nodes.NestedSDFG, outer: Set[str]) -> Set[str]:
    """Inner symbol names that a NestedSDFG binds to an expression mentioning ``outer``.

    The map parameter is renamed at the boundary, so an inner loop bound spelled ``_loop_it_0`` is
    only the outer parameter because ``symbol_mapping`` says so.
    """
    return {inner for inner, outer_expr in nsdfg.symbol_mapping.items() if outer_names_in(outer_expr, outer)}


def loop_trip_ramps(loop: LoopRegion, names: Set[str]) -> bool:
    """``True`` iff this loop's trip count depends on ``names``.

    Start, end and stride are all read: ``for j in range(i, N)`` ramps through its START, and
    ``for j in range(0, N - i)`` through its END.
    """
    return any(
        outer_names_in(part, names) for part in (loop_analysis.get_init_assignment(loop),
                                                 loop_analysis.get_loop_end(loop), loop_analysis.get_loop_stride(loop)))


def work_ramps_with_parameter(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """``True`` iff some inner loop or map does an amount of work that varies with ``entry``'s params.

    Three routes, because the pipeline produces three shapes for the same nest:

    * an inner Map still in the same state;
    * a ``LoopRegion`` inside the NestedSDFG that ``LoopToMap`` minted for the body;
    * a ramping DATA VOLUME -- the most general of the three, and the only one that catches a
      triangular nest whose inner loop was lifted to a library node. ``acc += a[i, j]`` over
      ``j in range(i, N)`` becomes a ``Reduce`` reading ``a[i, i:N]``, which has no loop left to
      read a bound off, but whose extent is still ``N - i``. An edge whose element COUNT mentions
      the parameter is doing an amount of work that varies with it, whatever the node is.
    """
    outer = set(entry.map.params)
    for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.MapEntry):
            if any(outer_names_in(part, outer) for rng in node.map.range.ndrange() for part in rng):
                return True
        elif isinstance(node, nodes.NestedSDFG):
            names = nested_names_for(node, outer)
            if not names:
                continue
            for region in node.sdfg.all_control_flow_regions(recursive=True):
                if isinstance(region, LoopRegion) and loop_trip_ramps(region, names):
                    return True
            if volume_ramps(node.sdfg, names):
                return True
    return False


def volume_ramps(sdfg: SDFG, names: Set[str]) -> bool:
    """``True`` iff some memlet inside ``sdfg`` moves an element COUNT that depends on ``names``.

    Read inside the nested SDFG rather than at the map boundary on purpose: memlet propagation
    summarises the boundary edge up to the whole array (``a[0:N, 0:N]``), which is a constant extent
    and hides the very ramp this is looking for. The per-iteration subset survives inside.
    """
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.subset is None:
                continue
            try:
                extent = edge.data.subset.num_elements()
            except Exception:  # noqa: BLE001 -- an unreadable subset proves nothing either way
                continue
            if outer_names_in(extent, names):
                return True
    return False


@properties.make_properties
class ScheduleRampedMaps(ppl.Pass):
    """Give a fork-once map with a ramping trip count a balancing OpenMP schedule."""

    CATEGORY: str = 'Optimization Preparation'

    omp_schedule = properties.EnumProperty(
        dtype=dtypes.OMPScheduleType,
        default=dtypes.OMPScheduleType.Dynamic,
        desc='Schedule handed to a ramped map. Dynamic measured best across the imbalanced kernels '
        '(geomean 2.21x over s141 and s1232, against 1.44x for static and 1.40x for guided).',
    )

    omp_chunk_size = properties.Property(
        dtype=int,
        default=1,
        desc='Chunk handed with it. 1 balances a linear ramp exactly; the per-chunk atomic is paid '
        'once per outer iteration, against inner work that is itself proportional to the parameter.',
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[List[str]]:
        touched: List[str] = []
        for g in sdfg.all_sdfgs_recursive():
            for state in g.all_states():
                for node in state.nodes():
                    if not isinstance(node, nodes.MapEntry):
                        continue
                    if node.map.schedule != dtypes.ScheduleType.CPU_Multicore:
                        continue
                    if node.map.omp_schedule != dtypes.OMPScheduleType.Default:
                        continue  # someone already decided; do not overrule it
                    if state.entry_node(node) is not None or not forks_once(state):
                        continue
                    if not work_ramps_with_parameter(state, node):
                        continue
                    node.map.omp_schedule = self.omp_schedule
                    node.map.omp_chunk_size = int(self.omp_chunk_size)
                    touched.append(node.map.label)
        return touched or None

    def report(self, pass_retval: List[str]) -> str:
        return f'Balanced the schedule of {len(pass_retval)} ramped map(s): {", ".join(pass_retval)}'
