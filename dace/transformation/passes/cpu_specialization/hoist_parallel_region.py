# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Open the OpenMP team ONCE for a whole sequential loop instead of once per trip.

A sequential loop wrapped around a parallel map is the shape almost every stencil and recurrence
kernel canonicalizes to: the loop carries the dependence, the map is the DOALL dimension under it.
Codegen emits a ``#pragma omp parallel for`` for the map, so the region is opened and closed on
EVERY trip of the loop::

    for (j = 1; j < N; j++) {
        #pragma omp parallel for simd
        for (i = 0; i < N - 1; i++) { ... }
    }

At the sizes these kernels run, the trip count is the array extent: tsvc ``s115`` forks 22,819
times, and at roughly 10 us of fork/join each that is ~230 ms of a 320 ms kernel. Two further costs
ride along -- the team is re-created per trip, so nothing a thread warmed stays warm and nothing
pins a thread to the quadrant whose pages it touched.

This pass emits the same computation with ONE region::

    #pragma omp parallel
    {
        for (j = 1; j < N; j++) {
            #pragma omp for simd
            for (i = 0; i < N - 1; i++) { ... }
        }
    }

It does so entirely through schedules, using machinery the CPU target already has: the loop is
outlined into a nested SDFG (:func:`~dace.transformation.helpers.nest_sdfg_subgraph`) and that node
is wrapped in a one-iteration ``CPU_Persistent`` map, which codegen emits as a bare
``#pragma omp parallel``. The maps inside then see a ``CPU_Persistent`` scope above them
(``is_in_scope``, which crosses nested-SDFG boundaries) and emit ``#pragma omp for``. No new
codegen rule, no new pragma.

Legality
--------

``#pragma omp parallel for X`` is by definition ``#pragma omp parallel { #pragma omp for X }``, so
the rewrite changes NOTHING about which thread runs which iteration, nor about the barriers: each
``omp for`` keeps its implicit exit barrier exactly where the ``parallel for``'s join barrier was.
The rewrite is therefore semantics-preserving for ANY dependence structure, loop-carried ones
included -- which is the point, because that is the case a band-and-``nowait`` scheme cannot serve.

What DOES change is that every statement of the loop body which is not inside a worksharing
construct is executed by all P threads instead of once. Hence the two conditions this pass demands:

**(H) Replication-freedom.** Every statement the loop body executes lies inside a ``CPU_Multicore``
map scope. Access nodes and map entries/exits emit nothing, and pure control flow (loop counters,
branch conditions) every thread evaluates identically, because the data it reads was last written
before a barrier. Three things at a state's top level DO emit a statement and so break (H): a
``Tasklet``, which the pass repairs by wrapping it in its own one-iteration ``CPU_Multicore`` map
(an ``omp for`` over one iteration -- run once, by one thread, with a barrier after); a library node
or nested SDFG, refused outright; and an edge between two ACCESS NODES, which is a bulk copy and
is refused too. The copy is the one that does not look like a statement: ``jacobi_2d``'s
``A[1:N-1, 1:N-1] = B[1:N-1, 1:N-1]`` is two access nodes and a memlet, and replicating it would
have every thread write the whole array with no barrier before the next trip reads it.

**(T) No accidental privatization.** Outlining moves a transient that nothing outside the loop
observes INTO the nest, where it is declared inside the parallel region -- one copy per thread. That
is exactly right for the per-iteration scalars a map body is built from, and it is what makes the
rewrite free: they never leave their ``omp for``. It is wrong for a transient that hands a value
from one ``omp for`` to the NEXT, and such a transient is recognisable without any dataflow
analysis, because value passing between two map scopes has to go through an access node at the
state's TOP level. A loop with one of those is refused, unless its lifetime keeps it in the state
struct (``Persistent`` / ``Global``), where it stays shared whichever SDFG owns the descriptor.

Not done here, deliberately
---------------------------

Dropping the barrier (``nowait``) is a separate rewrite, and a much narrower one. For a partition
``B_1..B_P`` of the map's index space and ``R_p(k)`` / ``W_p(k)`` the read / write footprints of
band ``p`` at trip ``k``, ``nowait`` is legal iff

    R_p(k+1) INTERSECT (UNION_q W_q(k))  SUBSET-OF  W_p(k)     for every p and k,

that is, everything a band re-reads from the previous trip it wrote itself. Since ``P`` is not known
until run time the condition has to hold for EVERY partition, which reduces to a local test on the
memlets: every loop-carried dependence must be at distance zero in the map's own parameter -- the
recurrence runs along the sequential axis and nothing else.

tsvc ``s233`` satisfies it (both its recurrences run along the sequential axis, so a band's reads
stay inside the band), and so do ``s231`` and ``s235``. ``s119`` (``aa[i,j] = aa[i-1,j-1] + ...``)
and ``wf_diff_skew`` (``a[i-1,j+1]``) violate it by ONE element at the band boundary; ``s115`` reads
a scalar every band needs but only one band writes. There only a point-to-point handshake -- a real
doacross -- is correct. This pass therefore emits no ``nowait`` at all: the barrier is what makes it
safe for any dependence structure, and telling the two cases apart needs a predicate and an emission
point that do not exist yet (``dace/codegen/targets/cpu.py`` carries the matching
``TODO(later): barriers and map_header += " nowait"``).
"""
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion,
                             ReturnBlock, SDFGState)
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl

#: The schedule whose maps become ``#pragma omp for`` once a ``CPU_Persistent`` scope encloses them.
#: ``Default`` is not accepted: this pass runs after ``cpu_specialize`` has resolved every schedule,
#: so a map still carrying ``Default`` here is one nothing decided and not one to reason about.
WORKSHARED = dtypes.ScheduleType.CPU_Multicore

#: Transient lifetimes that keep ONE instance for the whole program, wherever the descriptor sits.
#: A transient of any other lifetime that moves into the outlined nest is declared inside the
#: parallel region and becomes thread-private -- condition (T).
SHARED_LIFETIMES = (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.Global,
                    dtypes.AllocationLifetime.External)


def top_level_nodes(state: SDFGState) -> List[nodes.Node]:
    """The nodes of ``state`` that no map scope encloses -- the ones a hoisted team would replicate.

    :param state: the state to inspect.
    :returns: the state's scope-free nodes, in the state's own node order.
    """
    scopes = state.scope_dict()
    return [n for n in state.nodes() if scopes[n] is None]


def loop_local_transients(sdfg: SDFG, loop: LoopRegion) -> Set[str]:
    """The transients outlining ``loop`` would move into the nest rather than pass as a connector.

    Mirrors the ``unique_set`` rule of :func:`~dace.transformation.helpers.nest_sdfg_subgraph`: a
    transient read or written inside the loop that no block, and no interstate edge, outside it
    observes.

    :param sdfg: the SDFG holding ``loop``.
    :param loop: the loop region about to be outlined.
    :returns: the names of the transients that would move inside.
    """
    inside_blocks = {id(loop)} | {id(b) for b in loop.all_control_flow_blocks()}
    inside_names: Set[str] = set()
    outside_names: Set[str] = set()
    for block in sdfg.all_control_flow_blocks():
        target = inside_names if id(block) in inside_blocks else outside_names
        if isinstance(block, SDFGState):
            target.update(n.data for n in block.data_nodes())
        elif isinstance(block, ConditionalBlock):
            for cond, _ in block.branches:
                if cond is not None:
                    target.update(cond.get_free_symbols())
        elif isinstance(block, LoopRegion):
            target.update(block.loop_condition.get_free_symbols())
    for edge in sdfg.all_interstate_edges():
        if id(edge.src) not in inside_blocks or id(edge.dst) not in inside_blocks:
            outside_names.update(edge.data.free_symbols)
    return {n for n in inside_names - outside_names if n in sdfg.arrays and sdfg.arrays[n].transient}


@properties.make_properties
class HoistParallelRegion(ppl.Pass):
    """Wrap a sequential loop over parallel maps in one persistent OpenMP team."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Hoist the OpenMP team out of every loop that qualifies.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many loops were hoisted, or ``None`` if none were.
        """
        self.hoisted = 0
        self.visit_sdfg(sdfg)
        if self.hoisted:
            # Outlining moves whole states into a new SDFG, and a nested SDFG that travelled with
            # them still names the SDFG it left as its parent. Validation reads that pointer.
            sdutil.set_nested_sdfg_parent_references(sdfg)
        return self.hoisted or None

    def visit_sdfg(self, sdfg: SDFG) -> None:
        """Walk one SDFG known to sit outside every map scope.

        :param sdfg: the SDFG (root or nested) to walk.
        """
        self.visit_region(sdfg, sdfg)

    def visit_region(self, region: AbstractControlFlowRegion, sdfg: SDFG) -> None:
        """Hoist the OUTERMOST qualifying loop of each chain in ``region``, then descend.

        Outermost, because one region around the whole nest costs one fork where a region around an
        inner loop costs one per trip of the outer one.

        :param region: the control-flow region to walk.
        :param sdfg: the SDFG owning ``region``.
        """
        for block in list(region.nodes()):
            if isinstance(block, LoopRegion) and self.hoistable(block, sdfg):
                self.hoist(block, sdfg)
                self.hoisted += 1
            elif isinstance(block, AbstractControlFlowRegion):
                self.visit_region(block, sdfg)
            elif isinstance(block, SDFGState):
                for node in block.nodes():
                    if isinstance(node, nodes.NestedSDFG) and node.sdfg is not None and block.entry_node(node) is None:
                        self.visit_sdfg(node.sdfg)

    def hoistable(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Whether ``loop`` may be wrapped in a persistent team -- conditions (H) and (T).

        :param loop: the candidate loop region.
        :param sdfg: the SDFG owning ``loop``.
        :returns: ``True`` if wrapping preserves semantics and is worth doing.
        """
        worksharing = False
        for block in loop.all_control_flow_blocks():
            if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
                return False
            if not isinstance(block, SDFGState):
                continue
            for edge in block.edges():
                if (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)
                        and edge.data is not None and not edge.data.is_empty()):
                    return False
            for node in top_level_nodes(block):
                if isinstance(node, (nodes.AccessNode, nodes.Tasklet)):
                    continue
                if isinstance(node, (nodes.MapEntry, nodes.MapExit)) and node.map.schedule == WORKSHARED:
                    worksharing = True
                    continue
                return False
        if not worksharing:
            return False
        privatized = loop_local_transients(sdfg, loop)
        for block in loop.all_control_flow_blocks():
            if not isinstance(block, SDFGState):
                continue
            for node in top_level_nodes(block):
                if (isinstance(node, nodes.AccessNode) and node.data in privatized
                        and sdfg.arrays[node.data].lifetime not in SHARED_LIFETIMES):
                    return False
        return True

    def hoist(self, loop: LoopRegion, sdfg: SDFG) -> None:
        """Outline ``loop`` and put the nest inside a one-iteration ``CPU_Persistent`` map.

        :param loop: the loop region to wrap; must satisfy :meth:`hoistable`.
        :param sdfg: the SDFG owning ``loop``.
        """
        for block in loop.all_control_flow_blocks():
            if isinstance(block, SDFGState):
                for node in top_level_nodes(block):
                    # A statement outside a worksharing construct would run once per THREAD. One
                    # iteration of ``omp for`` runs it once, on one thread, with the barrier kept.
                    if isinstance(node, nodes.Tasklet):
                        xfh.wrap_code_node_in_unit_map(block, node, WORKSHARED, '_single')
        state = xfh.nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop]), start=loop)
        nsdfg = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
        xfh.wrap_code_node_in_unit_map(state, nsdfg, dtypes.ScheduleType.CPU_Persistent, '_team')
