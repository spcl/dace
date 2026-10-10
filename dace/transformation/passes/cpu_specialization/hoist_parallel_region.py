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
before a barrier. What does emit a statement at a state's top level is repaired where it can be and
refused where it cannot:

- a ``Tasklet`` is wrapped in its own one-iteration ``CPU_Multicore`` map (an ``omp for`` over one
  iteration -- run once, by one thread, with a barrier after);
- a ``Sequential`` map with no parallel map inside it -- one ``cpu_specialize`` sequentialized for
  its size -- becomes a worksharing map: inside the team it costs one barrier, not a fork;
- an edge between two ACCESS NODES is a bulk copy, the statement that does not look like one:
  ``jacobi_2d``'s ``A[1:N-1, 1:N-1] = B[1:N-1, 1:N-1]`` is two access nodes and a memlet, and
  replicated it has every thread write the whole array with no barrier before the next trip reads
  it. It becomes a worksharing map over the copied elements; a copy that re-linearizes has no
  element-for-element map and is refused;
- a library node whose expansion is one map over its output's elements (``np.where``'s
  ``MergeLibraryNode``, a ``FillLibraryNode``) is expanded into that map and shared out; any other
  library node, and any nested SDFG, is refused.

:func:`share_out_bulk_statements` makes the repairs. A loop is hoisted only if it already holds a
``CPU_Multicore`` map: one whose maps the cost model all made sequential has nothing worth a team. And
a repair is a barrier every time it runs, so each must run in a loop that also runs a worksharing map
whose fork the team saves (:func:`repairs_amortized`).

**(T) No accidental privatization.** Outlining moves a transient that nothing outside the loop
observes INTO the nest, where it is declared inside the parallel region -- one copy per thread. That
is exactly right for the per-iteration scalars a map body is built from, and it is what makes the
rewrite free: they never leave their ``omp for``. It is wrong for a transient that hands a value
from one ``omp for`` to the NEXT, and such a transient is recognisable without any dataflow
analysis, because value passing between two map scopes has to go through an access node at the
state's TOP level. Those stay in the enclosing SDFG and cross into the nest as connectors
(:func:`handed_between_scopes`), so they are allocated before the region opens and the team shares
them; a ``Persistent`` / ``Global`` one lives in the state struct and is shared already.

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

import copy
from typing import Any

from dace import SDFG, Memlet, data, dtypes, properties, subsets
from dace.libraries.standard.nodes import FillLibraryNode, MergeLibraryNode
from dace.sdfg import memlet_utils, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge, SubgraphView
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl

#: The schedule whose maps become ``#pragma omp for`` once a ``CPU_Persistent`` scope encloses them.
#: ``Default`` is not accepted: this pass runs after ``cpu_specialize`` has resolved every schedule,
#: so a map still carrying ``Default`` here is one nothing decided and not one to reason about.
WORKSHARED = dtypes.ScheduleType.CPU_Multicore

#: Transient lifetimes that keep ONE instance for the whole program, wherever the descriptor sits.
#: A transient of any other lifetime that moves into the outlined nest is declared inside the
#: parallel region and becomes thread-private -- condition (T).
SHARED_LIFETIMES = (
    dtypes.AllocationLifetime.Persistent,
    dtypes.AllocationLifetime.Global,
    dtypes.AllocationLifetime.External,
)


def top_level_nodes(state: SDFGState) -> list[nodes.Node]:
    """The nodes of ``state`` that no map scope encloses -- the ones a hoisted team would replicate.

    :param state: the state to inspect.
    :returns: the state's scope-free nodes, in the state's own node order.
    """
    scopes = state.scope_dict()
    return [n for n in state.nodes() if scopes[n] is None]


def binds_a_view(edge: MultiConnectorEdge[Memlet], state: SDFGState) -> bool:
    """Whether ``edge`` is the edge a view reads its data off: an alias, which emits no statement."""
    return any(
        isinstance(node.desc(state.sdfg), data.View) and sdutil.get_view_edge(state, node) is edge
        for node in (edge.src, edge.dst)
    )


def bulk_copies(state: SDFGState) -> list[MultiConnectorEdge[Memlet]]:
    """The edges of ``state`` that copy between two access nodes outside every map scope."""
    scopes = state.scope_dict()
    return [
        edge
        for edge in state.edges()
        if isinstance(edge.src, nodes.AccessNode)
        and isinstance(edge.dst, nodes.AccessNode)
        and not edge.data.is_empty()
        and scopes[edge.src] is None
        and not binds_a_view(edge, state)
    ]


def shares_out_as_a_map(edge: MultiConnectorEdge[Memlet], state: SDFGState) -> bool:
    """Whether the copy ``edge`` becomes a map that writes each element once: plain arrays on both sides,
    no accumulation, and the same extents once unit dimensions are dropped (no re-linearization)."""
    sdfg = state.sdfg
    if edge.data.wcr is not None:
        return False
    if any(isinstance(node.desc(sdfg), data.View) for node in (edge.src, edge.dst)):
        return False
    if not memlet_utils.can_memlet_be_turned_into_a_map(edge, state, sdfg):
        return False
    source = edge.data.get_src_subset(edge, state) or subsets.Range.from_array(edge.src.desc(sdfg))
    target = edge.data.get_dst_subset(edge, state) or subsets.Range.from_array(edge.dst.desc(sdfg))
    extents = [[size for size in subset.size() if size != 1] for subset in (source, target)]
    return bool(extents[0]) and [str(e) for e in extents[0]] == [str(e) for e in extents[1]]


def elementwise(node: nodes.Node) -> bool:
    """Whether ``node`` is a library node whose ``pure`` expansion is one map over its output's elements."""
    return isinstance(node, (FillLibraryNode, MergeLibraryNode))


def expand_in_place(node: nodes.LibraryNode, state: SDFGState) -> list[nodes.MapEntry]:
    """Expand the elementwise ``node`` with its ``pure`` implementation and inline the result.

    :returns: the map entries the expansion brought in.
    """
    from dace.transformation.interstate import InlineSDFG  # Avoid import loop

    before = set(state.nodes())
    node.expand(state, "pure")
    nested = next(n for n in state.nodes() if n not in before and isinstance(n, nodes.NestedSDFG))
    inner_maps = [n for n, _ in nested.sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    InlineSDFG.apply_to(state.sdfg, nested_sdfg=nested, verify=False, save=False)
    return inner_maps


def fill_as_one_map(entry: nodes.MapEntry, state: SDFGState) -> nodes.MapEntry | None:
    """Rebuild a sequential leaf map whose whole body is one static fill as ONE worksharing map over the
    map's positions and the fill's elements: ``SpecializeCpuTransfers``' row loop around a row memset
    goes back to the map it came from, so the body keeps one extent and the band cuts its columns.

    :returns: the new map's entry, or ``None`` if the map is not that shape.
    """
    exit_node = state.exit_node(entry)
    body = [n for n in state.scope_children()[entry] if n is not exit_node]
    outputs = state.out_edges(exit_node)
    if len(body) != 1 or not isinstance(body[0], FillLibraryNode) or body[0].value_edge(state) is not None:
        return None
    if len(outputs) != 1 or not isinstance(outputs[0].dst, nodes.AccessNode):
        return None
    if any(not e.data.is_empty() for e in state.in_edges(entry)):
        return None
    fill = body[0]
    (written,) = state.out_edges(fill)
    region = written.data.subset
    if not isinstance(region, subsets.Range):
        return None
    ranges = {
        param: str(subsets.Range([rng])) for param, rng in zip(entry.map.params, entry.map.range.ranges, strict=True)
    }
    index = []
    for dim, ((begin, _, _), size) in enumerate(zip(region.ranges, region.size(), strict=True)):
        if size == 1:
            index.append(str(begin))
        else:
            ranges[f"__fill{dim}"] = f"0:{size}"
            index.append(f"{begin} + __fill{dim}")
    target = outputs[0].dst
    predecessors = [e.src for e in state.in_edges(entry)]
    state.remove_nodes_from([entry, fill, exit_node])
    _, new_entry, _ = state.add_mapped_tasklet(
        f"{fill.label}_map",
        ranges,
        {},
        f"__out = {fill.value}",
        {"__out": Memlet(data=target.data, subset=", ".join(index))},
        schedule=WORKSHARED,
        external_edges=True,
        output_nodes={target.data: target},
    )
    for source in predecessors:
        state.add_nedge(source, new_entry, Memlet())
    return new_entry


def reads_by_data(sdfg: SDFG) -> dict[str, list[subsets.Subset | None]]:
    """Every read of every data container of ``sdfg`` (not of its nests), as the subset it reads; an
    accumulating write reads too."""
    reads: dict[str, list[subsets.Subset | None]] = {}
    for state in sdfg.states():
        for node in state.data_nodes():
            for edge in state.out_edges(node):
                if not edge.data.is_empty():
                    reads.setdefault(node.data, []).append(edge.data.get_src_subset(edge, state))
            for edge in state.in_edges(node):
                if edge.data.wcr is not None:
                    reads.setdefault(node.data, []).append(edge.data.get_dst_subset(edge, state))
    return reads


def whole_fill_hull(
    fill: FillLibraryNode, state: SDFGState, reads: dict[str, list[subsets.Subset | None]], inputs: set[str]
) -> subsets.Subset | None:
    """The part of a transient that ``fill`` must write when it fills the whole array: the hull of its
    ``reads``. ``None`` unless that hull is a strict part of the array and names only ``inputs`` -- the
    symbols the SDFG is called with, the only ones known wherever the fill runs.

    Nobody observes what the fill writes outside the hull, and only the hull has the extent the rest of
    the loop body is banded on (``psum_solqa[:] = 0`` beside ``[kidia-1:kfdia]``).
    """
    outputs = state.out_edges(fill)
    if len(outputs) != 1 or not isinstance(outputs[0].dst, nodes.AccessNode):
        return None
    edge = outputs[0]
    desc = edge.dst.desc(state.sdfg)
    full = subsets.Range.from_array(desc)
    parts = reads.get(edge.dst.data, [])
    if not desc.transient or isinstance(desc, data.View) or edge.data.get_dst_subset(edge, state) != full:
        return None
    if not parts or any(part is None for part in parts):
        return None
    hull = parts[0]
    for part in parts[1:]:
        hull = subsets.union(hull, part) if hull is not None else None
    if hull is None or str(hull) == str(full) or not hull.free_symbols <= inputs:
        return None
    return hull


def narrow_whole_fills(sdfg: SDFG) -> None:
    """Shrink every whole-array fill of a transient in ``sdfg`` and its nests to the hull of the
    transient's reads (:func:`whole_fill_hull`). Run before any loop is outlined: a nest's boundary
    memlet is the union of what it touches, and would widen every hull it is part of."""
    for owner in sdfg.all_sdfgs_recursive():
        reads = inputs = None
        for state in owner.states():
            for node in [n for n in state.nodes() if isinstance(n, FillLibraryNode)]:
                if reads is None or inputs is None:
                    reads, inputs = reads_by_data(owner), owner.free_symbols
                hull = whole_fill_hull(node, state, reads, inputs)
                if hull is not None:
                    (edge,) = state.out_edges(node)
                    edge.data.subset = copy.deepcopy(hull)  # the hull may be a read memlet's own subset


def split_shared_views(sdfg: SDFG, loop: LoopRegion) -> None:
    """Give every view ``loop`` shares with the rest of ``sdfg`` its own descriptor inside the loop.

    A view holds no data and is bound per access node, so renaming the uses in the loop's states changes
    nothing -- and outlining can then move the view into the nest with its binding, where as a connector
    it would have none.
    """
    private = loop_local_transients(sdfg, loop)
    renamed: dict[str, str] = {}
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        shared = {n.data for n in block.data_nodes() if isinstance(n.desc(sdfg), data.View) and n.data not in private}
        for name in sorted(shared):
            if name not in renamed:
                renamed[name] = sdfg.add_datadesc(f"{name}_loop", copy.deepcopy(sdfg.arrays[name]), find_new_name=True)
        if shared:
            block.replace_dict({name: renamed[name] for name in shared})


def sequential_leaf_map(node: nodes.Node, state: SDFGState) -> bool:
    """Whether ``node`` enters a ``Sequential`` map with no parallel map anywhere inside it."""
    if not isinstance(node, nodes.MapEntry) or node.map.schedule != dtypes.ScheduleType.Sequential:
        return False
    inner = state.scope_subgraph(node).nodes()
    nested = [n.sdfg for n in inner if isinstance(n, nodes.NestedSDFG)]
    return not any(
        isinstance(n, nodes.MapEntry) and n.map.schedule != dtypes.ScheduleType.Sequential
        for n in [*inner, *(m for sd in nested for m, _ in sd.all_nodes_recursive())]
    )


def replication_free(loop: LoopRegion) -> bool:
    """Condition (H): every statement at the top level of ``loop``'s states is a worksharing map or one
    :func:`share_out_bulk_statements` repairs, and at least one worksharing map is there already.

    :param loop: the candidate loop region.
    :returns: ``True`` if, once repaired, a team around ``loop`` replicates no statement.
    """
    worksharing = False
    for block in loop.all_control_flow_blocks():
        if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
            return False
        if not isinstance(block, SDFGState):
            continue
        if not all(shares_out_as_a_map(edge, block) for edge in bulk_copies(block)):
            return False
        for node in top_level_nodes(block):
            if isinstance(node, (nodes.AccessNode, nodes.Tasklet)) or elementwise(node):
                continue
            if isinstance(node, (nodes.MapEntry, nodes.MapExit)) and node.map.schedule == WORKSHARED:
                worksharing = True
                continue
            entry = block.entry_node(node) if isinstance(node, nodes.MapExit) else node
            if not sequential_leaf_map(entry, block):
                return False
    return worksharing


def innermost_loop(block: ControlFlowBlock, loop: LoopRegion) -> LoopRegion:
    """The innermost loop region strictly enclosing ``block`` inside ``loop`` (``loop`` itself if none is
    closer)."""
    region = block.parent_graph
    while region is not loop and not isinstance(region, LoopRegion):
        region = region.parent_graph
    return region


def repairs_amortized(loop: LoopRegion) -> bool:
    """Whether every statement :func:`share_out_bulk_statements` repairs in ``loop`` runs in a loop that also
    runs a worksharing map, directly or in a loop nested inside it.

    A repaired statement costs a barrier each time it runs, and the team pays for that with the fork it saves
    per worksharing map. Where a loop inside ``loop`` holds repairs and no worksharing map, every one of its
    trips adds barriers and saves nothing: ``seidel_2d``'s scalar recurrence would become five barriers per
    element.

    :param loop: a loop :func:`replication_free` accepts.
    :returns: ``True`` if no repair runs more often than the worksharing it rides on.
    """
    covered: set[int] = set()
    repaired: set[int] = set()
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        top = top_level_nodes(block)
        if any(isinstance(n, nodes.MapEntry) and n.map.schedule == WORKSHARED for n in top):
            region = innermost_loop(block, loop)
            covered.add(id(region))
            while region is not loop:
                region = innermost_loop(region, loop)
                covered.add(id(region))
        if bulk_copies(block) or any(
            isinstance(n, nodes.Tasklet) or elementwise(n) or sequential_leaf_map(n, block) for n in top
        ):
            repaired.add(id(innermost_loop(block, loop)))
    return repaired <= covered


def share_out_bulk_statements(sdfg: SDFG, loop: LoopRegion) -> None:
    """Turn every bulk copy, elementwise library node and sequential leaf map at the top level of
    ``loop``'s states into a ``CPU_Multicore`` map over the elements it writes, and split the views the
    loop shares with the rest of ``sdfg`` (:func:`split_shared_views`).

    A sequential leaf map holding nothing but a fill becomes one map over both
    (:func:`fill_as_one_map`), which keeps every map of the body on one extent.

    :param sdfg: the SDFG owning ``loop``.
    :param loop: a loop :func:`replication_free` accepts.
    """
    split_shared_views(sdfg, loop)
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        for edge in bulk_copies(block):
            entry, _ = memlet_utils.memlet_to_map(edge, block, block.sdfg)
            entry.map.schedule = WORKSHARED
        for node in [n for n in top_level_nodes(block) if elementwise(n)]:
            for entry in expand_in_place(node, block):
                entry.map.schedule = WORKSHARED
        for node in [n for n in top_level_nodes(block) if sequential_leaf_map(n, block)]:
            if fill_as_one_map(node, block) is None:
                node.map.schedule = WORKSHARED


def loop_local_transients(sdfg: SDFG, loop: LoopRegion) -> set[str]:
    """The transients outlining ``loop`` would move into the nest rather than pass as a connector.

    Mirrors the ``unique_set`` rule of :func:`~dace.transformation.helpers.nest_sdfg_subgraph`: a
    transient read or written inside the loop that no block, and no interstate edge, outside it
    observes.

    :param sdfg: the SDFG holding ``loop``.
    :param loop: the loop region about to be outlined.
    :returns: the names of the transients that would move inside.
    """
    inside_blocks = {id(loop)} | {id(b) for b in loop.all_control_flow_blocks()}
    inside_names: set[str] = set()
    outside_names: set[str] = set()
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


def top_level_loop_locals(sdfg: SDFG, loop: LoopRegion) -> set[str]:
    """The loop-local transients with an access node at the top level of one of ``loop``'s states and a
    lifetime that is not shared already: what outlining would privatize although it can carry a value
    from one map scope to another. Views are not among them: they alias storage and hold none.

    :param sdfg: the SDFG holding ``loop``.
    :param loop: the loop region about to be outlined.
    :returns: their names.
    """
    privatized = loop_local_transients(sdfg, loop)
    return {
        node.data
        for block in loop.all_control_flow_blocks()
        if isinstance(block, SDFGState)
        for node in top_level_nodes(block)
        if isinstance(node, nodes.AccessNode)
        and node.data in privatized
        and sdfg.arrays[node.data].lifetime not in SHARED_LIFETIMES
        and not isinstance(sdfg.arrays[node.data], data.View)
    }


def outlinable(sdfg: SDFG, loop: LoopRegion) -> bool:
    """Whether ``loop`` can be outlined with :func:`top_level_loop_locals` kept outside the nest.

    A container crossing the nest boundary -- one the rest of ``sdfg`` uses, or a transient kept outside
    -- cannot have an extent naming a symbol the loop defines (its iterators, the symbols its interstate
    edges assign): the nest would need that symbol from its caller before the loop has assigned it.
    Views the loop shares with the rest of the SDFG are not a reason to refuse: :func:`split_shared_views`
    gives them a descriptor of their own.

    :param sdfg: the SDFG holding ``loop``.
    :param loop: the loop region about to be outlined.
    :returns: ``True`` if outlining keeps the SDFG valid.
    """
    defined = {b.loop_variable for b in [loop, *loop.all_control_flow_blocks()] if isinstance(b, LoopRegion)}
    for edge in loop.all_interstate_edges(recursive=True):
        defined.update(edge.data.assignments)
    private = loop_local_transients(sdfg, loop) - top_level_loop_locals(sdfg, loop)
    used = {n.data for b in loop.all_control_flow_blocks() if isinstance(b, SDFGState) for n in b.data_nodes()}
    return not any(
        {str(s) for s in sdfg.arrays[name].free_symbols} & defined
        for name in used - private
        if not isinstance(sdfg.arrays[name], data.View)
    )


def outline(loop: LoopRegion, sdfg: SDFG, keep_outside: set[str]) -> None:
    """Nest ``loop`` into a nested SDFG wrapped in a one-iteration ``CPU_Persistent`` map.

    :param loop: the loop region to outline.
    :param sdfg: the SDFG owning ``loop``.
    :param keep_outside: transients that stay in ``sdfg``, allocated before the region opens.
    """
    # ``loop`` may not be a node of ``sdfg`` when nested -- use ``loop.parent_graph`` instead.
    state = xfh.nest_sdfg_subgraph(sdfg, SubgraphView(loop.parent_graph, [loop]), start=loop, keep_outside=keep_outside)
    nsdfg = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    # The outlining maps every symbol of ``sdfg`` into the nest. One that a LATER loop assigns then reads
    # as used before that loop runs -- a free symbol of ``sdfg`` -- and outlining that loop in turn
    # exports it back out through a ``symbolic_output`` state. Only what the nest reads crosses.
    read_inside = nsdfg.sdfg.free_symbols
    for name in [name for name in nsdfg.symbol_mapping if name not in read_inside]:
        del nsdfg.symbol_mapping[name]
    xfh.wrap_code_node_in_unit_map(state, nsdfg, dtypes.ScheduleType.CPU_Persistent, "_team")


@properties.make_properties
class HoistParallelRegion(ppl.Pass):
    """Wrap a sequential loop over parallel maps in one persistent OpenMP team."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Hoist the OpenMP team out of every loop that qualifies.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many loops were hoisted, or ``None`` if none were.
        """
        self.hoisted = 0
        narrow_whole_fills(sdfg)
        # Outlining moves whole states into a new SDFG through ``add_node``, which re-homes every nested
        # SDFG that travelled with them.
        self.visit_sdfg(sdfg)
        return self.hoisted or None

    def visit_sdfg(self, sdfg: SDFG) -> None:
        """Walk one SDFG known to sit outside every map scope.

        :param sdfg: the SDFG (root or nested) to walk.
        """
        self.visit_region(sdfg, sdfg)

    def visit_region(self, region: AbstractControlFlowRegion, sdfg: SDFG) -> None:
        """Hoist the OUTERMOST qualifying loop of each chain in ``region``, then descend.

        Outermost, because one region around the whole nest costs one fork where a region around an
        inner loop costs one per trip of the outer one. A loop satisfying (H) that can be outlined has
        its bulk statements shared out on the spot: the team hoist takes every
        such loop, so a repair is never left behind in a loop that stays sequential.

        :param region: the control-flow region to walk.
        :param sdfg: the SDFG owning ``region``.
        """
        for block in list(region.nodes()):
            if (
                isinstance(block, LoopRegion)
                and replication_free(block)
                and repairs_amortized(block)
                and outlinable(sdfg, block)
            ):
                share_out_bulk_statements(sdfg, block)
                if self.hoistable(block, sdfg):
                    self.hoist(block, sdfg)
                    self.hoisted += 1
                    continue
            if isinstance(block, AbstractControlFlowRegion):
                self.visit_region(block, sdfg)
            elif isinstance(block, SDFGState):
                for node in block.nodes():
                    if isinstance(node, nodes.NestedSDFG) and node.sdfg is not None and block.entry_node(node) is None:
                        self.visit_sdfg(node.sdfg)

    def hoistable(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Whether this pass rewrites the replication-free, shared-out ``loop``: always, for the team.

        :param loop: the candidate loop region.
        :param sdfg: the SDFG owning ``loop``.
        :returns: ``True``.
        """
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
                        xfh.wrap_code_node_in_unit_map(block, node, WORKSHARED, "_single")
        outline(loop, sdfg, top_level_loop_locals(sdfg, loop))
