# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization from the ``SMALL`` facts: which parallel scopes run sequentially, and how a map whose leading
axes are ``SMALL`` is shared out.

The canonical form is the maximally parallel, device-neutral one; here, with the target known, parallelism is given
back where a region cannot pay for itself. One notion of small decides it: a ``SMALL`` symbol or a literal below
``SMALL_CONSTANT_EXTENT`` (:func:`~dace.symbolic.is_small_extent`), plus the segment an enclosing loop reads from
data, which is small by derivation. In order:

1. **Nested parallelism.** A map (or library node) inside a parallel map of the same device schedule would open a
   team per outer iteration: sequential at any size, transitively across nested-SDFG boundaries.
2. **A small map.** A top-level map whose own iteration count (with the sequential maps it pins) is small runs
   sequentially; :func:`release_one_trip_maps` already took the maps of at most one iteration.
3. **A data segment under a long loop.** A map whose range starts and ends at values read from data (a CSR row,
   ``indptr[k]:indptr[k + 1]``), or whose extent each outer iteration reads from data (``0:cnt``), inside a
   sequential loop runs a handful of iterations per outer one: a derived ``SMALL`` extent. ilu0's elimination forked
   once per row over a row of about ten: 1.5 s against numba's sequential 11 ms; amg_setup's per-row gather, 9.5 s
   against 0.55 s. A short loop does not count (a level loop of 15 trips, :func:`short_loop`): each of its trips
   reads a whole level, not a handful.
4. **Leading ``SMALL`` axes.** A parallel map over ``(b, k, i)`` with ``b`` small shares ``b`` out together with
   the first larger axis (``collapse(2)``), keeping the axis order
   (:func:`~dace.transformation.helpers.share_out_past_leading_small_axes`).

An enclosing loop alone is no reason: a region entered ``E`` times costs ``E * (fork + work/P)`` against
``E * work``, so ``E`` cancels (npbench ``cavity_flow``'s 1,044,484-element pressure map).

Bulk transfers are decided in
:mod:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers`; a copy / memset node here follows
:func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer`.
"""

import copy
from collections.abc import Callable
from typing import Any

import sympy

from dace import SDFG, data, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.symbolic_facts import SMALL_CONSTANT_EXTENT
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import share_out_past_leading_small_axes
from dace.transformation.passes.analysis import loop_analysis

#: Schedules a CPU map or library node still parallelizes under. ``Default`` is included because
#: canonicalization leaves the schedule unset: codegen infers it, and a top-level ``Default`` map
#: becomes ``CPU_Multicore``. Anything else (GPU schedules, ``Unrolled``, ``CPU_Persistent``) is
#: someone else's decision and is left untouched.
CPU_PARALLEL_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)


#: The line a map made sequential here adds under its ``parallel`` class line, so the rendered loop -- which loses
#: its pragma -- says why. Each rule fills in its own reason.
SEQUENTIAL_ON_CPU = "run sequentially on CPU -- {reason}"
NESTED_REASON = "a parallel map encloses it, and a region opens one OpenMP team level only"
SMALL_REASON = "its {count} iterations are small"
ONE_TRIP_REASON = "it runs at most once, so it is a guard rather than a loop"
DATA_SEGMENT_REASON = (
    "its range {begin}:{end} is read from data inside a sequential loop, so every outer iteration would fork "
    "over a few elements"
)


def pinned_inner_trips(entry: nodes.MapEntry, children: dict[Any, list[nodes.Node]]) -> Any:
    """The iterations of the deepest chain of already ``Sequential`` maps in ``entry``'s body. No release can give
    their work a region of its own (a reduction axis split off by ``SequentializeReductionAxes``), so the region
    ``entry`` opens pays for them; the own-count rule assumes inner maps are released instead."""
    trips = [
        child.map.range.num_elements() * pinned_inner_trips(child, children)
        for child in children[entry]
        if isinstance(child, nodes.MapEntry) and child.map.schedule == dtypes.ScheduleType.Sequential
    ]
    return trips[0] if len(trips) == 1 else 1


def note_sequential(entry: nodes.MapEntry, reason: str) -> None:
    """Make ``entry``'s map ``Sequential`` and add the reason under its class line."""
    from dace.transformation.passes.canonicalize.annotate_loop_kinds import PARALLEL

    entry.map.schedule = dtypes.ScheduleType.Sequential
    entry.specialization_hint = f"{entry.specialization_hint or PARALLEL}\n{SEQUENTIAL_ON_CPU.format(reason=reason)}"


def data_derived(sdfg: SDFG, names: set[str], seen: set[tuple[int, str]] | None = None) -> bool:
    """Whether one of ``names`` is a container of ``sdfg``, or a symbol assigned from one: on an interstate edge
    of ``sdfg``, or through the mapping of the nested SDFG node holding it, transitively."""
    seen = set() if seen is None else seen
    for name in names:
        if (id(sdfg), name) in seen:
            continue
        seen.add((id(sdfg), name))
        if name in sdfg.arrays:
            return True
        for edge in sdfg.all_interstate_edges():
            value = edge.data.assignments.get(name)
            if value is not None and data_derived(
                sdfg, set(map(str, symbolic.free_symbols_and_functions(value) | symbolic.arrays(value))), seen
            ):
                return True
        node = sdfg.parent_nsdfg_node
        if node is not None and name in node.symbol_mapping:
            mapped = {str(s) for s in symbolic.pystr_to_symbolic(node.symbol_mapping[name]).free_symbols}
            if data_derived(sdfg.parent_sdfg, mapped, seen):
                return True
    return False


def innermost_loop(state: SDFGState) -> LoopRegion | None:
    """The innermost ``LoopRegion`` enclosing ``state``, crossing nested-SDFG boundaries, or ``None``."""
    region = state.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            return region
        if isinstance(region, SDFG):
            parent = region.parent
            region = parent.parent_graph if parent is not None else None
        else:
            region = region.parent_graph
    return None


def in_sequential_loop(state: SDFGState) -> bool:
    """Whether a ``LoopRegion`` encloses ``state``, crossing nested-SDFG boundaries."""
    return innermost_loop(state) is not None


def short_loop(loop: LoopRegion, facts: symbolic.Facts) -> bool:
    """Whether ``loop`` provably runs few trips: a ``SMALL`` or short literal count (:func:`symbolic.is_small_extent`)."""
    start, end, step = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    if start is None or end is None or step is None or step != 1:
        return False
    return symbolic.is_small_extent(end - start + 1, facts)


def assigned_in_enclosing_loop(state: SDFGState, names: set[str]) -> set[str]:
    """Those of ``names`` an interstate edge inside a ``LoopRegion`` enclosing ``state`` assigns, crossing nested-SDFG
    boundaries through the symbol mapping: values that change from one outer iteration to the next."""
    sdfg = state.sdfg
    assigned = set()
    region = state.parent_graph
    while region is not None and region is not sdfg:
        if isinstance(region, LoopRegion):
            assigned |= {name for edge in region.all_interstate_edges() for name in edge.data.assignments} & names
        region = region.parent_graph
    node = sdfg.parent_nsdfg_node
    if node is not None:
        for name in names - assigned:
            if name in node.symbol_mapping:
                outer = {str(s) for s in symbolic.pystr_to_symbolic(node.symbol_mapping[name]).free_symbols}
                if assigned_in_enclosing_loop(sdfg.parent, outer):
                    assigned.add(name)
    return assigned


def data_segment(
    state: SDFGState, entry: nodes.MapEntry, small: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """``(begin, end)`` of the first dimension of ``entry`` whose two ends are both read from data, or whose ends
    name only constants, ``small`` sizes and values an enclosing loop's iteration reads from data (``0:cnt`` with
    ``cnt = indptr[i + 1] - indptr[i]``, ``0:order + 2 - d`` with ``d = dl[ip]``), if any. A range a loop variable
    or a size moves (``0:i``, a chunk ``c:min(c + K, hit)``) is not one."""
    sdfg = state.sdfg
    for begin, end, _ in entry.map.range:
        ends = [{str(s) for s in symbolic.free_symbols_and_functions(bound)} for bound in (begin, end)]
        if all(ends) and all(data_derived(sdfg, names) for names in ends):
            return str(begin), str(end)
        names = set().union(*ends)
        local = assigned_in_enclosing_loop(state, names)
        if (
            local
            and names - local <= small
            and all(data_derived(sdfg, {name}) or segment_counter(state, name) for name in local)
        ):
            return str(begin), str(end)
    return None


def segment_counter(state: SDFGState, name: str) -> bool:
    """Whether ``name`` counts trips of data segments: inside the loop enclosing ``state`` every assignment to it is
    a literal or ``name + literal``, and each increment lies in a loop whose bounds are read from data (``ntouch``
    of a CSR row product), so it is no larger than those segments."""
    loop = innermost_loop(state)
    if loop is None or loop.sdfg is not state.sdfg:
        return False
    symbol = symbolic.symbol(name)
    increments = 0
    for region in loop.all_control_flow_regions():
        for edge in region.edges():
            value = edge.data.assignments.get(name)
            if value is None:
                continue
            expr = symbolic.pystr_to_symbolic(value)
            if expr.is_Integer:
                continue
            if not (expr - symbol).is_Integer:
                return False
            inner = region
            while not isinstance(inner, LoopRegion):
                inner = inner.parent_graph
            bounds = (loop_analysis.get_init_assignment(inner), loop_analysis.get_loop_end(inner))
            if inner is loop or any(b is None for b in bounds):
                return False
            if not all(
                data_derived(state.sdfg, {str(s) for s in symbolic.free_symbols_and_functions(b)}) for b in bounds
            ):
                return False
            increments += 1
    return increments > 0


def small_count(count: Any, facts: symbolic.Facts) -> bool:
    """Whether an iteration count is small: :func:`symbolic.is_small_extent`, or provably bounded below
    ``SMALL_CONSTANT_EXTENT`` (a peeled ``Min(0, N - 2) + 1``), or over the thread count alone (a per-thread seam)."""
    if symbolic.is_small_extent(count, facts):
        return True
    if not symbolic.issymbolic(count):
        return False
    if {str(free) for free in sympy.sympify(count).free_symbols} == {symbolic.NUM_THREADS_SYMBOL}:
        return True
    upper = symbolic.numeric_bounds(count)[1]
    if upper is not None and upper < SMALL_CONSTANT_EXTENT:
        return True
    return at_most_one_trip(count, lambda: facts)


def at_most_one_trip(count: Any, facts_of: Callable[[], symbolic.Facts]) -> bool:
    """Whether a map of ``count`` iterations provably runs at most once: no loop at all, but a guard. ``facts_of``
    gives the facts at the map, asked only when the count's numeric bounds do not decide it."""
    if not symbolic.issymbolic(count):
        return count <= 1
    upper = symbolic.numeric_bounds(count)[1]
    if upper is not None and upper <= 1:
        return True
    at_most_one = symbolic.Relation(symbolic.RelationKind.LE, sympy.sympify(count), sympy.Integer(1))
    return symbolic.ask(at_most_one, facts_of()) is symbolic.Truth.TRUE


def privatize_interior_transients(owner: SDFG, state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Give every transient the inside of ``entry``'s scope shares with another scope a descriptor of its own, so
    a parallel schedule allocates it per iteration; ``False`` (and nothing renamed) if one of them is read there
    before the scope writes it.

    Code generation allocates a transient at the scope common to all of its access nodes. Under a sequential
    parent, map fusion's per-element scalars may share one name across two maps of the same body, and are then
    declared once outside both: harmless while each map runs on one thread, a data race once one is shared out
    (``cloudsc_monolith``'s peeled level, ``__map_fusion_zmfdn_0``). A copy that every use in the scope writes
    before reading carries no value in from the other scope, so renaming it there is exact.
    """
    interior = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    names: dict[str, list[nodes.AccessNode]] = {}
    for node in interior:
        if isinstance(node, nodes.AccessNode) and owner.arrays[node.data].transient:
            names.setdefault(node.data, []).append(node)
    inside = {id(node) for node in interior}
    shared = []
    for name, local in names.items():
        if all(id(n) in inside for st in owner.states() for n in st.data_nodes() if n.data == name):
            continue
        written = [n for n in local if any(not e.data.is_empty() for e in state.in_edges(n))]
        if isinstance(owner.arrays[name], data.View) or len(written) != len(local):
            return False
        shared.append((name, local))
    for name, local in shared:
        fresh = owner.add_datadesc(name, copy.deepcopy(owner.arrays[name]), find_new_name=True)
        for node in local:
            node.data = fresh
            for edge in state.all_edges(node):
                if edge.data.data == name:
                    edge.data.data = fresh
    return True


def release_one_trip_maps(sdfg: SDFG) -> int:
    """Make every top-level CPU map of at most one iteration ``Sequential``, and the maps directly in its body
    (or at the top of a nested SDFG there) ``CPU_Multicore``, so schedule inference does not pin them under it.

    Runs BEFORE inference, while canonicalization's maps are still ``Default``: the peeled first vertical level
    of ``cloudsc_monolith`` is such a map, and inferred under it its plane maps ran on one thread. A map whose
    scope reads a shared transient before writing it stays under its parent's schedule
    (:func:`privatize_interior_transients`).

    :param sdfg: the SDFG whose own states are rewritten.
    :returns: how many such maps were found.
    """
    found = 0
    resolver = SymbolResolver()
    for state in sdfg.states():
        children = state.scope_children()
        for wrapper in children[None]:
            if not isinstance(wrapper, nodes.MapEntry) or wrapper.map.schedule != dtypes.ScheduleType.Default:
                continue
            if not at_most_one_trip(wrapper.map.range.num_elements(), lambda state=state: resolver.facts_at(state)):
                continue
            note_sequential(wrapper, ONE_TRIP_REASON)
            found += 1
            body = [(sdfg, state, n) for n in children[wrapper] if isinstance(n, nodes.MapEntry)]
            for nested in [n for n in children[wrapper] if isinstance(n, nodes.NestedSDFG) and n.sdfg is not None]:
                for inner_state in nested.sdfg.states():
                    inner = inner_state.scope_children()[None]
                    body += [(nested.sdfg, inner_state, n) for n in inner if isinstance(n, nodes.MapEntry)]
            for owner, body_state, entry in body:
                if entry.map.schedule == dtypes.ScheduleType.Default and privatize_interior_transients(
                    owner, body_state, entry
                ):
                    entry.map.schedule = dtypes.ScheduleType.CPU_Multicore
    return found


@properties.make_properties
class SequentializeSmallAxes(ppl.Pass):
    """Make every small or nested CPU parallel scope sequential, and share out leading ``SMALL`` axes (module
    docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Sequentialize the small and nested parallel scopes, then share out leading ``SMALL`` axes.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many scopes were pinned or split, or ``None`` if none were.
        """
        self.pinned = 0
        # One pass's worth of enclosing-loop trip-count verdicts: this pass never mutates a
        # LoopRegion's bounds, so a verdict computed under one scope stays valid for a sibling
        # transfer under the same loop.
        self.loop_cache: dict[int, bool] = {}
        self.visit_region(sdfg, False)
        resolver = SymbolResolver()
        candidates = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CPU_Multicore
        ]
        split = sum(
            share_out_past_leading_small_axes(state, entry, resolver) is not None for entry, state in candidates
        )
        return (self.pinned + split) or None

    def visit_region(self, region: ControlFlowRegion, in_parallel: bool) -> None:
        """Walk ``region``'s blocks, threading whether a parallel map encloses them.

        :param region: the control-flow region (or nested SDFG) to walk.
        :param in_parallel: whether a device-parallel map encloses this region.
        """
        for block in region.nodes():
            if isinstance(block, SDFGState):
                self.visit_scope(block, block.scope_children(), None, in_parallel)
            elif isinstance(block, ConditionalBlock):
                # A ConditionalBlock is NOT a ControlFlowRegion, so its branches are only reachable
                # through ``branches``. Everything an ``if`` guards was invisible here otherwise --
                # amg_setup puts its whole body under one, and every map inside it kept a region it
                # could not pay for.
                for _condition, branch in block.branches:
                    self.visit_region(branch, in_parallel)
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, in_parallel)

    def visit_scope(
        self,
        state: SDFGState,
        children: dict[Any, list[nodes.Node]],
        entry: nodes.MapEntry | None,
        in_parallel: bool,
    ) -> None:
        """Decide the nodes directly inside one scope, then descend into their scopes.

        :param state: the state holding the scope.
        :param children: ``state.scope_children()``, built once per state.
        :param entry: the scope's map entry, or ``None`` for the state's top level.
        :param in_parallel: whether a device-parallel map encloses this scope.
        """
        for node in children[entry]:
            if isinstance(node, nodes.MapEntry):
                self.visit_scope(state, children, node, self.decide_map(state, node, in_parallel, children))
            elif isinstance(node, nodes.LibraryNode):
                # A node whose work per entry is the chunk an enclosing loop hands it (FindFirst) opts out of
                # the loop half of the re-entry rule; an enclosing parallel map still serializes it.
                loop_reentry = node.serial_on_loop_reentry
                if node.schedule in CPU_PARALLEL_SCHEDULES and (
                    in_parallel or (loop_reentry and self.loop_serializes(state, node))
                ):
                    node.schedule = dtypes.ScheduleType.Sequential
                    self.pinned += 1
            elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                self.visit_region(node.sdfg, in_parallel)

    def loop_serializes(self, state: SDFGState, node: nodes.LibraryNode) -> bool:
        """Whether the sequential loops around ``node`` make its own region unprofitable. A transfer counts a long
        loop that varies its size (:func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer`); any other
        node is a map over a blackbox, and a loop alone is no reason (rule 3: a data segment it reads)."""
        from dace.libraries.standard.helper import is_reentered_cpu_transfer
        from dace.libraries.standard.nodes.copy import CopyLibraryNode
        from dace.libraries.standard.nodes.fill import FillLibraryNode
        from dace.transformation.auto.auto_optimize import libnode_reads_data_segment  # Avoid an import cycle

        if isinstance(node, (CopyLibraryNode, FillLibraryNode)):
            return is_reentered_cpu_transfer(node, state, loop_cache=self.loop_cache)
        return libnode_reads_data_segment(node, state)

    def decide_map(
        self,
        state: SDFGState,
        node: nodes.MapEntry,
        in_parallel: bool,
        children: dict[Any, list[nodes.Node]] | None = None,
    ) -> bool:
        """Set ``node``'s schedule and report whether its body runs inside a parallel map.

        :param state: the state holding ``node``.
        :param node: the map entry to decide.
        :param in_parallel: whether a device-parallel map encloses this map.
        :param children: ``state.scope_children()``, to count the pinned inner maps the region pays for.
        :returns: whether the map's body is enclosed by a parallel map.
        """
        if node.map.schedule not in CPU_PARALLEL_SCHEDULES:
            return in_parallel
        if in_parallel:
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                note_sequential(node, NESTED_REASON)
                self.pinned += 1
            return True
        count = node.map.range.num_elements() * pinned_inner_trips(node, children or state.scope_children())
        facts = SymbolResolver().facts_at(state, node)
        if small_count(count, facts):
            note_sequential(node, SMALL_REASON.format(count=count))
        elif (
            (loop := innermost_loop(state)) is not None
            and not short_loop(loop, facts)
            and (segment := data_segment(state, node, facts.small)) is not None
        ):
            note_sequential(node, DATA_SEGMENT_REASON.format(begin=segment[0], end=segment[1]))
        else:
            return True
        self.pinned += 1
        return False
