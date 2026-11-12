# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization's one decision on parallel axes: which maps share their iterations out over the OpenMP team,
over which of their axes, and which run sequentially.

Canonicalization leaves the maximally parallel, device-neutral form with every schedule unset (``Default``). This
pass decides every CPU map's schedule, top-down, before schedule inference: the verdict for a map depends on its
body (a few-trip map hands the team to its body maps), so it cannot be made after inference has pinned every
nested map ``Sequential``. A map with an enclosing map runs sequentially, as inference would make it; at the top
level, in order:

1. **A one- or few-trip map.** A map of at most one iteration is a guard rather than a loop; one of fewer
   iterations than a team has threads (:func:`few_trips`) leaves threads idle when a map in its body shares out
   more (:func:`wider`). Either runs sequentially and its body maps become the top-level ones: the peeled first
   vertical level of ``cloudsc_monolith``, and ``newdxx_g``'s map over 5 atoms around its ``ngms`` maps.
2. **A reducing map.** A map whose write-conflict-resolved outputs are indexed by some of its axes shares those
   out, all of them (an OpenMP ``collapse``), and walks the reduced ones in an inner sequential map, so no write
   needs an atomic (tsvc ``s176``: 7.1 s of ``omp atomic`` against numba's 1.0 s; gem's blocks and points). When
   those axes are all ``SMALL`` and a reduced one is not, the split is inverted: the indexing axes become the outer
   sequential loop and the reduced axes the parallel map, whose write becomes an OpenMP ``reduction`` (ICON's
   ``max_abs_vcfl[blk]``).
3. **A small map.** A map whose own iteration count (with the sequential maps it pins) is small
   (:func:`small_count`) runs sequentially.
4. **A data segment under a long loop.** A map whose range starts and ends at values read from data (a CSR row,
   ``indptr[k]:indptr[k + 1]``), or whose extent each outer iteration reads from data (``0:cnt``), inside a
   sequential loop runs a handful of iterations per outer one: a derived ``SMALL`` extent. ilu0's elimination forked
   once per row over a row of about ten: 1.5 s against numba's sequential 11 ms; amg_setup's per-row gather, 9.5 s
   against 0.55 s. A short loop does not count (a level loop of 15 trips, :func:`short_loop`): each of its trips
   reads a whole level, not a handful.

Every other top-level map runs in parallel. An enclosing loop alone is no reason against it: a region entered ``E``
times costs ``E * (fork + work/P)`` against ``E * work``, so ``E`` cancels (npbench ``cavity_flow``'s
1,044,484-element pressure map). A ``CPU_Multicore`` map inside a parallel one (a schedule set before this pass)
runs sequentially: one region opens one OpenMP team level. Last, a parallel map's leading ``SMALL`` axes are shared
out with the first larger one (``collapse(2)`` for ``(b, k, i)`` with ``b`` small), keeping the axis order
(:func:`~dace.transformation.helpers.share_out_past_leading_small_axes`), and a map without a reduced output
collapses its leading axes until they provably fill the team, the innermost axis of a leaf body excepted
(:func:`collapse_leading_axes`).

A copy / memset node here follows :func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer`, as its
implementation selection does (:func:`~dace.libraries.standard.helper.transfer_runs_on_one_core`).
"""

import copy
from collections.abc import Callable
from typing import Any

import sympy

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.symbolic_facts import SMALL_CONSTANT_EXTENT, prove
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import share_out_past_leading_small_axes
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.cpu_specialization.machine import topology

#: Schedules a CPU map or library node still parallelizes under. ``Default`` is included because
#: canonicalization leaves the schedule unset: codegen infers it, and a top-level ``Default`` map
#: becomes ``CPU_Multicore``. Anything else (GPU schedules, ``Unrolled``, ``CPU_Persistent``) is
#: someone else's decision and is left untouched.
CPU_PARALLEL_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)

#: Schedules of a map nested in a CPU map that make it run sequentially: ``Sequential``, and ``Default``, which
#: schedule inference resolves to ``Sequential`` under any CPU map (``dtypes.SCOPEDEFAULT_SCHEDULE``).
SEQUENTIAL_UNDER_A_MAP = (dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential)

#: Iterations per core the leading axes of a parallel map must provably reach before they stop collapsing more axes.
ITERATIONS_PER_THREAD = 4


#: The line a map made sequential here adds under its ``parallel`` class line, so the rendered loop -- which loses
#: its pragma -- says why. Each rule fills in its own reason.
SEQUENTIAL_ON_CPU = "run sequentially on CPU -- {reason}"
NESTED_REASON = "a parallel map encloses it, and a region opens one OpenMP team level only"
SMALL_REASON = "its {count} iterations are small"
ONE_TRIP_REASON = "it runs at most once, so it is a guard rather than a loop"
FEW_TRIPS_REASON = "its {count} iterations leave threads idle, so the maps in its body share out their work instead"
DATA_SEGMENT_REASON = (
    "its range {begin}:{end} is read from data inside a sequential loop, so every outer iteration would fork "
    "over a few elements"
)


def pinned_inner_trips(entry: nodes.MapEntry, children: dict[Any, list[nodes.Node]]) -> Any:
    """The iterations of the deepest chain of sequential maps in ``entry``'s body: ``Sequential`` ones (a reduced
    axis split off) and the unscheduled ones a parallel ``entry`` makes sequential. The region ``entry`` opens pays
    for them."""
    trips = [
        child.map.range.num_elements() * pinned_inner_trips(child, children)
        for child in children[entry]
        if isinstance(child, nodes.MapEntry) and child.map.schedule in SEQUENTIAL_UNDER_A_MAP
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


def few_trips(count: Any, facts_of: Callable[[], symbolic.Facts]) -> bool:
    """Whether a map of ``count`` iterations provably leaves threads of a team idle: a literal below the host's
    physical cores (at least ``SMALL_CONSTANT_EXTENT``), or a small count (:func:`small_count`)."""
    if not symbolic.issymbolic(count):
        return count < max(SMALL_CONSTANT_EXTENT, topology().physical_cores)
    return small_count(count, facts_of())


def wider(count: Any, outer: Any, facts_of: Callable[[], symbolic.Facts]) -> bool:
    """Whether a map of ``count`` iterations inside a map of ``outer`` gives a team more work to share out: not
    small, and more iterations than ``outer`` when both are literals."""
    if not symbolic.issymbolic(count):
        return count >= SMALL_CONSTANT_EXTENT and (symbolic.issymbolic(outer) or count > outer)
    return not small_count(count, facts_of())


def release_few_trips(
    state: SDFGState, wrapper: nodes.MapEntry, children: dict[Any, list[nodes.Node]], resolver: SymbolResolver
) -> bool:
    """Make the unscheduled top-level map ``wrapper`` ``Sequential``, and the maps directly in its body (or at the
    top of a nested SDFG there) ``CPU_Multicore``, when it runs at most one iteration, or fewer than a team has
    threads (:func:`few_trips`) and one of those maps shares out more (:func:`wider`): the team goes to the inner
    maps. A body map whose scope reads a shared transient before writing it stays unscheduled, and so sequential
    (:func:`privatize_interior_transients`).

    :returns: whether ``wrapper`` was released.
    """
    count = wrapper.map.range.num_elements()
    facts_of = lambda: resolver.facts_at(state)
    one_trip = at_most_one_trip(count, facts_of)
    if not one_trip and not few_trips(count, facts_of):
        return False
    # Only unscheduled maps can take the team: one already ``Sequential`` (a split-off reduced axis) cannot
    body = [(state.sdfg, state, n) for n in children[wrapper] if unscheduled_map(n)]
    for nested in [n for n in children[wrapper] if isinstance(n, nodes.NestedSDFG) and n.sdfg is not None]:
        for inner_state in nested.sdfg.states():
            body += [(nested.sdfg, inner_state, n) for n in inner_state.scope_children()[None] if unscheduled_map(n)]
    if not one_trip and not any(wider(e.map.range.num_elements(), count, facts_of) for _, _, e in body):
        return False
    note_sequential(wrapper, ONE_TRIP_REASON if one_trip else FEW_TRIPS_REASON.format(count=count))
    for owner, body_state, entry in body:
        if privatize_interior_transients(owner, body_state, entry):
            entry.map.schedule = dtypes.ScheduleType.CPU_Multicore
    return True


def fills_team(trips: Any, team: int, facts: symbolic.Facts) -> bool:
    """Whether ``trips`` is provably at least ``team``, from the relations alone (no scale assumption)."""
    if not symbolic.issymbolic(trips):
        return trips >= team
    return prove(symbolic.Relation(symbolic.RelationKind.LE, sympy.Integer(team), sympy.sympify(trips)), facts) is (
        symbolic.Truth.TRUE
    )


def collapse_leading_axes(state: SDFGState, entry: nodes.MapEntry, resolver: SymbolResolver) -> bool:
    """Share out the leading axes of a parallel map that writes no write-conflict-resolved output together (an OpenMP
    ``collapse``) until their iterations provably fill ``ITERATIONS_PER_THREAD`` per core: an elementwise map over
    ``(nblocks, 1024, k)`` shares out ``nblocks`` alone otherwise. The innermost axis of a leaf body stays out of the
    collapse, in an inner ``Sequential`` map ``MarkSIMDMaps`` vectorizes; an axis whose range reads an earlier one
    ends the collapse.

    :returns: whether ``entry`` was collapsed.
    """
    from dace.transformation.dataflow import MapExpansion  # Avoid an import cycle
    from dace.transformation.passes.mark_simd_maps import map_body_is_leaf, map_has_wcr

    params, ranges = entry.map.params, entry.map.range
    if entry.map.collapse > 1 or len(params) < 2 or map_has_wcr(state, entry):
        return False
    limit = len(params) - 1 if map_body_is_leaf(state, entry) else len(params)
    team, facts = ITERATIONS_PER_THREAD * topology().physical_cores, resolver.facts_at(state, entry)
    sizes, depth = ranges.size(), 1
    trips = sizes[0]
    while depth < limit and not fills_team(trips, team, facts):
        if set(subsets.Range([ranges[depth]]).free_symbols) & set(params[:depth]):
            break
        trips, depth = trips * sizes[depth], depth + 1
    if depth == 1:
        return False
    if depth < len(params):
        MapExpansion.apply_to(
            state.sdfg,
            map_entry=entry,
            options={"inner_schedule": dtypes.ScheduleType.Sequential, "split_at": depth},
            save=False,
        )
    entry.map.collapse = depth
    return True


def unscheduled_map(node: nodes.Node) -> bool:
    """Whether ``node`` is a map entry whose schedule is not decided yet (``Default``)."""
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.Default


def reduction_output_params(state: SDFGState, entry: nodes.MapEntry) -> set[str] | None:
    """The parameters of ``entry`` that index its write-conflict-resolved outputs.

    ``None`` when there is nothing to split: no such output, outputs indexed by DIFFERENT parameters
    (no one order serves them all), every parameter indexes them (no axis is reduced) or none does
    (a reduction to one element, which OpenMP's ``reduction`` clause already serves), or a data-dependent
    scatter of rows (a dynamic write whose first dimension no parameter indexes, cp2k_trs4's
    ``sink[x0[i0], t0, t1]``): its parallel axis is the one that scatters, which
    :class:`~dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions.RangePrivatizeScatterReductions`
    serves by rows.
    """
    params = set(entry.map.params)
    indexing: set[str] | None = None
    for edge in state.in_edges(state.exit_node(entry)):
        if edge.data.is_empty() or edge.data.wcr is None:
            continue
        begin, end, _ = edge.data.subset.ndrange()[0]
        if edge.data.dynamic and begin != end and not ({str(s) for s in edge.data.subset[0][0].free_symbols} & params):
            return None
        used = {str(symbol) for symbol in edge.data.subset.free_symbols} & params
        if indexing is not None and used != indexing:
            return None
        indexing = used
    if not indexing or indexing == params:
        return None
    return indexing


def small_indexing_axes(state: SDFGState, entry: nodes.MapEntry, indexing: set[str]) -> bool:
    """Whether every axis indexing ``entry``'s reduced outputs is SMALL and some reduced axis is not: sharing the
    indexing axes out would leave the threads idle."""
    facts = SymbolResolver().facts_at(state, entry)
    small = {p: symbolic.is_small_extent(size, facts) for p, size in zip(entry.map.params, entry.map.range.size())}
    return all(small[p] for p in indexing) and any(not small[p] for p in small if p not in indexing)


def share_out_indexing_axes(state: SDFGState, entry: nodes.MapEntry, indexing: set[str]) -> None:
    """Move the ``indexing`` axes of ``entry`` to the front and split the reduced ones off into an inner
    ``Sequential`` map: ``entry`` keeps the indexing axes, collapsed. Inverted when those axes are SMALL
    (:func:`small_indexing_axes`): ``entry`` becomes a ``Sequential`` loop over them around a ``CPU_Multicore`` map
    over the reduced axes (module docstring, rule 2)."""
    from dace.transformation.dataflow import MapDimShuffle, MapExpansion  # Avoid an import cycle

    order = [p for p in entry.map.params if p in indexing] + [p for p in entry.map.params if p not in indexing]
    inverted = small_indexing_axes(state, entry, indexing)
    if order != entry.map.params:
        MapDimShuffle.apply_to(state.sdfg, map_entry=entry, options={"parameters": order}, save=False)
    if inverted:
        MapExpansion.apply_to(
            state.sdfg,
            map_entry=entry,
            options={"inner_schedule": dtypes.ScheduleType.Sequential, "expansion_limit": len(indexing)},
            save=False,
        )
        inner = entry
        for _ in indexing:
            inner.map.schedule = dtypes.ScheduleType.Sequential
            inner = next(e.dst for e in state.out_edges(inner) if isinstance(e.dst, nodes.MapEntry))
        inner.map.schedule = dtypes.ScheduleType.CPU_Multicore
        return
    MapExpansion.apply_to(
        state.sdfg,
        map_entry=entry,
        options={"inner_schedule": dtypes.ScheduleType.Sequential, "split_at": len(indexing)},
        save=False,
    )
    entry.map.collapse = len(indexing)


@properties.make_properties
class DecideCpuParallelAxes(ppl.Pass):
    """Decide the schedule of every CPU map and the axes a parallel one shares out (module docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Decide every CPU map's schedule top-down, then share out leading ``SMALL`` axes.

        :param sdfg: the SDFG to specialize, in place, with its maps' schedules not yet inferred.
        :param _pipeline_results: unused.
        :returns: how many maps were released, split, pinned or collapsed, or ``None`` if none were.
        """
        self.pinned = 0
        self.resolver = SymbolResolver()
        # One pass's worth of enclosing-loop trip-count verdicts: this pass never mutates a
        # LoopRegion's bounds, so a verdict computed under one scope stays valid for a sibling
        # transfer under the same loop.
        self.loop_cache: dict[int, bool] = {}
        self.varying_cache: dict[int, set[str]] = {}
        self.visit_region(sdfg, False, None)
        candidates = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CPU_Multicore
        ]
        resolver = SymbolResolver()
        split = sum(
            share_out_past_leading_small_axes(state, entry, resolver) is not None for entry, state in candidates
        )
        parallel = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CPU_Multicore
        ]
        split += sum(collapse_leading_axes(state, entry, resolver) for entry, state in parallel)
        return (self.pinned + split) or None

    def visit_region(self, region: ControlFlowRegion, in_parallel: bool, enclosing: dtypes.ScheduleType | None) -> None:
        """Walk ``region``'s blocks, threading whether a parallel map encloses them.

        :param region: the control-flow region (or nested SDFG) to walk.
        :param in_parallel: whether a device-parallel map encloses this region.
        :param enclosing: the schedule of the innermost map enclosing this region, or ``None``.
        """
        for block in region.nodes():
            if isinstance(block, SDFGState):
                self.visit_scope(block, block.scope_children(), None, in_parallel, enclosing)
            elif isinstance(block, ConditionalBlock):
                # A ConditionalBlock is NOT a ControlFlowRegion, so its branches are only reachable
                # through ``branches``. Everything an ``if`` guards was invisible here otherwise --
                # amg_setup puts its whole body under one, and every map inside it kept a region it
                # could not pay for.
                for _condition, branch in block.branches:
                    self.visit_region(branch, in_parallel, enclosing)
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, in_parallel, enclosing)

    def visit_scope(
        self,
        state: SDFGState,
        children: dict[Any, list[nodes.Node]],
        entry: nodes.MapEntry | None,
        in_parallel: bool,
        enclosing: dtypes.ScheduleType | None,
    ) -> None:
        """Decide the nodes directly inside one scope, then descend into their scopes.

        :param state: the state holding the scope.
        :param children: ``state.scope_children()``, built once per state and again after a map is split.
        :param entry: the scope's map entry, or ``None`` for the state's top level.
        :param in_parallel: whether a device-parallel map encloses this scope.
        :param enclosing: the schedule of the innermost map enclosing this scope, or ``None``.
        """
        for node in list(children[entry]):
            if isinstance(node, nodes.MapEntry):
                body_in_parallel, reshaped = self.decide_map(state, node, in_parallel, enclosing, children)
                if reshaped:
                    children = state.scope_children()
                # A map left to schedule inference leaves its body to it too
                inner = enclosing if node.map.schedule == dtypes.ScheduleType.Default else node.map.schedule
                self.visit_scope(state, children, node, body_in_parallel, inner)
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
                self.visit_region(node.sdfg, in_parallel, enclosing)

    def loop_serializes(self, state: SDFGState, node: nodes.LibraryNode) -> bool:
        """Whether the sequential loops around ``node`` make its own region unprofitable. A transfer counts a long
        loop that varies its size (:func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer`); any other
        node is a map over a blackbox, and a loop alone is no reason (rule 4: a data segment it reads)."""
        from dace.libraries.standard.helper import is_reentered_cpu_transfer
        from dace.libraries.standard.nodes.copy import CopyLibraryNode
        from dace.libraries.standard.nodes.fill import FillLibraryNode
        from dace.transformation.auto.auto_optimize import libnode_reads_data_segment  # Avoid an import cycle

        if isinstance(node, (CopyLibraryNode, FillLibraryNode)):
            return is_reentered_cpu_transfer(node, state, self.loop_cache, self.varying_cache)
        return libnode_reads_data_segment(node, state)

    def decide_map(
        self,
        state: SDFGState,
        node: nodes.MapEntry,
        in_parallel: bool,
        enclosing: dtypes.ScheduleType | None,
        children: dict[Any, list[nodes.Node]],
    ) -> tuple[bool, bool]:
        """Set ``node``'s schedule, splitting off the axes it reduces.

        :param state: the state holding ``node``.
        :param node: the map entry to decide.
        :param in_parallel: whether a device-parallel map encloses this map.
        :param enclosing: the schedule of the innermost map enclosing this one, or ``None``.
        :param children: ``state.scope_children()``.
        :returns: whether the map's body runs inside a parallel map, and whether the map was split (``children``
            is then stale).
        """
        schedule = node.map.schedule
        if schedule == dtypes.ScheduleType.Default and enclosing is not None:
            # Another schedule (a GPU kernel, ``Unrolled``) leaves its inner maps to schedule inference
            if enclosing in SEQUENTIAL_UNDER_A_MAP + CPU_PARALLEL_SCHEDULES:
                node.map.schedule = dtypes.ScheduleType.Sequential
            return in_parallel, False
        if schedule not in CPU_PARALLEL_SCHEDULES:
            return in_parallel, False
        if in_parallel:
            note_sequential(node, NESTED_REASON)
            self.pinned += 1
            return True, False
        # Only the root SDFG's own top-level maps: a released one hands the team to its body
        if (
            schedule == dtypes.ScheduleType.Default
            and state.sdfg.parent_nsdfg_node is None
            and release_few_trips(state, node, children, self.resolver)
        ):
            self.pinned += 1
            return False, False
        reshaped = False
        if node.map.collapse <= 1 and (indexing := reduction_output_params(state, node)) is not None:
            share_out_indexing_axes(state, node, indexing)
            self.pinned += 1
            reshaped = True
            children = state.scope_children()
            if node.map.schedule == dtypes.ScheduleType.Sequential:
                return False, True
        count = node.map.range.num_elements() * pinned_inner_trips(node, children)
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
            node.map.schedule = dtypes.ScheduleType.CPU_Multicore
            return True, reshaped
        self.pinned += 1
        return False, reshaped
