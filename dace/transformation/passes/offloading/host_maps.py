# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that stay on the HOST; the maps under them become the kernels.

The offloading otherwise makes every top-level map a ``GPU_Device`` kernel. That is wrong for a map
whose purpose is to LAUNCH work rather than do it -- ICON's shape, an ``nblks`` map over one nested
SDFG of ``nproma``/``nlev`` maps -- Which maps those are is named by the caller, or derived structurally; see :func:`host_maps`.
"""

import itertools

import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace import symbolic
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.transformation.passes.offloading.taskloop import is_computation, sdfg_only_launches

#: What a caller may pass as ``host_maps``:
#:
#: * ``False`` (the default) -- run no host-map detection at all.
#: * ``None`` or ``[]`` -- name no host maps; the same outcome, spelled for a caller that computes
#:   the list and finds it empty.
#: * ``True`` -- derive them with the built-in heuristics.
#: * a list -- exactly these maps, each given as a map label or as the ``MapEntry`` itself.
HostMapSpec = bool | list[str | nodes.MapEntry] | None


def body_extents_depend_on_entry(entry: nodes.MapEntry, scope_children: dict) -> bool:
    """An inner map whose extent mentions one of ``entry``'s own parameters.

    Keeping such a map on the host is not a trade-off, it is broken: the extent has to reach the
    launch configuration, where an outer parameter is not in scope. npbench correlation's
    ``symmetrize_col(j: _[i + 1:M])`` emits ``dim3(((M - __i) - 1), 1, 1)`` and nvcc rejects the
    program outright. So this refuses a map even when the caller named it.

    The ICON shape this rule exists beside is unaffected: an ``nblks`` map over ``nproma``/``nlev``
    bodies has extents that do not mention the block index.
    """
    params = OrderedSet(entry.map.params)
    for node in scope_children.get(entry, ()):
        if isinstance(node, nodes.MapEntry):
            # By NAME, through symbolic's own reader: the question is whether the inner extent
            # mentions this map's parameter, which is a naming question. Comparing symbol objects
            # would answer it differently for two symbols that share a name but not their
            # assumptions, and either answer would be about the wrong thing.
            extent_names: OrderedSet = OrderedSet()
            for rng in node.map.range:
                for bound in rng:
                    extent_names |= OrderedSet(symbolic.free_symbols_and_functions(bound))
            if params & extent_names:
                return True
            if body_extents_depend_on_entry(node, scope_children):
                return True
    return False


def only_launches(state: SDFGState, entry: nodes.MapEntry, scope_children: dict) -> bool:
    """``entry``'s scope launches work rather than doing any itself."""
    launches = False
    for node in scope_children.get(entry, ()):
        if is_computation(node):
            return False
        if isinstance(node, nodes.MapEntry):
            launches = True
        elif isinstance(node, nodes.NestedSDFG):
            if not sdfg_only_launches(node.sdfg):
                return False
            launches = True
    return launches


def is_host_map(
    state: SDFGState,
    entry: nodes.MapEntry,
    scope_children: dict,
    auto: bool,
    pinned_labels: OrderedSet,
    pinned_entries: OrderedSet,
    sdfg: SDFG = None,
    callback_names: OrderedSet | None = None,
) -> bool:
    """``entry`` belongs on the host, so the maps under it become the kernels.

    A map the caller NAMED is a host map whatever the structure looks like -- a caller who names a
    map has looked at the kernel and these rules have not -- except where the lowering could not be
    emitted at all (:func:`body_extents_depend_on_entry`).

    ``auto`` adds the only other reason a map is kept on the host: a scope that launches work
    rather than doing any of its own. ``callback_names`` is ``sdfg``'s
    :func:`~dace.transformation.passes.offloading.offloading_helpers.callback_symbol_names`.
    """
    named = entry in pinned_entries or entry.map.label in pinned_labels
    if named or auto:
        if body_extents_depend_on_entry(entry, scope_children):
            return False
    if named:
        return True
    # A callback is host code whatever the shape around it: a Python callback needs the interpreter,
    # and a GPU callback is itself a launch, so a kernel cannot issue one. Not gated on ``auto`` --
    # a kernel around one would not run at all, which makes this a requirement rather than a
    # preference, and no default behaviour depends on offloading one.
    if sdfg is not None and helpers.scope_holds_callback(state, entry, scope_children, sdfg, callback_names):
        return True
    if not auto:
        return False
    return only_launches(state, entry, scope_children)


def host_maps(sdfg: SDFG, spec: HostMapSpec = False) -> OrderedSet:
    """The map entries in ``sdfg`` that must keep a host schedule.

    :param sdfg: the SDFG to scan, nested SDFGs included.
    :param spec: see :data:`HostMapSpec`. ``False`` (the default), ``None`` and ``[]`` all name no
        host maps and run no heuristics; ``True`` derives them; a list names them outright.
    :return: the map entries to leave on the host, in a deterministic order.

    A map holding a callback is returned whatever ``spec`` says: a kernel cannot issue one, so that
    is a requirement rather than a scheduling opinion.
    """
    auto = spec is True
    pinned_labels: OrderedSet = OrderedSet()
    pinned_entries: OrderedSet = OrderedSet()
    if isinstance(spec, (list, tuple, OrderedSet)):
        for item in spec:
            if isinstance(item, nodes.MapEntry):
                pinned_entries.add(item)
            elif isinstance(item, str):
                pinned_labels.add(item)
            else:
                raise TypeError(
                    f"host_maps takes map labels or MapEntry nodes, got {item!r} of type {type(item).__name__}"
                )
    elif spec not in (None, True, False):
        raise TypeError(
            f"host_maps must be None, a bool or a list of labels / MapEntry nodes, got {type(spec).__name__}"
        )

    found: OrderedSet = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        # Once per SDFG: asked per tasklet, the callback test walks every nested symbol table.
        callback_names = helpers.callback_symbol_names(nested)
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_host_map(
                    state, node, scope_children, auto, pinned_labels, pinned_entries, nested, callback_names
                ):
                    found.add(node)
    return found


def host_code_containers(sdfg: SDFG, region: ControlFlowRegion) -> OrderedSet:
    """Containers the host code under ``region`` touches: tasklets outside every map, interstate
    edges, and the conditions of its loops and branches."""
    touched: OrderedSet = OrderedSet()
    for state in region.states():
        scopes = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and scopes[node] is None:
                touched |= OrderedSet(e.data.data for e in state.all_edges(node) if e.data.data is not None)
    for edge in region.all_interstate_edges(recursive=True):
        touched |= OrderedSet(m.data for m in edge.data.get_read_memlets(sdfg.arrays))
    for block in itertools.chain([region], region.all_control_flow_blocks(recursive=True)):
        if isinstance(block, (LoopRegion, ConditionalBlock)):
            touched |= OrderedSet(m.data for m in block.get_meta_read_memlets())
    return touched


def map_containers_and_traffic(
    state: SDFGState, entry: nodes.MapEntry
) -> tuple[OrderedSet[str], symbolic.SymbolicType | int]:
    """The containers a top-level map reads or writes, and the elements it moves (dynamic memlets count 0)."""
    names: OrderedSet[str] = OrderedSet()
    traffic: symbolic.SymbolicType | int = 0
    exit_node = state.exit_node(entry)
    for edge in itertools.chain(state.in_edges(entry), state.out_edges(exit_node)):
        if edge.data.data is None:
            continue
        names.add(edge.data.data)
        if not edge.data.dynamic:
            inner = (
                state.out_edges_by_connector(entry, "OUT_" + edge.dst_conn[3:])
                if edge.dst is entry and edge.dst_conn
                else state.in_edges_by_connector(exit_node, "IN_" + edge.src_conn[4:])
                if edge.src is exit_node and edge.src_conn
                else []
            )
            per_iteration = nest_accesses(list(inner))
            if per_iteration is None:
                traffic = traffic + moved_elements(edge.data)
            else:
                traffic = traffic + entry.map.range.num_elements() * per_iteration
    return names, traffic


def nest_accesses(inner_edges: list) -> symbolic.SymbolicType | int | None:
    """Elements one map iteration moves through ``inner_edges`` when each ends at a loop-free nested SDFG: the
    accesses its states and interstate edges make to the connector, which the memlet into the nest overstates
    when an interstate edge reads it (amg_setup's ``c = touched[t]`` widens to all of ``touched``). None otherwise."""
    if not inner_edges:
        return None
    count: symbolic.SymbolicType | int = 0
    for edge in inner_edges:
        nsdfg = edge.dst if isinstance(edge.dst, nodes.NestedSDFG) else edge.src
        conn = edge.dst_conn if nsdfg is edge.dst else edge.src_conn
        if not isinstance(nsdfg, nodes.NestedSDFG) or conn is None:
            return None
        inner = nsdfg.sdfg
        if any(isinstance(block, LoopRegion) for block in inner.all_control_flow_blocks(recursive=True)):
            return None
        for body in inner.states():
            for node in body.data_nodes():
                if node.data != conn:
                    continue
                for e in body.all_edges(node):
                    if e.data.data != conn:
                        continue
                    if e.data.dynamic:
                        return None
                    count = count + e.data.volume
        for ise in inner.all_interstate_edges(recursive=True):
            count = count + sum(m.volume for m in ise.data.get_read_memlets(inner.arrays) if m.data == conn)
    return count


def moved_elements(memlet) -> symbolic.SymbolicType | int:
    """The elements ``memlet`` moves: its volume, or its subset's size when the volume is that size times a factor
    of the map's own -- the propagation of a read at an index it cannot see (ilu0's ``col_j = A_indices[m]`` per
    iteration reads as ``nnz * len``), which can touch no more than the subset holds."""
    elements = memlet.subset.num_elements()
    if not symbolic.issymbolic(elements):
        return memlet.volume
    factor = symbolic.simplify(memlet.volume / elements)
    sizes = {str(sym) for sym in elements.free_symbols}
    if symbolic.simplify(factor - 1) != 0 and not {str(sym) for sym in factor.free_symbols} & sizes:
        return elements
    return memlet.volume


def provably_moves_less(
    traffic: symbolic.SymbolicType | int, size: symbolic.SymbolicType | int, facts: symbolic.Facts
) -> bool:
    """Whether ``traffic`` is provably below ``size`` under ``facts`` (at the map); undecidable answers False, keeping
    the map a kernel.

    A symbolic extent is taken as big enough, so a constant count is below any symbolic ``size``.
    """
    if not symbolic.issymbolic(traffic):
        return symbolic.issymbolic(size) or symbolic.provably_nonnegative(size - traffic - 1, symbolic.Facts.none())
    return symbolic.provably_nonnegative(size - traffic, facts) and not symbolic.provably_nonnegative(
        traffic - size, facts
    )


def stays_with_the_loop(
    traffic: symbolic.SymbolicType | int, size: symbolic.SymbolicType | int, facts: symbolic.Facts, assigned: set[str]
) -> bool:
    """Whether a map moving ``traffic`` beside host code sharing ``size`` elements stays on the host: provably less,
    or not provably at least and sized by a symbol the loop assigns. A per-iteration extent is the row a serial
    algorithm works on (amg_setup's ``ntouch`` touched columns), small beside the arrays it copies; a program-wide
    one no rule ranks is taken as big."""
    if provably_moves_less(traffic, size, facts):
        return True
    named = (
        {str(sym) for sym in symbolic.pystr_to_symbolic(traffic).free_symbols}
        if symbolic.issymbolic(traffic)
        else set()
    )
    return bool(named & assigned) and not symbolic.provably_nonnegative(traffic - size, facts)


def is_fallback_loop(loop: LoopRegion) -> bool:
    """Whether ``loop`` is a guarded specialization's sequential fallback: pinned sequential at the top
    level of a conditional's branch. A carried-dependence or wavefront pin elsewhere is no fallback."""
    branch = loop.parent_graph
    return (
        loop.pinned_sequential
        and isinstance(branch, ControlFlowRegion)
        and isinstance(branch.parent_graph, ConditionalBlock)
    )


def in_fallback_loop(loop: LoopRegion) -> bool:
    """Whether ``loop`` is, or lies inside, a loop pinned sequential as a specialization's fallback."""
    region = loop
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion) and is_fallback_loop(region):
            return True
        region = region.parent_graph
    return False


def maps_pinned_by_host_loops(sdfg: SDFG) -> OrderedSet:
    """Top-level maps of a serial host loop, kept on the host with their whole subtree.

    A specialization's sequential fallback loop is the conflict path, host code by construction:
    every map in it is pinned, except in a state that also calls a library (a state has one
    location), so the arm copies only at its own boundary. In any other loop, offloading a map
    copies each non-scalar container it shares with the loop's host code once per iteration, and
    when the map moves less than those hold the copies dominate (:func:`stays_with_the_loop`) (amg_setup: 344 s on GPU
    against 0.8 s on CPU). Such a loop is pinned all or nothing: any other device work in it (a map
    not provably smaller, a library call, a second map in a state) keeps every kernel on the device,
    since the offload gives a state one location. Loops of nested SDFGs are kernel code.
    """
    pinned: OrderedSet = OrderedSet()
    for loop in sdfg.all_control_flow_regions():
        if not isinstance(loop, LoopRegion):
            continue
        if is_fallback_loop(loop):
            for state in loop.states():
                top = state.scope_children()[None]
                if not any(isinstance(n, nodes.LibraryNode) for n in top):
                    pinned |= OrderedSet(n for n in top if isinstance(n, nodes.MapEntry))
            continue
        if in_fallback_loop(loop):
            continue
        host = host_code_containers(sdfg, loop)
        assigned = {lhs for edge in loop.all_interstate_edges(recursive=True) for lhs in edge.data.assignments}
        written = {node.data for state in loop.states() for node in state.data_nodes() if state.in_degree(node) > 0}
        candidates: OrderedSet = OrderedSet()
        device_loop = False
        for state in loop.states():
            work = [n for n in state.scope_children()[None] if isinstance(n, (nodes.MapEntry, nodes.LibraryNode))]
            for node in work:
                if len(work) != 1 or not isinstance(node, nodes.MapEntry):
                    device_loop = True
                    continue
                names, traffic = map_containers_and_traffic(state, node)
                # Read on both sides only, a container is copied once at the entry, not per iteration
                shared = [n for n in names & host & written if n in sdfg.arrays and sdfg.arrays[n].total_size != 1]
                total = sum(sdfg.arrays[n].total_size for n in shared)
                if shared and stays_with_the_loop(traffic, total, SymbolResolver().facts_at(state, node), assigned):
                    candidates.add(node)
                else:
                    device_loop = True
        if not device_loop:
            pinned |= candidates
    return pinned
