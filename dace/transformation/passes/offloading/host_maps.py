# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that stay on the host so the maps under them become the kernels (ICON's ``nblks`` over ``nproma``/``nlev``)."""

import itertools
from typing import Any
from collections.abc import Iterable

import sympy

from ordered_set import OrderedSet

from dace import subsets, symbolic
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers

#: ``False``/``None``/``[]``: no host maps; ``True``: derive them; a list: these map labels or ``MapEntry`` nodes.
HostMapSpec = bool | list[str | nodes.MapEntry] | None


def only_launches(children: Iterable[nodes.Node]) -> bool:
    """None of ``children`` computes (a tasklet or library node), and at least one launches: a map or a nested SDFG
    that only launches."""
    launches = False
    for node in children:
        if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
            return False
        if isinstance(node, nodes.NestedSDFG) and not only_launches(
            itertools.chain.from_iterable(state.scope_children()[None] for state in node.sdfg.states())
        ):
            return False
        launches = launches or isinstance(node, (nodes.MapEntry, nodes.NestedSDFG))
    return launches


def body_extents_depend_on_entry(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """An inner map whose extent names one of ``entry``'s parameters, which a host launch cannot pass
    (npbench correlation's ``dim3(((M - __i) - 1), 1, 1)``). Refused even for a map the caller named."""
    params = OrderedSet(entry.map.params)
    for node in state.scope_children()[entry]:
        if not isinstance(node, nodes.MapEntry):
            continue
        # By name: two symbols that share a name but not their assumptions are one parameter.
        extent_names = OrderedSet(
            name for rng in node.map.range for bound in rng for name in symbolic.free_symbols_and_functions(bound)
        )
        if params & extent_names or body_extents_depend_on_entry(state, node):
            return True
    return False


def is_host_map(
    state: SDFGState,
    entry: nodes.MapEntry,
    auto: bool,
    pinned_labels: OrderedSet[str],
    pinned_entries: OrderedSet[nodes.MapEntry],
    callbacks: OrderedSet[str],
) -> bool:
    """A named map, a map holding a callback, or with ``auto`` a map that only launches, stays on the host."""
    named = entry in pinned_entries or entry.map.label in pinned_labels
    if (named or auto) and body_extents_depend_on_entry(state, entry):
        return False
    if named:
        return True
    # A kernel cannot issue a callback, so this is a requirement and not gated on ``auto``.
    if helpers.scope_holds_callback(state, entry, callbacks):
        return True
    # Nor a host-issued library call: the map around it launches it from the host.
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
    if any(helpers.holds_device_wide_libnode(node) for node in scope):
        return True
    return auto and only_launches(state.scope_children()[entry])


def find_host_maps(sdfg: SDFG, spec: HostMapSpec = False) -> OrderedSet[nodes.MapEntry]:
    """The map entries in ``sdfg`` that must keep a host schedule.

    :param sdfg: the SDFG to scan, nested SDFGs included.
    :param spec: see :data:`HostMapSpec`.
    :return: the map entries to leave on the host, in a deterministic order. A map holding a callback
        is among them whatever ``spec`` says.
    """
    pinned_labels: OrderedSet[str] = OrderedSet()
    pinned_entries: OrderedSet[nodes.MapEntry] = OrderedSet()
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

    callbacks = helpers.callback_symbol_names(sdfg)
    return OrderedSet(
        node
        for nested in sdfg.all_sdfgs_recursive()
        for state in nested.states()
        for node in state.nodes()
        if isinstance(node, nodes.MapEntry)
        and is_host_map(state, node, spec is True, pinned_labels, pinned_entries, callbacks)
    )


def host_code_containers(sdfg: SDFG, region: ControlFlowRegion) -> OrderedSet[str]:
    """Containers touched by top-level tasklets, interstate edges and loop or branch conditions under ``region``."""
    touched: OrderedSet[str] = OrderedSet()
    for state in region.all_states():
        scopes = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and scopes[node] is None:
                touched |= OrderedSet(e.data.data for e in state.all_edges(node) if not e.data.is_empty())
    for edge in region.all_interstate_edges(recursive=True):
        touched |= OrderedSet(m.data for m in edge.data.get_read_memlets(sdfg.arrays))
    for block in itertools.chain([region], region.all_control_flow_blocks(recursive=True)):
        if isinstance(block, (LoopRegion, ConditionalBlock)):
            touched |= OrderedSet(m.data for m in block.get_meta_read_memlets())
    return touched


def map_containers_and_traffic(state: SDFGState, entry: nodes.MapEntry) -> tuple[OrderedSet[str], Any]:
    """The containers a top-level map reads or writes, and the elements it moves (dynamic memlets count 0)."""
    names: OrderedSet[str] = OrderedSet()
    traffic = 0
    for edge in itertools.chain(state.in_edges(entry), state.out_edges(state.exit_node(entry))):
        if edge.data.is_empty():
            continue
        names.add(edge.data.data)
        if not edge.data.dynamic:
            traffic = traffic + edge.data.volume
    return names, traffic


def provably_nonnegative(expr: Any) -> bool:
    """``expr >= 0`` for every nonnegative value of its symbols, also when read as a polynomial with
    nonnegative coefficients (SymPy leaves a sum with a negative term open); undecided is False."""
    expr = sympy.expand(subsets.nng(sympy.sympify(expr)))
    if expr.is_nonnegative:
        return True
    if not expr.free_symbols:
        return False
    try:
        polynomial = sympy.Poly(expr, *expr.free_symbols)
    except sympy.PolynomialError:
        return False
    return all(coefficient.is_number and coefficient >= 0 for coefficient in polynomial.coeffs())


def provably_moves_less(traffic: Any, size: Any) -> bool:
    """``traffic`` provably below ``size``; a constant is below any symbolic size; undecided is False."""
    if not symbolic.issymbolic(traffic):
        return symbolic.issymbolic(size) or provably_nonnegative(size - traffic - 1)
    return provably_nonnegative(size - traffic) and not provably_nonnegative(traffic - size)


def pinnable_maps(sdfg: SDFG, loop: LoopRegion) -> OrderedSet | None:
    """The maps of ``loop`` to keep on the host, or None when ``loop`` is a device loop."""
    host = host_code_containers(sdfg, loop)
    candidates: OrderedSet = OrderedSet()
    for state in loop.all_states():
        work = [n for n in state.scope_children()[None] if isinstance(n, (nodes.MapEntry, nodes.LibraryNode))]
        for node in work:
            if len(work) != 1 or not isinstance(node, nodes.MapEntry):
                return None
            names, traffic = map_containers_and_traffic(state, node)
            shared = [n for n in names & host if n in sdfg.arrays and sdfg.arrays[n].total_size != 1]
            if not shared or not provably_moves_less(traffic, sum(sdfg.arrays[n].total_size for n in shared)):
                return None
            candidates.add(node)
    return candidates


def maps_pinned_by_host_loops(sdfg: SDFG) -> OrderedSet:
    """Top-level maps of a serial host loop kept on the host with their subtree: offloaded, each iteration would
    copy the containers they share with the loop's host code (amg_setup: 344 s on the GPU, 0.8 s on the CPU).

    All or nothing per loop, since a state has one location: any other device work keeps every kernel.
    """
    pinned: OrderedSet = OrderedSet()
    for loop in sdfg.all_control_flow_regions():
        if isinstance(loop, LoopRegion):
            pinned |= pinnable_maps(sdfg, loop) or OrderedSet()
    return pinned
