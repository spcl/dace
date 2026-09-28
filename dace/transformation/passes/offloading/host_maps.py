# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that stay on the host so the maps under them become the kernels (ICON's ``nblks`` over ``nproma``/``nlev``)."""
import itertools
from typing import Dict, Iterable, List, Optional, Union

from ordered_set import OrderedSet

from dace import symbolic
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers

#: ``False``/``None``/``[]``: no host maps; ``True``: derive them; a list: these map labels or ``MapEntry`` nodes.
HostMapSpec = Optional[Union[bool, List[Union[str, nodes.MapEntry]]]]


def is_computation(node: nodes.Node) -> bool:
    """Access nodes stage and scopes launch; only tasklets and library nodes compute."""
    return isinstance(node, (nodes.Tasklet, nodes.LibraryNode))


def only_launches(children: Iterable[nodes.Node]) -> bool:
    """None of ``children`` computes, and at least one launches: a map or a nested SDFG that only launches."""
    launches = False
    for node in children:
        if is_computation(node):
            return False
        if isinstance(node, nodes.NestedSDFG) and not sdfg_only_launches(node.sdfg):
            return False
        launches = launches or isinstance(node, (nodes.MapEntry, nodes.NestedSDFG))
    return launches


def sdfg_only_launches(sdfg: SDFG) -> bool:
    return only_launches(itertools.chain.from_iterable(state.scope_children()[None] for state in sdfg.states()))


def body_extents_depend_on_entry(entry: nodes.MapEntry, scope_children: Dict) -> bool:
    """An inner map whose extent names one of ``entry``'s parameters, which a host launch cannot pass
    (npbench correlation's ``dim3(((M - __i) - 1), 1, 1)``). Refused even for a map the caller named."""
    params = OrderedSet(entry.map.params)
    for node in scope_children.get(entry, ()):
        if isinstance(node, nodes.MapEntry):
            # By name: two symbols that share a name but not their assumptions are one parameter.
            extent_names: OrderedSet = OrderedSet()
            for rng in node.map.range:
                for bound in rng:
                    extent_names |= OrderedSet(symbolic.free_symbols_and_functions(bound))
            if params & extent_names:
                return True
            if body_extents_depend_on_entry(node, scope_children):
                return True
    return False


def is_host_map(state: SDFGState,
                entry: nodes.MapEntry,
                scope_children: Dict,
                auto: bool,
                pinned_labels: OrderedSet,
                pinned_entries: OrderedSet,
                sdfg: SDFG = None,
                callback_names: Optional[OrderedSet] = None) -> bool:
    """A named map, a map holding a callback, or with ``auto`` a map that only launches, stays on the host."""
    named = entry in pinned_entries or entry.map.label in pinned_labels
    if named or auto:
        if body_extents_depend_on_entry(entry, scope_children):
            return False
    if named:
        return True
    # A kernel cannot issue a callback, so this is a requirement and not gated on ``auto``.
    if sdfg is not None and helpers.scope_holds_callback(state, entry, scope_children, sdfg, callback_names):
        return True
    if not auto:
        return False
    return only_launches(scope_children.get(entry, ()))


def host_maps(sdfg: SDFG, spec: HostMapSpec = False) -> OrderedSet:
    """The map entries in ``sdfg`` that must keep a host schedule.

    :param sdfg: the SDFG to scan, nested SDFGs included.
    :param spec: see :data:`HostMapSpec`.
    :return: the map entries to leave on the host, in a deterministic order. A map holding a callback
        is among them whatever ``spec`` says.
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
                raise TypeError(f"host_maps takes map labels or MapEntry nodes, got {item!r} "
                                f"of type {type(item).__name__}")
    elif spec not in (None, True, False):
        raise TypeError(f"host_maps must be None, a bool or a list of labels / MapEntry nodes, "
                        f"got {type(spec).__name__}")

    found: OrderedSet = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        callback_names = helpers.callback_symbol_names(nested)
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_host_map(state, node, scope_children, auto, pinned_labels,
                                                                    pinned_entries, nested, callback_names):
                    found.add(node)
    return found
