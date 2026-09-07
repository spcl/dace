# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that stay on the HOST; the maps under them become the kernels.

The offloading otherwise makes every top-level map a ``GPU_Device`` kernel. That is wrong for a map
whose purpose is to LAUNCH work rather than do it -- ICON's shape, an ``nblks`` map over one nested
SDFG of ``nproma``/``nlev`` maps -- Which maps those are is named by the caller, or derived structurally; see :func:`host_maps`.
"""
from typing import Dict, List, Optional, Union

from dace.ordered import OrderedSet

from dace import symbolic
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers

#: What a caller may pass as ``host_maps``:
#:
#: * ``False`` (the default) -- run no host-map detection at all.
#: * ``None`` or ``[]`` -- name no host maps; the same outcome, spelled for a caller that computes
#:   the list and finds it empty.
#: * ``True`` -- derive them with the built-in heuristics.
#: * a list -- exactly these maps, each given as a map label or as the ``MapEntry`` itself.
HostMapSpec = Optional[Union[bool, List[Union[str, nodes.MapEntry]]]]


def is_computation(node: nodes.Node) -> bool:
    """Only these compute: access nodes stage, map scopes and nested SDFGs launch, and interstate
    edges and control-flow blocks prepare symbols, which is why neither is ever looked at."""
    return isinstance(node, (nodes.Tasklet, nodes.LibraryNode))


def sdfg_only_launches(sdfg: SDFG) -> bool:
    """Every state computes only inside maps, and there is at least one.

    ``states()`` recurses through regions and never yields an interstate edge.
    """
    found_map = False
    for state in sdfg.states():
        for node in state.scope_children()[None]:
            if is_computation(node):
                return False
            if isinstance(node, nodes.MapEntry):
                found_map = True
            elif isinstance(node, nodes.NestedSDFG):
                if not sdfg_only_launches(node.sdfg):
                    return False
                found_map = True
    return found_map


def body_extents_depend_on_entry(entry: nodes.MapEntry, scope_children: Dict) -> bool:
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


def only_launches(state: SDFGState, entry: nodes.MapEntry, scope_children: Dict) -> bool:
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


def is_host_map(state: SDFGState,
                entry: nodes.MapEntry,
                scope_children: Dict,
                auto: bool,
                pinned_labels: OrderedSet,
                pinned_entries: OrderedSet,
                sdfg: SDFG = None) -> bool:
    """``entry`` belongs on the host, so the maps under it become the kernels.

    A map the caller NAMED is a host map whatever the structure looks like -- a caller who names a
    map has looked at the kernel and these rules have not -- except where the lowering could not be
    emitted at all (:func:`body_extents_depend_on_entry`).

    ``auto`` adds the only other reason a map is kept on the host: a scope that launches work
    rather than doing any of its own.
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
    if sdfg is not None and helpers.scope_holds_callback(state, entry, scope_children, sdfg):
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
                raise TypeError(f"host_maps takes map labels or MapEntry nodes, got {item!r} "
                                f"of type {type(item).__name__}")
    elif spec not in (None, True, False):
        raise TypeError(f"host_maps must be None, a bool or a list of labels / MapEntry nodes, "
                        f"got {type(spec).__name__}")

    found: OrderedSet = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_host_map(state, node, scope_children, auto, pinned_labels,
                                                                    pinned_entries, nested):
                    found.add(node)
    return found
