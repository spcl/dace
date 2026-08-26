# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that only LAUNCH work stay on the host; their bodies become the kernels.
ICON's shape: an ``nblks`` map over one nested SDFG of ``nproma``/``nlev`` maps."""
from ordered_set import OrderedSet

from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState


def is_computation(node: nodes.Node) -> bool:
    """Only these compute: access nodes stage, map scopes and nested SDFGs launch, and interstate
    edges and control-flow blocks prepare symbols, which is why neither is ever looked at."""
    return isinstance(node, (nodes.Tasklet, nodes.LibraryNode))


def is_copy_or_fill_libnode(node: nodes.Node) -> bool:
    """Moves or initializes only; a type test, since nothing carries GPU storage yet."""
    from dace.libraries.standard.nodes.copy import CopyLibraryNode
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    return isinstance(node, (CopyLibraryNode, FillLibraryNode))


def encloses_device_wide_libnode(state: SDFGState, entry: nodes.MapEntry, scope_children: dict) -> bool:
    """Its scope holds a library node expanding to a call only host code can issue."""
    for node in scope_children.get(entry, ()):
        if isinstance(node, nodes.LibraryNode) and not is_copy_or_fill_libnode(node):
            return True
        if isinstance(node, nodes.MapEntry) and encloses_device_wide_libnode(state, node, scope_children):
            return True
        if isinstance(node, nodes.NestedSDFG) and sdfg_holds_device_wide_libnode(node.sdfg):
            return True
    return False


def sdfg_holds_device_wide_libnode(sdfg: SDFG) -> bool:
    return any(
        isinstance(node, nodes.LibraryNode) and not is_copy_or_fill_libnode(node)
        for node, _ in sdfg.all_nodes_recursive())


def sdfg_only_launches(sdfg: SDFG) -> bool:
    """Every state computes only inside maps, and there is at least one; ``states()`` recurses
    through regions and never yields an interstate edge."""
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


def body_extents_depend_on_entry(entry: nodes.MapEntry, scope_children: dict) -> bool:
    """An inner map whose extent mentions one of ``entry``'s own parameters.

    Keeping such a map on the host is a loss twice over. Every host iteration then launches a
    DIFFERENT, mostly smaller kernel, so the nest pays a launch per iteration for work that shrinks
    as it goes -- measured at 1.12x on polybench cholesky and 1.14x on npbench cholesky2, against a
    5% floor read off the kernels whose two arms compile the same graph. And the extent has to reach
    the launch configuration, where an outer parameter is not in scope: correlation's
    ``symmetrize_col(j: _[i + 1:M])`` emits ``dim3(((M - __i) - 1), 1, 1)`` and nvcc rejects the
    program outright.

    The ICON shape this rule exists for is unaffected: an ``nblks`` map over ``nproma``/``nlev``
    bodies has extents that do not mention the block index.
    """
    params = OrderedSet(entry.map.params)
    for node in scope_children.get(entry, ()):
        if isinstance(node, nodes.MapEntry):
            if params & OrderedSet(str(s) for s in node.map.range.free_symbols):
                return True
            if body_extents_depend_on_entry(node, scope_children):
                return True
    return False


def is_taskloop_map(state: SDFGState, entry: nodes.MapEntry, scope_children: dict, launch_only: bool = True) -> bool:
    """``entry`` launches work rather than doing it, so it belongs on the host.

    Enclosing a device-wide library node is a requirement, not a heuristic: a cuBLAS call is issued by
    host code. ``launch_only`` adds the optional reason -- an uncollapsed nest counts, and collapsing
    it into one kernel is canonicalization's job -- but only for a body whose extents do not depend
    on this map, see :func:`body_extents_depend_on_entry`.
    """
    if encloses_device_wide_libnode(state, entry, scope_children):
        return True
    if not launch_only:
        return False
    if body_extents_depend_on_entry(entry, scope_children):
        return False

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


def taskloop_maps(sdfg: SDFG, launch_only: bool = True) -> OrderedSet:
    found = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_taskloop_map(state, node, scope_children, launch_only):
                    found.add(node)
    return found
