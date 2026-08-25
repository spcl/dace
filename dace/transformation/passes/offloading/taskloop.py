# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that only LAUNCH work stay on the host; their bodies become the kernels.
ICON's shape: an ``nblks`` map over one nested SDFG of ``nproma``/``nlev`` maps."""
from ordered_set import OrderedSet

from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

# A nested SDFG is judged by what IT holds, not counted as computation here.
LAUNCH_ONLY_NODES = (nodes.MapEntry, nodes.MapExit, nodes.AccessNode, nodes.NestedSDFG)


def is_copy_or_fill_libnode(node: nodes.Node) -> bool:
    """Moves or initializes data only; a type test, since nothing carries GPU storage yet."""
    from dace.libraries.standard.nodes.copy import CopyLibraryNode
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    return isinstance(node, (CopyLibraryNode, FillLibraryNode))


def encloses_device_wide_libnode(state: SDFGState, entry: nodes.MapEntry, scope_children: dict) -> bool:
    """``entry``'s scope holds a library node expanding to a call only host code can issue."""
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
    """Every state computes only inside maps, and there is at least one. Interstate edges do not count."""
    found_map = False
    for state in sdfg.states():
        for node in state.scope_children()[None]:
            if isinstance(node, nodes.MapEntry):
                found_map = True
            elif isinstance(node, nodes.NestedSDFG):
                if not sdfg_only_launches(node.sdfg):
                    return False
                found_map = True
            elif not isinstance(node, LAUNCH_ONLY_NODES):
                return False
    return found_map


def is_taskloop_map(state: SDFGState, entry: nodes.MapEntry, scope_children: dict) -> bool:
    """``entry`` launches work rather than doing it: it encloses a device-wide library node, or all
    computation under it sits in an inner map or a launching nested SDFG. An uncollapsed perfect nest
    counts -- collapsing it into one kernel is canonicalization's job."""
    if encloses_device_wide_libnode(state, entry, scope_children):
        return True

    launches = False
    for node in scope_children.get(entry, ()):
        if isinstance(node, nodes.MapEntry):
            launches = True
        elif isinstance(node, nodes.NestedSDFG):
            if not sdfg_only_launches(node.sdfg):
                return False
            launches = True
        elif not isinstance(node, LAUNCH_ONLY_NODES):
            return False
    return launches


def taskloop_maps(sdfg: SDFG) -> OrderedSet:
    found = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_taskloop_map(state, node, scope_children):
                    found.add(node)
    return found
