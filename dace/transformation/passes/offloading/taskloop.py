# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps that only LAUNCH work stay on the host; their bodies become the kernels.
ICON's shape: an ``nblks`` map over one nested SDFG of ``nproma``/``nlev`` maps."""
from typing import Dict, Optional

from dace.ordered import OrderedSet

from dace import symbolic
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState
from dace.subsets import Range


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


#: What an extent nobody has pinned is worth when two of them are compared. The comparison below only
#: needs to know which side has MORE iterations, and at compile time ``nproma`` and ``nblks`` are both
#: just names -- so every unpinned symbol is given the same value and the shapes decide. A concrete
#: extent keeps its own value, which is the case that matters: a literal ``[0:4]`` inner map really is
#: four iterations, and must not be mistaken for a full-sized axis.
NOMINAL_EXTENT = 1024


def nominal_volume(subset: Range) -> Optional[int]:
    """Iterations in ``subset`` with every unpinned symbol set to :data:`NOMINAL_EXTENT`.

    ``None`` when the count does not survive substitution -- a data-dependent bound, say. The caller
    treats that as "no opinion" rather than guessing at a number.
    """
    count = subset.num_elements()
    try:
        resolved = symbolic.evaluate(count, {symbol: NOMINAL_EXTENT for symbol in count.free_symbols})
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    return int(resolved) if float(resolved).is_integer() and resolved > 0 else None


def nest_volume(entry: nodes.MapEntry, scope_children: dict) -> Optional[int]:
    """Threads a kernel rooted at ``entry`` would have: its own iterations times its deepest nest.

    Nested maps under one kernel are collapsed into the launch, so their extents multiply. Sibling
    nests are launched one after another, so the widest of them is what the kernel has to fill.
    """
    own = nominal_volume(entry.map.range)
    if own is None:
        return None
    inner = [
        nest_volume(node, scope_children) for node in scope_children.get(entry, ()) if isinstance(node, nodes.MapEntry)
    ]
    if any(volume is None for volume in inner):
        return None
    return own * max(inner, default=1)


def body_has_more_threads(entry: nodes.MapEntry, scope_children: dict) -> bool:
    """Does moving the kernel INSIDE ``entry`` launch more threads than leaving it outside?

    The two lowerings differ in exactly one way. Left alone, ``entry`` IS the kernel and everything
    under it is sequential, so the launch is as wide as ``entry``'s own iteration space. Made a
    taskloop, ``entry`` runs on the host and each of its bodies becomes a kernel of its own, so the
    launch is as wide as the widest body nest. More iterations in the kernel is more threads, which
    is what a GPU wants, so the taskloop is worth it when the body is the wider of the two.

    Unresolvable either side means no opinion, and the caller keeps the structural answer: the
    extent-dependence gate has already removed the shapes that were actively harmful.
    """
    outside = nominal_volume(entry.map.range)
    if outside is None:
        return True
    inside = [
        nest_volume(node, scope_children) for node in scope_children.get(entry, ()) if isinstance(node, nodes.MapEntry)
    ]
    if not inside or any(volume is None for volume in inside):
        return True
    return max(inside) >= outside


def is_taskloop_map(state: SDFGState,
                    entry: nodes.MapEntry,
                    scope_children: dict,
                    launch_only: bool = True,
                    overrides: Optional[Dict[str, bool]] = None) -> bool:
    """``entry`` launches work rather than doing it, so it belongs on the host.

    ``overrides`` is the caller's answer for a map, by label, and it is final in BOTH directions: a
    name mapped to ``True`` is a taskloop whatever the rules below say, and one mapped to ``False``
    never is. Nothing else consults the heuristics, because a caller who names a map has looked at
    the kernel and the rules have not.

    Enclosing a device-wide library node is a requirement, not a heuristic: a cuBLAS call is issued by
    host code. ``launch_only`` adds the optional reason -- an uncollapsed nest counts, and collapsing
    it into one kernel is canonicalization's job -- for a body whose extents do not depend on this map
    (:func:`body_extents_depend_on_entry`) and which is the wider of the two lowerings
    (:func:`body_has_more_threads`).
    """
    if overrides and entry.map.label in overrides:
        return bool(overrides[entry.map.label])
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
    return launches and body_has_more_threads(entry, scope_children)


def taskloop_maps(sdfg: SDFG, launch_only: bool = True, overrides: Optional[Dict[str, bool]] = None) -> OrderedSet:
    found = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            scope_children = state.scope_children()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and is_taskloop_map(state, node, scope_children, launch_only,
                                                                        overrides):
                    found.add(node)
    return found
