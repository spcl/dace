# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility functions for traversing the SDFG hierarchy to discover parent
map scopes, loop regions, and nested SDFG boundaries.
"""
from typing import List, Set, Union

import dace
import dace.sdfg.nodes
from dace.properties import CodeBlock
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalBlock


def _get_parent_state(sdfg: dace.SDFG, nsdfg_node: dace.sdfg.nodes.NestedSDFG) -> Union[dace.SDFGState, None]:
    """Find the state that contains a given NestedSDFG node."""
    if nsdfg_node is None:
        return None
    for n, g in sdfg.all_nodes_recursive():
        if n is nsdfg_node:
            return g
    return None


def get_parent_map_and_loop_scopes(
        root_sdfg: dace.SDFG, node: Union[dace.sdfg.nodes.MapEntry, ControlFlowRegion, dace.sdfg.nodes.Tasklet,
                                          ConditionalBlock, dace.sdfg.nodes.LibraryNode],
        parent_state: Union[dace.SDFGState, None]) -> List[Union[dace.sdfg.nodes.MapEntry, LoopRegion]]:
    """
    Collect all parent map entries and loop regions enclosing *node*,
    traversing upward through scope dicts, control-flow regions, and
    nested SDFG boundaries until the root SDFG is reached.

    :param root_sdfg: The top-level SDFG.
    :param node: The starting node (MapEntry, Tasklet, etc.) or a
        ControlFlowRegion / ConditionalBlock.
    :param parent_state: The SDFGState containing *node*, or ``None`` if
        *node* is a ControlFlowRegion.
    :return: A list of parent scopes (MapEntry or LoopRegion), ordered
        from innermost to outermost.
    """
    scope_dict = parent_state.scope_dict() if parent_state is not None else None
    parent_scopes: List[Union[dace.sdfg.nodes.MapEntry, LoopRegion]] = []
    cur_node = node

    # Walk up the scope dict inside the current state
    if isinstance(cur_node, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.Tasklet, dace.sdfg.nodes.LibraryNode)):
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.sdfg.nodes.MapEntry):
                parent_scopes.append(scope_dict[cur_node])
            cur_node = scope_dict[cur_node]

    # Walk up control-flow regions (LoopRegion, etc.)
    parent_graph = (parent_state.parent_graph if parent_state is not None else node.parent_graph)
    parent_sdfg = (parent_state.sdfg if parent_state is not None else node.parent_graph.sdfg)
    while parent_graph != parent_sdfg:
        if isinstance(parent_graph, LoopRegion):
            parent_scopes.append(parent_graph)
        parent_graph = parent_graph.parent_graph

    # Walk up through nested SDFG boundaries
    parent_nsdfg_node = parent_sdfg.parent_nsdfg_node
    parent_nsdfg_parent_state = _get_parent_state(root_sdfg, parent_nsdfg_node)
    while parent_nsdfg_node is not None and parent_nsdfg_parent_state is not None:
        scope_dict = parent_nsdfg_parent_state.scope_dict()
        cur_node = parent_nsdfg_node
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.sdfg.nodes.MapEntry):
                parent_scopes.append(scope_dict[cur_node])
            cur_node = scope_dict[cur_node]

        parent_graph = parent_nsdfg_parent_state.parent_graph
        parent_sdfg = parent_graph.sdfg
        while parent_graph != parent_sdfg:
            if isinstance(parent_graph, LoopRegion):
                parent_scopes.append(parent_graph)
            parent_graph = parent_graph.parent_graph

        parent_nsdfg_node = parent_sdfg.parent_nsdfg_node
        parent_nsdfg_parent_state = _get_parent_state(root_sdfg, parent_nsdfg_node)

    return parent_scopes


def replace_length_one_arrays_with_scalars(sdfg: dace.SDFG,
                                           recursive: bool = True,
                                           transient_only: bool = False) -> Set[str]:
    """Rewrite every length-1 ``Array`` (shape ``(1,)``) on ``sdfg`` to
    a true ``Scalar`` of the same dtype, and drop the ``[0]`` accessors
    from interstate-edge assignments, conditional-block branch guards,
    and loop-region condition expressions.

    Args:
        sdfg:           Top-level SDFG to rewrite in place.
        recursive:      Recurse into nested SDFGs (only their TRANSIENT
                        length-1 arrays get rewritten -- a non-transient
                        nested-SDFG arg is part of its parent's signature
                        and would change the caller's contract).
        transient_only: Restrict the top-level pass to transient arrays
                        (default: False -- the bridge wants both signature
                        and local rewrites).

    Returns:
        The set of array names that were rewritten to scalars.

    Notes:
        Lifted from ``yakup/dev`` for use as a post-generation cleanup
        in the HLFIR Fortran -> DaCe bridge.  ``Scalar`` data on the
        SDFG signature is bound by plain Python ``int`` / ``float``,
        whereas a length-1 ``Array`` is bound by a numpy 1-element
        buffer -- this pass moves the bridge's outputs from the latter
        to the former wherever it's safe (bridge inputs are already
        emitted as ``Scalar`` directly; this fixes leftover length-1
        outputs and any locals that landed as 1-element transients).
    """
    scalarized: Set[str] = set()
    for arr_name, arr in [(k, v) for k, v in sdfg.arrays.items()]:
        if isinstance(arr, dace.data.Array) and (arr.shape == (1, ) or arr.shape == [1]):
            if (not transient_only) or arr.transient:
                sdfg.remove_data(arr_name, validate=False)
                sdfg.add_scalar(name=arr_name,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                transient=arr.transient,
                                lifetime=arr.lifetime,
                                debuginfo=arr.debuginfo,
                                find_new_name=False)
                scalarized.add(arr_name)

    # Strip ``[0]`` from interstate-edge assignment RHSs.
    for edge in sdfg.all_interstate_edges():
        new_assigns = {}
        for k, v in edge.data.assignments.items():
            nv = v
            for nm in scalarized:
                if f'{nm}[0]' in nv:
                    nv = nv.replace(f'{nm}[0]', nm)
            new_assigns[k] = nv
        edge.data.assignments = new_assigns

    # Strip ``[0]`` from conditional-block branch guards.
    for node in sdfg.all_control_flow_blocks():
        if isinstance(node, ConditionalBlock):
            for cond, _body in node.branches:
                if cond is None:
                    continue
                src = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
                for nm in scalarized:
                    if f'{nm}[0]' in src:
                        src = src.replace(f'{nm}[0]', nm)
                # Mutate the existing CodeBlock in place when possible.
                if isinstance(cond, CodeBlock):
                    cond.as_string = src

    # Strip ``[0]`` from loop-region condition expressions.
    for node in sdfg.all_control_flow_regions():
        if isinstance(node, LoopRegion):
            cond = node.loop_condition
            src = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
            for nm in scalarized:
                if f'{nm}[0]' in src:
                    src = src.replace(f'{nm}[0]', nm)
            if isinstance(cond, CodeBlock):
                cond.as_string = src
            else:
                node.loop_condition = CodeBlock(src, dace.dtypes.Language.Python)

    # Strip ``[<expr>]`` -- any subset, not just ``[0]`` -- from memlet
    # subsets that reference the scalarized arrays.  When the array is
    # literally length 1 there's only one element, so any subset
    # expression resolves to that single value -- the bridge sometimes
    # synthesises ``arr[(je) - offset_arr_d0]`` even for size-1 arrays
    # because the renderer doesn't special-case the rank-1 length-1
    # shape; collapse those to a scalar memlet here.
    from dace import Memlet
    for state in sdfg.all_states():
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            if mem.data not in scalarized:
                continue
            edge.data = Memlet(data=mem.data, subset='0', wcr=mem.wcr)

    # The offset / dimension symbols that were carried purely for the
    # rewritten arrays are now dead.  Drop them from the SDFG so the
    # signature shrinks and downstream codegen doesn't pass un-used
    # parameters.  We keep symbols that are still referenced by another
    # array's shape / lower bounds.
    referenced: set[str] = set()
    for desc in sdfg.arrays.values():
        for s in getattr(desc, 'shape', ()):
            referenced.update(str(x) for x in dace.symbolic.symlist(s).values())
        for s in getattr(desc, 'offset', ()):
            referenced.update(str(x) for x in dace.symbolic.symlist(s).values())
    for nm in list(sdfg.symbols):
        if nm in referenced:
            continue
        prefixes = [f'offset_{a}_d' for a in scalarized] + [f'{a}_d' for a in scalarized]
        if any(nm.startswith(p) for p in prefixes):
            sdfg.symbols.pop(nm, None)

    if recursive:
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    replace_length_one_arrays_with_scalars(node.sdfg, recursive=True, transient_only=True)

    return scalarized
