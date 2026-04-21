# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility functions for manipulating/constructing/analyzing SDFG components
"""
from typing import List, Union

import dace
import dace.sdfg.nodes
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
