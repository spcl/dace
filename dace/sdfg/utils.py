# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Various utility functions to create, traverse, and modify SDFGs. """

import collections
import copy
import os
import warnings
import networkx as nx
import time

import dace.sdfg.nodes
from dace.codegen import compiled_sdfg as csdfg
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.nodes import Node, NestedSDFG
from dace.sdfg.state import (AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, SDFGState,
                             StateSubgraphView, LoopRegion, ControlFlowRegion)
from dace.sdfg.scope import ScopeSubgraphView
from dace.sdfg import nodes as nd, graph as gr, propagation
from dace import config, data as dt, dtypes, memlet as mm, subsets as sbs
from dace.cli.progress import optional_progressbar
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Sequence, Tuple, Type, Union
from dace.properties import CodeBlock


def node_path_graph(*args) -> gr.OrderedDiGraph:
    """
    Generates a path graph passing through the input nodes.

    The function generates a graph using as nodes the input arguments.
    Subsequently, it creates a path passing through all the nodes, in
    the same order as they were given in the function input.

    :param args: Variable number of nodes or a list of nodes.
    :return: A directed graph based on the input arguments.
    """

    # 1. Create new networkx directed graph.
    path = gr.OrderedDiGraph()
    # 2. Place input nodes in a list.
    if len(args) == 1 and isinstance(args[0], list):
        # Input is a single list of nodes.
        input_nodes = args[0]
    else:
        # Input is a variable number of nodes.
        input_nodes = list(args)
    # 3. Add nodes to the graph.
    path.add_nodes_from(input_nodes)
    # 4. Add path edges to the graph.
    for i in range(len(input_nodes) - 1):
        path.add_edge(input_nodes[i], input_nodes[i + 1], None)
    # 5. Return the graph.
    return path


def depth_limited_search(source, depth):
    """ Return best node and its value using a limited-depth Search (depth-limited DFS). """
    value = source.evaluate()
    if depth == 0:
        return source, value

    candidate = source
    candidate_value = value

    # Node, depth, children generator
    stack = [(source, 0, source.children_iter())]
    while stack:
        node, cur_depth, children = stack[-1]
        try:
            child = next(children)
            child_val = child.evaluate()
            # Check for best candidate
            if child_val > candidate_value:
                candidate = child
                candidate_value = child_val

            if cur_depth < depth - 1:
                stack.append((child, cur_depth + 1, child.children_iter()))
        except StopIteration:
            stack.pop()

    # Return maximal candidate
    return candidate, candidate_value


def depth_limited_dfs_iter(source, depth):
    """ Produce nodes in a Depth-Limited DFS. """
    if depth == 0:
        yield source
        return

    # Node, depth, children generator
    stack = [(source, 0, source.children_iter())]
    while stack:
        node, cur_depth, children = stack[-1]
        try:
            child = next(children)
            yield child

            if cur_depth < depth - 1:
                stack.append((child, cur_depth + 1, child.children_iter()))
        except StopIteration:
            stack.pop()


def dfs_topological_sort(G, sources=None, condition=None, reverse=False):
    """
    Produce nodes in a depth-first topological ordering.

    The function produces nodes in a depth-first topological ordering
    (DFS to make sure maps are visited properly), with the condition
    that each node visited had all its predecessors visited. Applies
    for DAGs only, but works on any directed graph.

    :param G: An input DiGraph (assumed acyclic).
    :param sources: (optional) node or list of nodes that
                    specify starting point(s) for depth-first search and return
                    edges in the component reachable from source.
    :param reverse: If True, traverses the graph backwards from the sources.
    :return: A generator of nodes in the lastvisit depth-first-search.

    :note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py by D. Eppstein, July 2004.

    :note: If a source is not specified then a source is chosen arbitrarily and
           repeatedly until all components in the graph are searched.
    """
    if reverse:
        source_nodes = 'sink_nodes'
        predecessors = G.successors
        neighbors = G.predecessors
    else:
        source_nodes = 'source_nodes'
        predecessors = G.predecessors
        neighbors = G.successors

    if sources is None:
        # produce edges for all components
        src_nodes = getattr(G, source_nodes, lambda: G)
        nodes = list(src_nodes())
        if len(nodes) == 0:
            nodes = G
    else:
        # produce edges for components with source
        try:
            nodes = iter(sources)
        except TypeError:
            nodes = [sources]

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        yield start
        visited.add(start)
        stack = [(start, iter(neighbors(start)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # Make sure that all predecessors have been visited
                    skip = False
                    for pred in predecessors(child):
                        if pred not in visited:
                            skip = True
                            break
                    if skip:
                        continue

                    visited.add(child)
                    if condition is None or condition(parent, child):
                        yield child
                        stack.append((child, iter(neighbors(child))))
            except StopIteration:
                stack.pop()


def _find_nodes_impl(
    node_to_start: Node,
    state: SDFGState,
    forward: bool,
    seen: Optional[Set[Node]],
) -> Set[Node]:
    to_scan: List[Node] = [node_to_start]
    scanned_nodes: Set[Node] = set() if seen is None else seen
    if forward:
        get_edges = state.out_edges
        get_node = lambda e: e.dst
    else:
        get_edges = state.in_edges
        get_node = lambda e: e.src
    while len(to_scan) != 0:
        node_to_scan = to_scan.pop()
        if node_to_scan in scanned_nodes:
            continue
        to_scan.extend(get_node(edge) for edge in get_edges(node_to_scan) if get_node(edge) not in scanned_nodes)
        scanned_nodes.add(node_to_scan)
    return scanned_nodes


def find_downstream_nodes(node_to_start: Node, state: SDFGState, seen: Optional[Set[Node]] = None) -> Set[Node]:
    """Find all downstream nodes of `node_to_start`.

    The function will explore the state, similar to a BFS, just that the order in which the nodes of
    the dataflow is explored is unspecific. It is possible to pass a `set` of nodes that should be
    considered as already visited. It is important that the function will return the set of found
    nodes. In case `seen` was passed that `set` will be updated in place and be returned.

    :param node_to_start: Where to start the exploration of the state.
    :param state: The state on which we operate on.
    :param seen: The set of already seen nodes.

    :note: See also `find_upstream_nodes()` in case the dataflow should be explored in the reverse direction.
    """
    return _find_nodes_impl(node_to_start=node_to_start, state=state, seen=seen, forward=True)


def find_upstream_nodes(node_to_start: Node, state: SDFGState, seen: Optional[Set[Node]] = None) -> Set[Node]:
    """Find all upstream nodes of `node_to_start`.

    The function will explore the state, similar to a BFS, just that the order in which the nodes of
    the dataflow is explored is unspecific. It is possible to pass a `set` of nodes that should be
    considered as already visited. It is important that the function will return the set of found
    nodes. In case `seen` was passed that `set` will be updated in place and be returned.

    The main difference to `find_downstream_nodes()` is that the dataflow is traversed in reverse
    order or "against the flow".

    :param node_to_start: Where to start the exploration of the state.
    :param state: The state on which we operate on.
    :param seen: The set of already seen nodes.
    """
    return _find_nodes_impl(node_to_start=node_to_start, state=state, seen=seen, forward=False)


class StopTraversal(Exception):
    """
    Special exception that stops DFS conditional traversal beyond the current node.

    :see: dfs_conditional
    """
    pass


def dfs_conditional(G, sources=None, condition=None, reverse=False, yield_parent=False):
    """
    Produce nodes in a depth-first ordering with an optional condition to stop traversal.
    If ``StopTraversal`` is raised during iteration, the outgoing edges of the current node
    will not be traversed.

    :param G: An input DiGraph (may have cycles).
    :param sources: (optional) node or list of nodes that
                    specify starting point(s) for depth-first search and return
                    edges in the component reachable from source. If None, traverses from
                    every node in the graph.
    :param condition: (optional) a callback that receives the traversed parent and child.
                      Called before each child node is traversed.
                      If it returns True, traversal proceeds normally. If False, the child
                      and reachable nodes are not traversed.
    :param reverse: If True, traverses the graph backwards from the sources.
    :param yield_parent: If True, yields a 2-tuple of (parent, child)
    :return: A generator of edges in the lastvisit depth-first-search.

    :note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py by D. Eppstein, July 2004.

    :note: If a source is not specified then a source is chosen arbitrarily and
           repeatedly until all components in the graph are searched.
    """
    if reverse:
        successors = G.predecessors
    else:
        successors = G.successors

    if sources is None:
        # produce edges for all components
        nodes = G
    else:
        # produce edges for components with source
        try:
            nodes = iter(sources)
        except TypeError:
            nodes = [sources]

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        try:
            if yield_parent:
                yield None, start
            else:
                yield start
        except StopTraversal:
            return
        visited.add(start)
        stack = [(start, iter(successors(start)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    if condition is None or condition(parent, child):
                        try:
                            if yield_parent:
                                yield parent, child
                            else:
                                yield child
                            stack.append((child, iter(successors(child))))
                        except StopTraversal:
                            pass
            except StopIteration:
                stack.pop()


def scope_aware_topological_sort(G: SDFGState,
                                 sources: Optional[Sequence[Node]] = None,
                                 condition: Optional[Callable[[Node, Node], bool]] = None,
                                 reverse: bool = False,
                                 visited: Optional[Set[Node]] = None):
    """
    Traverses an SDFG state in topological order, yielding one node at a time, with the requirement that every scope
    (e.g., map) is traversed continuously. This means that the sort will start on the outer nodes, and as it
    encounters an entry node it will traverse the scope completely, without skipping out of it,
    until all nodes in the scope (and sub-scopes) have been visited.

    :param G: The state to traverse.
    :param sources: An optional sequence of nodes to start traversal from. If not given, all source
                    (or sink) nodes will be used.
    :param condition: An optional callable that receives (current node, child node), and upon returning
                      False, will stop traversal of the child node and its descendants.
    :param reverse: If True, the graph will be traversed in reverse order (entering scopes via their exit node)
    :param visited: An optional set that will be filled with the visited nodes.
    """
    if reverse:
        source_nodes = 'sink_nodes'
        predecessors = G.successors
        neighbors = G.predecessors
    else:
        source_nodes = 'source_nodes'
        predecessors = G.predecessors
        neighbors = G.successors

    if sources is None:
        # produce edges for all components
        src_nodes = getattr(G, source_nodes, lambda: G)
        nodes = list(src_nodes())
        if len(nodes) == 0:
            nodes = G
    else:
        # produce edges for components with source
        try:
            nodes = iter(sources)
        except TypeError:
            nodes = [sources]

    visited = visited if visited is not None else set()
    for start in nodes:
        if start in visited:
            continue
        yield start
        visited.add(start)
        stack = [(start, iter(neighbors(start)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # Make sure that all predecessors have been visited
                    skip = False
                    for pred in predecessors(child):
                        if pred not in visited:
                            skip = True
                            break
                    if skip:
                        continue

                    visited.add(child)
                    if ((reverse and isinstance(child, dace.nodes.ExitNode))
                            or (not reverse and isinstance(child, dace.nodes.EntryNode))):
                        if reverse:
                            entry = G.entry_node(child)
                            scope_subgraph = G.scope_subgraph(entry)
                        else:
                            scope_subgraph = G.scope_subgraph(child)
                        yield from scope_aware_topological_sort(scope_subgraph,
                                                                sources=[child],
                                                                condition=condition,
                                                                reverse=reverse,
                                                                visited=visited)
                    if condition is None or condition(parent, child):
                        yield child
                        stack.append((child, iter(neighbors(child))))
            except StopIteration:
                stack.pop()
    return visited


def nodes_in_all_simple_paths(G, source, target, condition: Callable[[Any], bool] = None) -> Set[Any]:
    """
    Returns a set of nodes that appear in any of the paths from ``source``
    to ``targets``. Optionally, a condition can be given to control traversal.

    :param G: The graph to traverse.
    :param source: Source node.
    :param targets:

    :note: This algorithm uses a modified depth-first search, adapted from
           ``networkx.all_simple_paths``.

    :note: The algorithm is written for directed *graphs*. For multigraphs, use
           ``networkx.all_simple_paths``!

    References:
    [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms", Addison Wesley Professional, 3rd ed., 2001.
    """

    cutoff = len(G) - 1
    result = set()
    visited = dict.fromkeys([source])
    targets = {target}
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if condition is not None:
            while child is not None and not condition(child):
                child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child is target:
                result.update(list(visited))
                result.add(child)
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for tgt in (targets & (set(children) | {child})) - set(visited.keys()):
                result.update(list(visited))
                result.add(tgt)
            stack.pop()
            visited.popitem()

    return result


def change_edge_dest(graph: gr.OrderedDiGraph, node_a: Union[nd.Node, gr.OrderedMultiDiConnectorGraph],
                     node_b: Union[nd.Node, gr.OrderedMultiDiConnectorGraph]):
    """ Changes the destination of edges from node A to node B.

        The function finds all edges in the graph that have node A as their
        destination. It then creates a new edge for each one found,
        using the same source nodes and data, but node B as the destination.
        Afterwards, it deletes the edges found and inserts the new ones into
        the graph.

        :param graph: The graph upon which the edge transformations will be
                      applied.
        :param node_a: The original destination of the edges.
        :param node_b: The new destination of the edges to be transformed.
    """

    # Create new incoming edges to node B, by copying the incoming edges to
    # node A and setting their destination to node B.
    edges = list(graph.in_edges(node_a))
    for e in edges:
        # Delete the incoming edges to node A from the graph.
        graph.remove_edge(e)
        # Insert the new edges to the graph.
        if isinstance(e, gr.MultiConnectorEdge):
            graph.add_edge(e.src, e.src_conn, node_b, e.dst_conn, e.data)
        else:
            graph.add_edge(e.src, node_b, e.data)


def change_edge_src(graph: gr.OrderedDiGraph, node_a: Union[nd.Node, gr.OrderedMultiDiConnectorGraph],
                    node_b: Union[nd.Node, gr.OrderedMultiDiConnectorGraph]):
    """ Changes the sources of edges from node A to node B.

        The function finds all edges in the graph that have node A as their
        source. It then creates a new edge for each one found, using the same
        destination nodes and data, but node B as the source. Afterwards, it
        deletes the edges
        found and inserts the new ones into the graph.

        :param graph: The graph upon which the edge transformations will be
                      applied.
        :param node_a: The original source of the edges to be transformed.
        :param node_b: The new source of the edges to be transformed.
    """

    # Create new outgoing edges from node B, by copying the outgoing edges from
    # node A and setting their source to node B.
    edges = list(graph.out_edges(node_a))
    for e in edges:
        # Delete the outgoing edges from node A from the graph.
        graph.remove_edge(e)
        # Insert the new edges to the graph.
        if isinstance(e, gr.MultiConnectorEdge):
            graph.add_edge(node_b, e.src_conn, e.dst, e.dst_conn, e.data)
        else:
            graph.add_edge(node_b, e.dst, e.data)


ParamsType = List['dace.symbolic.symbol']
RangesType = List[sbs.Subset]


def merge_maps(
    graph: SDFGState,
    outer_map_entry: nd.MapEntry,
    outer_map_exit: nd.MapExit,
    inner_map_entry: nd.MapEntry,
    inner_map_exit: nd.MapExit,
    param_merge: Callable[[ParamsType, ParamsType], ParamsType] = lambda p1, p2: p1 + p2,
    range_merge: Callable[[RangesType, RangesType], RangesType] = lambda r1, r2: type(r1)(r1.ranges + r2.ranges)
) -> Tuple[nd.MapEntry, nd.MapExit]:
    """ Merges two maps (their entries and exits). It is assumed that the
    operation is valid. """

    outer_map = outer_map_entry.map
    inner_map = inner_map_entry.map

    # Create merged map by inheriting attributes from outer map and using
    # the merge functions for parameters and ranges.
    merged_map = copy.deepcopy(outer_map)
    merged_map.label = outer_map.label
    merged_map.params = param_merge(outer_map.params, inner_map.params)
    merged_map.range = range_merge(outer_map.range, inner_map.range)

    merged_entry = nd.MapEntry(merged_map)
    merged_entry.in_connectors = outer_map_entry.in_connectors
    merged_entry.out_connectors = outer_map_entry.out_connectors

    merged_exit = nd.MapExit(merged_map)
    merged_exit.in_connectors = outer_map_exit.in_connectors
    merged_exit.out_connectors = outer_map_exit.out_connectors

    graph.add_nodes_from([merged_entry, merged_exit])

    # Handle the case of dynamic map inputs in the inner map
    inner_dynamic_map_inputs = dynamic_map_inputs(graph, inner_map_entry)
    for edge in inner_dynamic_map_inputs:
        remove_conn = (len(list(graph.out_edges_by_connector(edge.src, edge.src_conn))) == 1)
        conn_to_remove = edge.src_conn[4:]
        if remove_conn:
            merged_entry.remove_in_connector('IN_' + conn_to_remove)
            merged_entry.remove_out_connector('OUT_' + conn_to_remove)
        merged_entry.add_in_connector(edge.dst_conn, inner_map_entry.in_connectors[edge.dst_conn])
        outer_edge = next(graph.in_edges_by_connector(outer_map_entry, 'IN_' + conn_to_remove))
        graph.add_edge(outer_edge.src, outer_edge.src_conn, merged_entry, edge.dst_conn, outer_edge.data)
        if remove_conn:
            graph.remove_edge(outer_edge)

    # Redirect inner in edges.
    for edge in graph.out_edges(inner_map_entry):
        if edge.src_conn is None:  # Empty memlets
            graph.add_edge(merged_entry, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
            continue

        # Get memlet path and edge
        path = graph.memlet_path(edge)
        ind = path.index(edge)
        # Add an edge directly from the previous source connector to the
        # destination
        graph.add_edge(merged_entry, path[ind - 1].src_conn, edge.dst, edge.dst_conn, edge.data)

    # Redirect inner out edges.
    for edge in graph.in_edges(inner_map_exit):
        if edge.dst_conn is None:  # Empty memlets
            graph.add_edge(edge.src, edge.src_conn, merged_exit, edge.dst_conn, edge.data)
            continue

        # Get memlet path and edge
        path = graph.memlet_path(edge)
        ind = path.index(edge)
        # Add an edge directly from the source to the next destination
        # connector
        graph.add_edge(edge.src, edge.src_conn, merged_exit, path[ind + 1].dst_conn, edge.data)

    # Redirect outer edges.
    change_edge_dest(graph, outer_map_entry, merged_entry)
    change_edge_src(graph, outer_map_exit, merged_exit)

    # Clean-up
    graph.remove_nodes_from([outer_map_entry, outer_map_exit, inner_map_entry, inner_map_exit])

    return merged_entry, merged_exit


def canonicalize_memlet_trees_of_scope_node(
    state: SDFGState,
    scope_node: Union[nd.EntryNode, nd.ExitNode],
) -> int:
    """Canonicalize the Memlet trees of a single scope nodes.

    The function will modify all Memlets that are adjacent to `scope_node` such that
    the Memlet always refers to the data that is on the outside.
    This function only operates on a single scope node, i.e. either a `MapEntry` or
    a `MapExit`, if you want to process a whole Map then you should use
    `canonicalize_memlet_trees_for_map()`.

    :param state: The SDFG state in which the scope to consolidate resides.
    :param scope_node: The scope node whose edges will be consolidated.
    :return: Number of modified Memlets.

    :note: This is the "historical" expected format of Memlet trees at scope nodes,
        which was present before the introduction of `other_subset`. Running this
        transformation might fix some issues.
    """
    if isinstance(scope_node, nd.EntryNode):
        may_have_dynamic_map_range = True
        is_downward_tree = True
        outer_edges = state.in_edges(scope_node)
        get_outer_edge_connector = lambda e: e.dst_conn
        inner_edges_for = lambda conn: state.out_edges_by_connector(scope_node, conn)
        inner_prefix = 'OUT_'
        outer_prefix = 'IN_'

        def get_outer_data(e: MultiConnectorEdge[dace.Memlet]):
            mpath = state.memlet_path(e)
            assert isinstance(mpath[0].src, nd.AccessNode)
            return mpath[0].src.data

    else:
        may_have_dynamic_map_range = False
        is_downward_tree = False
        outer_edges = state.out_edges(scope_node)
        get_outer_edge_connector = lambda e: e.src_conn
        inner_edges_for = lambda conn: state.in_edges_by_connector(scope_node, conn)
        inner_prefix = 'IN_'
        outer_prefix = 'OUT_'

        def get_outer_data(e: MultiConnectorEdge[dace.Memlet]):
            mpath = state.memlet_path(e)
            assert isinstance(mpath[-1].dst, nd.AccessNode)
            return mpath[-1].dst.data

    def swap_prefix(conn: str) -> str:
        if conn.startswith(inner_prefix):
            return outer_prefix + conn[len(inner_prefix):]
        else:
            assert conn.startswith(
                outer_prefix), f"Expected connector to start with '{outer_prefix}', but it was '{conn}'."
            return inner_prefix + conn[len(outer_prefix):]

    modified_memlet = 0
    for outer_edge in outer_edges:
        outer_edge_connector = get_outer_edge_connector(outer_edge)
        if may_have_dynamic_map_range and (not outer_edge_connector.startswith(outer_prefix)):
            continue
        assert outer_edge_connector.startswith(outer_prefix)
        corresponding_inner_connector = swap_prefix(outer_edge_connector)

        # In case `scope_node` is at the global scope it should be enough to run
        #  `outer_edge.data.data` but this way it is more in line with consolidate.
        outer_data = get_outer_data(outer_edge)

        for inner_edge in inner_edges_for(corresponding_inner_connector):
            for mtree in state.memlet_tree(inner_edge).traverse_children(include_self=True):
                medge: MultiConnectorEdge[dace.Memlet] = mtree.edge
                if medge.data.data == outer_data:
                    # This edge is already referring to the outer data, so no change is needed.
                    continue

                # Now we have to extract subset from the Memlet.
                if is_downward_tree:
                    subset = medge.data.get_src_subset(medge, state)
                    other_subset = medge.data.dst_subset
                else:
                    subset = medge.data.get_dst_subset(medge, state)
                    other_subset = medge.data.src_subset

                # Now update them correctly.
                medge.data._data = outer_data
                medge.data._subset = subset
                medge.data._other_subset = other_subset
                medge.data.try_initialize(state.sdfg, state, medge)
                modified_memlet += 1

    return modified_memlet


def canonicalize_memlet_trees_for_map(
    state: SDFGState,
    map_node: Union[nd.EntryNode, nd.ExitNode],
) -> int:
    """Canonicalize the Memlets of an entire Map scope.

    This function is similar to `canonicalize_memlet_trees_of_scope_node()`, but it acts
    on both scope nodes, i.e. `MapEntry` and `MapExit`, that constitute the Map scope.
    The function returns the number of canonicalized Memlets.

    :param state: The state that contains the Map.
    :param map_node: Either the `MapEntry` or `MapExit` node of the map that should be canonicalized.
    """
    if isinstance(map_node, nd.MapEntry):
        me = map_node
        mx = state.exit_node(me)
    else:
        assert isinstance(map_node, nd.MapExit)
        mx = map_node
        me = state.entry_node(mx)

    ret = canonicalize_memlet_trees_of_scope_node(state, me)
    ret += canonicalize_memlet_trees_of_scope_node(state, mx)
    return ret


def canonicalize_memlet_trees(
    sdfg: 'dace.SDFG',
    starting_scope: Optional['dace.sdfg.scope.ScopeTree'] = None,
) -> int:
    """Canonicalize the Memlet trees of all scopes in the SDFG.

    This function runs `canonicalize_memlet_trees_of_scope_node()` on all scopes
    in the SDFG. Note that this function does not recursively processes
    nested SDFGs.

    :param sdfg: The SDFG to consolidate.
    :param starting_scope: If not None, starts with a certain scope. Note in that
        mode only the state in which the scope is located will be processes.
    :return: Number of modified Memlets.
    """

    total_modified_memlets = 0
    for state in sdfg.states():
        # Start bottom-up
        if starting_scope is not None and starting_scope.entry not in state.nodes():
            continue

        queue = [starting_scope] if starting_scope else state.scope_leaves()
        next_queue = []
        while len(queue) > 0:
            for scope in queue:
                if scope.entry is not None:
                    total_modified_memlets += canonicalize_memlet_trees_of_scope_node(state, scope.entry)
                if scope.exit is not None:
                    total_modified_memlets += canonicalize_memlet_trees_of_scope_node(state, scope.exit)
                if scope.parent is not None:
                    next_queue.append(scope.parent)
            queue = next_queue
            next_queue = []

        if starting_scope is not None:
            # No need to traverse other states
            break

    return total_modified_memlets


def consolidate_edges_scope(state: SDFGState, scope_node: Union[nd.EntryNode, nd.ExitNode]) -> int:
    """
        Union scope-entering memlets relating to the same data node in a scope.
        This effectively reduces the number of connectors and allows more
        transformations to be performed, at the cost of losing the individual
        per-tasklet memlets.

        :param state: The SDFG state in which the scope to consolidate resides.
        :param scope_node: The scope node whose edges will be consolidated.
        :return: Number of edges removed.
    """
    if scope_node is None:
        return 0
    data_to_conn = {}
    consolidated = 0
    if isinstance(scope_node, nd.EntryNode):
        outer_edges = state.in_edges
        inner_edges = state.out_edges
        inner_conn = lambda e: e.src_conn
        remove_outer_connector = scope_node.remove_in_connector
        remove_inner_connector = scope_node.remove_out_connector
        prefix, oprefix = 'IN_', 'OUT_'

        def get_outer_data(e: MultiConnectorEdge[dace.Memlet]):
            mpath = state.memlet_path(e)
            assert isinstance(mpath[0].src, nd.AccessNode)
            return mpath[0].src.data

        def get_outer_subset(e: MultiConnectorEdge[dace.Memlet]):
            return e.data.get_src_subset(e, state)

        def set_outer_subset(e: MultiConnectorEdge[dace.Memlet], new_subset: sbs.Subset):
            e.data.src_subset = new_subset

    else:
        outer_edges = state.out_edges
        inner_edges = state.in_edges
        inner_conn = lambda e: e.dst_conn
        remove_outer_connector = scope_node.remove_out_connector
        remove_inner_connector = scope_node.remove_in_connector
        prefix, oprefix = 'OUT_', 'IN_'

        def get_outer_data(e: MultiConnectorEdge[dace.Memlet]):
            mpath = state.memlet_path(e)
            assert isinstance(mpath[-1].dst, nd.AccessNode)
            return mpath[-1].dst.data

        def get_outer_subset(e: MultiConnectorEdge[dace.Memlet]):
            return e.data.get_dst_subset(e, state)

        def set_outer_subset(e: MultiConnectorEdge[dace.Memlet], new_subset: sbs.Subset):
            e.data.dst_subset = new_subset

    edges_by_connector = collections.defaultdict(list)
    connectors_to_remove = set()
    for e in inner_edges(scope_node):
        if e.data.is_empty():
            continue
        conn = inner_conn(e)
        edges_by_connector[conn].append(e)
        odata = get_outer_data(e)
        if odata not in data_to_conn:
            data_to_conn[odata] = conn
        elif data_to_conn[odata] != conn:  # Need to consolidate
            connectors_to_remove.add(conn)

    for conn in connectors_to_remove:
        e = edges_by_connector[conn][0]
        odata = get_outer_data(e)
        offset = 3 if conn.startswith('IN_') else (4 if conn.startswith('OUT_') else len(oprefix))
        # Outer side of the scope - remove edge and union subsets
        target_conn = prefix + data_to_conn[odata][offset:]
        conn_to_remove = prefix + conn[offset:]
        remove_outer_connector(conn_to_remove)
        if isinstance(scope_node, nd.EntryNode):
            out_edges = [ed for ed in outer_edges(scope_node) if ed.dst_conn == target_conn]
            edges_to_remove = [ed for ed in outer_edges(scope_node) if ed.dst_conn == conn_to_remove]
        else:
            out_edges = [ed for ed in outer_edges(scope_node) if ed.src_conn == target_conn]
            edges_to_remove = [ed for ed in outer_edges(scope_node) if ed.src_conn == conn_to_remove]
        assert len(edges_to_remove) == 1 and len(out_edges) == 1
        edge_to_remove = edges_to_remove[0]
        out_edge = out_edges[0]
        set_outer_subset(out_edge, sbs.union(get_outer_subset(out_edge), get_outer_subset(edge_to_remove)))

        # Check if dangling connectors have been created and remove them,
        # as well as their parent edges
        remove_edge_and_dangling_path(state, edge_to_remove)

        consolidated += 1
        # Inner side of the scope - remove and reconnect
        if isinstance(scope_node, nd.EntryNode):
            remove_inner_connector(e.src_conn)
            for e in edges_by_connector[conn]:
                e._src_conn = data_to_conn[odata]
        else:
            remove_inner_connector(e.dst_conn)
            for e in edges_by_connector[conn]:
                e._dst_conn = data_to_conn[odata]

    return consolidated


def remove_edge_and_dangling_path(state: SDFGState, edge: MultiConnectorEdge):
    """
    Removes an edge and all of its parent edges in a memlet path, cleaning
    dangling connectors and isolated nodes resulting from the removal.

    :param state: The state in which the edge exists.
    :param edge: The edge to remove.
    """
    mtree = state.memlet_tree(edge)
    inwards = (isinstance(edge.src, nd.EntryNode) or isinstance(edge.dst, nd.EntryNode))

    # Traverse tree upwards, removing edges and connectors as necessary
    curedge = mtree
    while curedge is not None:
        e = curedge.edge
        state.remove_edge(e)
        if inwards:
            neighbors = [] if not e.src_conn else [
                neighbor for neighbor in state.out_edges_by_connector(e.src, e.src_conn)
            ]
        else:
            neighbors = [] if not e.dst_conn else [
                neighbor for neighbor in state.in_edges_by_connector(e.dst, e.dst_conn)
            ]
        if len(neighbors) > 0:  # There are still edges connected, leave as-is
            break

        # Remove connector and matching outer connector
        if inwards:
            if e.src_conn:
                e.src.remove_out_connector(e.src_conn)
                e.src.remove_in_connector('IN' + e.src_conn[3:])
        else:
            if e.dst_conn:
                e.dst.remove_in_connector(e.dst_conn)
                e.dst.remove_out_connector('OUT' + e.dst_conn[2:])

        # Continue traversing upwards
        curedge = curedge.parent
    else:
        # Check if an isolated node have been created at the root and remove
        root_edge = mtree.root().edge
        root_node: nd.Node = root_edge.src if inwards else root_edge.dst
        if state.degree(root_node) == 0:
            state.remove_node(root_node)


def consolidate_edges(
    sdfg: SDFG,
    starting_scope=None,
    propagate: bool = True,
) -> int:
    """
    Union scope-entering memlets relating to the same data node in all states.
    This effectively reduces the number of connectors and allows more
    transformations to be performed, at the cost of losing the individual
    per-tasklet memlets.

    :param sdfg: The SDFG to consolidate.
    :param starting_scope: If not None, starts with a certain scope
    :param propagate: If True, applies memlet propagation after consolidation
    :return: Number of edges removed.
    """
    from dace.sdfg.propagation import propagate_memlets_scope

    total_consolidated = 0
    for state in sdfg.states():
        # Start bottom-up
        if starting_scope and starting_scope.entry not in state.nodes():
            continue

        queue = [starting_scope] if starting_scope else state.scope_leaves()
        next_queue = []
        while len(queue) > 0:
            for scope in queue:
                propagate_entry, propagate_exit = False, False

                consolidated = consolidate_edges_scope(state, scope.entry)
                total_consolidated += consolidated
                if consolidated > 0:
                    propagate_entry = True

                consolidated = consolidate_edges_scope(state, scope.exit)
                total_consolidated += consolidated
                if consolidated > 0:
                    propagate_exit = True

                # Repropagate memlets
                if propagate:
                    propagate_memlets_scope(sdfg, state, scope, propagate_entry, propagate_exit)

                if scope.parent is not None:
                    next_queue.append(scope.parent)
            queue = next_queue
            next_queue = []

        if starting_scope is not None:
            # No need to traverse other states
            break

    return total_consolidated


def is_array_stream_view(sdfg: SDFG, dfg: SDFGState, node: nd.AccessNode):
    """ Test whether a stream is directly connected to an array. """

    # Test all memlet paths from the array. If the path goes directly
    # to/from a stream, construct a stream array view
    all_source_paths = []
    source_paths = []
    all_sink_paths = []
    sink_paths = []
    for e in dfg.in_edges(node):
        src_node = dfg.memlet_path(e)[0].src
        # Append empty path to differentiate between a copy and an array-view
        if isinstance(src_node, nd.CodeNode):
            all_source_paths.append(None)
        # Append path from source node
        if isinstance(src_node, nd.AccessNode) and isinstance(src_node.desc(sdfg), dt.Array):
            source_paths.append(src_node)
    for e in dfg.out_edges(node):
        sink_node = dfg.memlet_path(e)[-1].dst

        # Append empty path to differentiate between a copy and an array-view
        if isinstance(sink_node, nd.CodeNode):
            all_sink_paths.append(None)
        # Append path to sink node
        if isinstance(sink_node, nd.AccessNode) and isinstance(sink_node.desc(sdfg), dt.Array):
            sink_paths.append(sink_node)

    all_sink_paths.extend(sink_paths)
    all_source_paths.extend(source_paths)

    # Special case: stream can be represented as a view of an array
    if ((len(all_source_paths) > 0 and len(sink_paths) == 1) or (len(all_sink_paths) > 0 and len(source_paths) == 1)):
        # TODO: What about a source path?
        arrnode = sink_paths[0]
        # Only works if the stream itself is not an array of streams
        if list(node.desc(sdfg).shape) == [1]:
            node.desc(sdfg).sink = arrnode.data  # For memlet generation
            arrnode.desc(sdfg).src = node.data  # TODO: Move src/sink to node, not array
            return True
    return False


def get_view_node(state: SDFGState, view: nd.AccessNode) -> nd.AccessNode:
    """
    Given a view access node, returns the viewed access node
    if existent, else None
    """
    view_edge = get_view_edge(state, view)
    if view_edge is None:
        return None
    if view_edge.dst == view:
        return view_edge.src
    else:
        return view_edge.dst


def get_last_view_node(state: SDFGState, view: nd.AccessNode) -> nd.AccessNode:
    """
    Given a view access node, returns the last viewed access node
    if existent, else None
    """
    sdfg = state.parent
    node = view
    desc = sdfg.arrays[node.data]
    while isinstance(desc, dt.View):
        node = get_view_node(state, node)
        if node is None or not isinstance(node, nd.AccessNode):
            return None
        desc = sdfg.arrays[node.data]
    return node


def get_all_view_nodes(state: SDFGState, view: nd.AccessNode) -> List[nd.AccessNode]:
    """
    Given a view access node, returns a list of viewed access nodes
    if existent, else None
    """
    sdfg = state.parent
    node = view
    desc = sdfg.arrays[node.data]
    result = [node]
    while isinstance(desc, dt.View):
        node = get_view_node(state, node)
        if node is None or not isinstance(node, nd.AccessNode):
            return None
        desc = sdfg.arrays[node.data]
        result.append(node)
    return result


def get_all_view_edges(state: SDFGState, view: nd.AccessNode) -> List[gr.MultiConnectorEdge[mm.Memlet]]:
    """
    Given a view access node, returns a list of viewed access nodes as edges
    if existent, else None
    """
    sdfg = state.parent
    previous_node = view
    result = []

    desc = sdfg.arrays[previous_node.data]
    forward = None
    while isinstance(desc, dt.View):
        edge = get_view_edge(state, previous_node)
        if edge is None:
            break

        if forward is None:
            forward = edge.src is previous_node

        if forward:
            next_node = edge.dst
        else:
            next_node = edge.src

        if previous_node is next_node:
            break
        if not isinstance(next_node, nd.AccessNode):
            break
        desc = sdfg.arrays[next_node.data]
        result.append(edge)
        previous_node = next_node

    return result


def get_view_edge(state: SDFGState, view: nd.AccessNode) -> gr.MultiConnectorEdge[mm.Memlet]:
    """
    Given a view access node, returns the
    incoming/outgoing edge which points to the viewed access node.
    See the ruleset in the documentation of ``dace.data.View``.

    :param state: The state in which the view resides.
    :param view: The view access node.
    :return: An edge pointing to the viewed data or None if view is invalid.
    :see: ``dace.data.View``
    """

    in_edges = state.in_edges(view)
    # We should ignore empty synchronization edges
    in_edges = [e for e in in_edges if not e.data.is_empty()]
    out_edges = state.out_edges(view)

    # Invalid case: No data to view
    if len(in_edges) == 0 and len(out_edges) == 0:
        return None

    # If there is one edge (in/out) that leads (via memlet path) to an access
    # node, and the other side (out/in) has a different number of edges.
    if len(in_edges) == 1 and len(out_edges) != 1:
        # If the edge is not leading to an access node, fail
        mpath = state.memlet_path(in_edges[0])
        if not isinstance(mpath[0].src, nd.AccessNode):
            return None

        return in_edges[0]
    if len(out_edges) == 1 and len(in_edges) != 1:
        # If the edge is not leading to an access node, fail
        mpath = state.memlet_path(out_edges[0])
        if not isinstance(mpath[-1].dst, nd.AccessNode):
            return None

        return out_edges[0]
    if len(out_edges) == len(in_edges) and len(out_edges) != 1:
        return None

    in_edge = in_edges[0]
    out_edge = out_edges[0] if len(out_edges) > 0 else None

    # If there is one incoming and one outgoing edge, and one leads to a code
    # node, the one that leads to an access node is the viewed data.
    inmpath = state.memlet_path(in_edge)
    outmpath = state.memlet_path(out_edge) if out_edge else None
    src_is_data, dst_is_data = False, False
    if isinstance(inmpath[0].src, nd.AccessNode):
        src_is_data = True
    if outmpath and isinstance(outmpath[-1].dst, nd.AccessNode):
        dst_is_data = True

    if src_is_data and not dst_is_data:
        return in_edge
    if not src_is_data and dst_is_data:
        return out_edge
    if not src_is_data and not dst_is_data:
        return None

    # Check if there is a 'views' connector
    if in_edge.dst_conn and in_edge.dst_conn == 'views':
        return in_edge
    if out_edge.src_conn and out_edge.src_conn == 'views':
        return out_edge

    # TODO: This sounds arbitrary and is not well communicated to the frontends. Revisit in the future.
    # If both sides lead to access nodes, if one memlet's data points to the view it cannot point to the viewed node.
    if in_edge.data.data == view.data and out_edge.data.data != view.data:
        return out_edge
    if in_edge.data.data != view.data and out_edge.data.data == view.data:
        return in_edge
    if in_edge.data.data == view.data and out_edge.data.data == view.data:
        return None

    # If both memlets' data are the respective access nodes, the access
    # node at the highest scope is the one that is viewed.
    if isinstance(in_edge.src, nd.EntryNode):
        return in_edge
    if isinstance(out_edge.dst, nd.ExitNode):
        return out_edge

    # If both access nodes reside in the same scope, the input data is viewed.
    warnings.warn(f"Ambiguous view: in_edge {in_edge} -> view {view.data} -> out_edge {out_edge}")
    return in_edge


def dynamic_map_inputs(state: SDFGState, map_entry: nd.MapEntry) -> List[gr.MultiConnectorEdge]:
    """
    For a given map entry node, returns a list of dynamic-range input edges.

    :param state: The state in which the map entry node resides.
    :param map_entry: The given node.
    :return: A list of edges in state whose destination is map entry and denote
             dynamic-range input memlets.
    """
    return [e for e in state.in_edges(map_entry) if e.dst_conn and not e.dst_conn.startswith('IN_')]


def has_dynamic_map_inputs(state: SDFGState, map_entry: nd.MapEntry) -> bool:
    """
    Returns True if a map entry node has dynamic-range inputs.

    :param state: The state in which the map entry node resides.
    :param map_entry: The given node.
    :return: True if there are dynamic-range input memlets, False otherwise.
    """
    return len(dynamic_map_inputs(state, map_entry)) > 0


def is_parallel(state: SDFGState, node: Optional[nd.Node] = None) -> bool:
    """
    Returns True if a node or state are contained within a parallel
    section.

    :param state: The state to test.
    :param node: An optional node in the state to test. If None, only checks
                 state.
    :return: True if the state or node are located within a map scope that
             is scheduled to run in parallel, False otherwise.
    """
    if node is not None:
        sdict = state.scope_dict()
        curnode = node
        while curnode is not None:
            curnode = sdict[curnode]
            if curnode.schedule != dtypes.ScheduleType.Sequential:
                return True
    if state.parent.parent is not None:
        # Find nested SDFG node and continue recursion
        nsdfg_node = next(n for n in state.parent.parent if isinstance(n, nd.NestedSDFG) and n.sdfg == state.parent)
        return is_parallel(state.parent.parent, nsdfg_node)

    return False


def find_input_arraynode(graph, edge):
    result = graph.memlet_path(edge)[0]
    if not isinstance(result.src, nd.AccessNode):
        raise RuntimeError("Input array node not found for memlet " + str(edge.data))
    return result.src


def find_output_arraynode(graph, edge):
    result = graph.memlet_path(edge)[-1]
    if not isinstance(result.dst, nd.AccessNode):
        raise RuntimeError("Output array node not found for memlet " + str(edge.data))
    return result.dst


def weakly_connected_component(dfg, node_in_component: Node) -> StateSubgraphView:
    """
    Returns a subgraph of all nodes that form the weakly connected component in
    `dfg` that contains `node_in_component`.
    """
    seen = set()
    to_search = [node_in_component]
    while to_search:
        node = to_search.pop()
        if node in seen:
            continue
        seen.add(node)
        for succ in dfg.successors(node):
            to_search.append(succ)
    to_search = [node_in_component]
    seen.remove(node_in_component)
    while to_search:
        node = to_search.pop()
        if node in seen:
            continue
        seen.add(node)
        for succ in dfg.predecessors(node):
            to_search.append(succ)
    subgraph = StateSubgraphView(dfg, seen)
    return subgraph


def concurrent_subgraphs(graph):
    """ Finds subgraphs of an SDFGState or ScopeSubgraphView that can
        run concurrently. """
    from dace.sdfg.scope import ScopeSubgraphView

    if not isinstance(graph, (SDFGState, ScopeSubgraphView)):
        raise TypeError("Expected SDFGState or ScopeSubgraphView, got: {}".format(type(graph).__name__))
    candidates = graph.source_nodes()
    components = collections.OrderedDict()  # {start node: nodes in component}
    for cand in candidates:
        if isinstance(cand, nd.AccessNode):
            # AccessNodes can be read from multiple concurrent components, so
            # check all out edges
            start_nodes = [e.dst for e in graph.out_edges(cand)]
            for n in start_nodes:
                if n not in components:
                    components[n] = {cand, n}
                else:
                    # Components can read from multiple start arrays
                    components[n].add(cand)
        else:
            # The source node == the first control or compute node
            components[cand] = {cand}
    subgraphs = []  # [{nodes in subgraph}]
    for i, start_node in enumerate(components):
        # Do BFS and find all nodes reachable from this start node
        seen = set()
        to_search = [start_node]
        while len(to_search) > 0:
            node = to_search.pop()
            if node in seen:
                continue
            seen.add(node)
            for e in graph.out_edges(node):
                if e.dst not in seen:
                    to_search.append(e.dst)
        # If this component overlaps with any previously determined components,
        # fuse them
        to_delete = []
        for i, other in enumerate(subgraphs):
            if len(other & seen) > 0:
                to_delete.append(i)
        if len(to_delete) == 0:
            # If there was no overlap, this is a concurrent subgraph
            subgraphs.append(seen | components[start_node])
        else:
            # Merge overlapping subgraphs
            new_subgraph = seen | components[start_node]

            for index in reversed(to_delete):
                new_subgraph |= subgraphs.pop(index)

            subgraphs.append(new_subgraph)

    # Now stick each of the found components in a ScopeSubgraphView and return
    # them. Sort according to original order of nodes
    all_nodes = graph.nodes()
    return [ScopeSubgraphView(graph, [n for n in all_nodes if n in sg], None) for sg in subgraphs]


def separate_maps(state, dfg, schedule):
    """ Separates the given ScopeSubgraphView into subgraphs with and without
        maps of the given schedule type. The function assumes that the given
        ScopeSubgraph view does not contain any concurrent segments (i.e. pass
        it through concurrent_subgraphs first). Only top level maps will be
        accounted for, if the desired schedule occurs in another (undesired)
        map, it will be ignored.

        Returns a list with the subgraph views in order of the original DFG.
        ScopeSubgraphViews for the parts with maps, StateSubgraphViews for the
        parts without maps. """

    from dace import nodes
    from dace.sdfg.scope import StateSubgraphView

    sorted_nodes = list(dfs_topological_sort(dfg, dfg.source_nodes()[0]))
    nodes_to_skip = [dfg.source_nodes()[0], dfg.sink_nodes()[0]]
    result = []

    current = []
    for node in sorted_nodes:
        if node in nodes_to_skip:
            continue
        if isinstance(node, nodes.MapEntry):
            if node.map.schedule == schedule:
                result.append(StateSubgraphView(state, current))
                result.append(state.scope_subgraph(node))
                nodes_to_skip += result[-1].nodes()
                current = []
            else:
                temp_nodes = state.scope_subgraph(node).nodes()
                nodes_to_skip += temp_nodes
                current += temp_nodes
        else:
            current.append(node)

    if len(current) > 0:
        result.append(StateSubgraphView(state, current))

    return result


def _transients_in_scope(sdfg, outer_scope, scope_dict, include_nested):
    scopes = [outer_scope.entry]
    transients = set()
    while scopes:
        scope = scopes.pop()
        for node in scope_dict[scope]:
            if (isinstance(node, nd.AccessNode) and sdfg.arrays[node.data].transient):
                transients.add(node.data)
            if (isinstance(node, nd.EntryNode) and node is not scope and include_nested):
                # "Recurse" into nested scopes
                scopes.append(node)
        if not include_nested:
            # Only run the first iteration of the while loop
            break
    return transients


def local_transients(sdfg, dfg, entry_node, include_nested=False):
    """
    Returns transients local to the scope defined by the specified entry node in
    the dataflow graph.

    :param entry_node: The entry node that opens the scope. If `None`, the
                       top-level scope is used.
    :param include_nested: Include transients defined in nested scopes.
    """
    state: SDFGState = dfg._graph
    scope_children = state.scope_children()
    scope_tree = state.scope_tree()
    current_scope = scope_tree[entry_node]

    # Start by setting shared transients as defined
    defined_transients = set(sdfg.shared_transients())

    # Get access nodes in current scope
    transients = _transients_in_scope(sdfg, current_scope, scope_children, include_nested)

    # Add transients defined in parent scopes
    while current_scope.parent is not None:
        current_scope = current_scope.parent
        defined_transients.update(_transients_in_scope(sdfg, current_scope, scope_children, False))

    return sorted(list(transients - defined_transients))


def trace_nested_access(node: nd.AccessNode, state: SDFGState,
                        sdfg: SDFG) -> List[Tuple[nd.AccessNode, SDFGState, SDFG]]:
    """
    Given an AccessNode in a nested SDFG, trace the accessed memory
    back to the outermost scope in which it is defined.

    :param node: An access node.
    :param state: State in which the access node is located.
    :param sdfg: SDFG in which the access node is located.
    :return: A list of scopes ((input_node, output_node), (memlet_read, memlet_write), state, sdfg) in which
             the given data is accessed, from outermost scope to innermost
             scope.
    """
    curr_sdfg = sdfg
    curr_read = None
    memlet_read = None
    for m in state.out_edges(node):
        if not m.data.is_empty():
            curr_read = node
            memlet_read = m.data
            break

    curr_write = None
    memlet_write = None
    for m in state.in_edges(node):
        if not m.data.is_empty():
            curr_write = node
            memlet_read = m.data
            break

    trace = [((curr_read, curr_write), (memlet_read, memlet_write), state, sdfg)]

    while curr_sdfg.parent is not None:
        curr_state = curr_sdfg.parent

        # Find the nested SDFG containing ourself in the parent state
        nested_sdfg = curr_sdfg.parent_nsdfg_node

        if curr_read is not None:
            for e in curr_state.in_edges(nested_sdfg):
                if e.dst_conn == curr_read.data:
                    # See if the input to this connector traces back to an
                    # access node. If not, just give up here
                    n = find_input_arraynode(curr_state, e)
                    if isinstance(n, nd.AccessNode):
                        curr_read = n
                        memlet_read = e.data
                        break
            else:
                curr_read = None
                memlet_read = None
        if curr_write is not None:
            for e in curr_state.out_edges(nested_sdfg):
                if e.src_conn == curr_write.data:
                    # See if the output of this connector traces back to an
                    # access node. If not, just give up here
                    n = find_output_arraynode(curr_state, e)
                    if isinstance(curr_write, nd.AccessNode):
                        curr_write = n
                        memlet_write = e.data
                        break
            else:
                curr_write = None
                memlet_write = None
        if curr_read is not None or curr_write is not None:
            trace.append(((curr_read, curr_write), (memlet_read, memlet_write), curr_state, curr_state.parent))
        else:
            break
        curr_sdfg = curr_state.parent  # Recurse
    return list(reversed(trace))


def fuse_states(sdfg: SDFG, permissive: bool = False, progress: bool = None) -> int:
    """
    Fuses all possible states of an SDFG (and all sub-SDFGs) using an optimized
    routine that uses the structure of the StateFusion transformation.

    :param sdfg: The SDFG to transform.
    :param permissive: If True, operates in permissive mode, which ignores some
                       race condition checks.
    :param progress: If True, prints out a progress bar of fusion (may be
                     inaccurate, requires ``tqdm``). If None, prints out
                     progress if over 5 seconds have passed. If False, never
                     shows progress bar.
    :return: The total number of states fused.
    """
    from dace.transformation.interstate import StateFusion, BlockFusion  # Avoid import loop

    if progress is None and not config.Config.get_bool('progress'):
        progress = False

    if progress is True or progress is None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

    counter = 0
    if progress is True or progress is None:
        fusible_states = 0
        for cfg in sdfg.all_control_flow_regions():
            fusible_states += cfg.number_of_edges()

    if progress is True:
        pbar = tqdm(total=fusible_states, desc='Fusing states')

    start = time.time()

    for sd in sdfg.all_sdfgs_recursive():
        for cfg in sd.all_control_flow_regions():
            while True:
                edges = list(cfg.nx.edges)
                applied = 0
                skip_nodes = set()
                for u, v in edges:
                    if (progress is None and tqdm is not None and (time.time() - start) > 5):
                        progress = True
                        pbar = tqdm(total=fusible_states, desc='Fusing states', initial=counter)

                    if u in skip_nodes or v in skip_nodes:
                        continue

                    if isinstance(u, SDFGState) and isinstance(v, SDFGState):
                        candidate = {StateFusion.first_state: u, StateFusion.second_state: v}
                        sf = StateFusion()
                        sf.setup_match(cfg, cfg.cfg_id, -1, candidate, 0, override=True)
                        if sf.can_be_applied(cfg, 0, sd, permissive=permissive):
                            sf.apply(cfg, sd)
                            applied += 1
                            counter += 1
                            if progress:
                                pbar.update(1)
                            skip_nodes.add(u)
                            skip_nodes.add(v)
                    else:
                        candidate = {BlockFusion.first_block: u, BlockFusion.second_block: v}
                        bf = BlockFusion()
                        bf.setup_match(cfg, cfg.cfg_id, -1, candidate, 0, override=True)
                        if bf.can_be_applied(cfg, 0, sd, permissive=permissive):
                            bf.apply(cfg, sd)
                            applied += 1
                            counter += 1
                            if progress:
                                pbar.update(1)
                            skip_nodes.add(u)
                            skip_nodes.add(v)
                if applied == 0:
                    break
    if progress:
        pbar.close()
    return counter


def inline_control_flow_regions(sdfg: SDFG,
                                types: Optional[List[Type[AbstractControlFlowRegion]]] = None,
                                ignore_region_types: Optional[List[Type[AbstractControlFlowRegion]]] = None,
                                progress: bool = None,
                                lower_returns: bool = False,
                                eliminate_dead_states: bool = False) -> int:
    if types:
        blocks = [n for n, _ in sdfg.all_nodes_recursive() if type(n) in types]
    elif ignore_region_types:
        blocks = [
            n for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, AbstractControlFlowRegion) and type(n) not in ignore_region_types
        ]
    else:
        blocks = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, AbstractControlFlowRegion)]
    count = 0

    for _block in optional_progressbar(reversed(blocks),
                                       title='Inlining control flow regions',
                                       n=len(blocks),
                                       progress=progress):
        block: ControlFlowRegion = _block
        # Control flow regions where the parent is a conditional block are not inlined.
        if block.parent_graph and type(block.parent_graph) == ConditionalBlock:
            continue
        if block.inline(lower_returns=lower_returns)[0]:
            count += 1
    if eliminate_dead_states:
        # Avoid cyclic imports.
        from dace.transformation.passes.dead_state_elimination import DeadStateElimination
        DeadStateElimination().apply_pass(sdfg, {})

    sdfg.reset_cfg_list()

    return count


def inline_sdfgs(sdfg: SDFG, permissive: bool = False, progress: bool = None, multistate: bool = True) -> int:
    """
    Inlines all possible nested SDFGs (or sub-SDFGs) using an optimized
    routine that uses the structure of the SDFG hierarchy.

    :param sdfg: The SDFG to transform.
    :param permissive: If True, operates in permissive mode, which ignores some
                       checks.
    :param progress: If True, prints out a progress bar of inlining (may be
                     inaccurate, requires ``tqdm``). If None, prints out
                     progress if over 5 seconds have passed. If False, never
                     shows progress bar.
    :param multistate: Include
    :return: The total number of SDFGs inlined.
    """
    # Avoid import loops
    from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG

    counter = 0
    nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, NestedSDFG)]

    for nsdfg_node in optional_progressbar(reversed(nsdfgs), title='Inlining SDFGs', n=len(nsdfgs), progress=progress):
        # We have to reevaluate every time due to changing IDs
        # e.g., InlineMultistateSDFG may fission states
        nsdfg: SDFG = nsdfg_node.sdfg
        parent_state = nsdfg.parent
        parent_sdfg = parent_state.sdfg
        parent_state_id = parent_state.block_id

        if multistate:
            candidate = {
                InlineMultistateSDFG.nested_sdfg: nsdfg_node,
            }
            inliner = InlineMultistateSDFG()
            inliner.setup_match(sdfg=parent_sdfg,
                                cfg_id=parent_state.parent_graph.cfg_id,
                                state_id=parent_state_id,
                                subgraph=candidate,
                                expr_index=0,
                                override=True)
            if inliner.can_be_applied(parent_state, 0, parent_sdfg, permissive=permissive):
                inliner.apply(parent_state, parent_sdfg)
                counter += 1
                continue

        candidate = {
            InlineSDFG.nested_sdfg: nsdfg_node,
        }
        inliner = InlineSDFG()
        inliner.setup_match(sdfg=parent_sdfg,
                            cfg_id=parent_state.parent_graph.cfg_id,
                            state_id=parent_state_id,
                            subgraph=candidate,
                            expr_index=0,
                            override=True)
        if inliner.can_be_applied(parent_state, 0, parent_sdfg, permissive=permissive):
            inliner.apply(parent_state, parent_sdfg)
            counter += 1

    return counter


def load_precompiled_sdfg(folder: str, argnames: Optional[List[str]] = None) -> csdfg.CompiledSDFG:
    """
    Loads a pre-compiled SDFG from an output folder (e.g. ".dacecache/program").
    Folder must contain a file called "program.sdfg" and a subfolder called
    "build" with the shared object.

    :param folder: Path to SDFG output folder.
    :param argnames: Names of arguments of the compiled SDFG.
    :return: A callable CompiledSDFG object.
    """
    sdfg = SDFG.from_file(os.path.join(folder, 'program.sdfg'))
    suffix = config.Config.get('compiler', 'library_extension')
    return csdfg.CompiledSDFG(
        sdfg,
        csdfg.ReloadableDLL(os.path.join(folder, 'build', f'lib{sdfg.name}.{suffix}'), sdfg.name),
        argnames,
    )


def distributed_compile(sdfg: SDFG, comm, validate: bool = True) -> csdfg.CompiledSDFG:
    """
    Compiles an SDFG in rank 0 of MPI communicator ``comm``. Then, the compiled SDFG is loaded in all other ranks.

    :param sdfg: SDFG to be compiled.
    :param comm: MPI communicator. ``Intracomm`` is the base mpi4py communicator class.
    :return: Compiled SDFG.
    :note: This method can be used only if the module mpi4py is installed.
    """

    rank = comm.Get_rank()
    func = None
    folder = None

    # Rank 0 compiles SDFG.
    if rank == 0:
        func = sdfg.compile(validate=validate)
        folder = sdfg.build_folder

    # Broadcasts build folder.
    folder = comm.bcast(folder, root=0)

    # Loads compiled SDFG.
    if rank > 0:
        func = load_precompiled_sdfg(folder)

    comm.Barrier()

    return func


def get_next_nonempty_states(sdfg: SDFG, state: SDFGState) -> Set[SDFGState]:
    """
    From the given state, return the next set of states that are reachable
    in the SDFG, skipping empty states. Traversal stops at the non-empty
    state.

    This function is used to determine whether synchronization should happen
    at the end of a GPU state.

    :param sdfg: The SDFG that contains the state.
    :param state: The state to start from.
    :return: A set of reachable non-empty states.
    """
    result: Set[SDFGState] = set()

    # Traverse children until states are not empty
    for succ in state.parent_graph.successors(state):
        result |= set(
            dfs_conditional(state.parent_graph,
                            sources=[succ],
                            condition=lambda parent, _: parent.number_of_nodes() == 0))

    # Filter out empty states
    result = {s for s in result if not s.number_of_nodes() == 0}

    return result


def unique_node_repr(graph: Union[SDFGState, ScopeSubgraphView], node: Node) -> str:
    """
    Returns unique string representation of the given node,
    considering its placement into the SDFG graph.
    Useful for hashing, or building node-based dictionaries.

    :param graph: the state/subgraph that contains the node
    :param node: node to represent
    :return: the unique representation
    """

    # Build a unique representation
    sdfg = graph.parent
    state = graph if isinstance(graph, SDFGState) else graph._graph
    return str(sdfg.cfg_id) + "_" + str(sdfg.node_id(state)) + "_" + str(state.node_id(node))


def is_nonfree_sym_dependent(node: nd.AccessNode, desc: dt.Data, state: SDFGState, fsymbols: Set[str]) -> bool:
    """
    Checks whether the Array or View descriptor is non-free symbol dependent.
    An Array is non-free symbol dependent when its attributes (e.g., shape)
    depend on non-free symbols. A View is non-free symbol dependent when either
    its adjacent edges or its viewed node depend on non-free symbols.

    :param node: the access node to check
    :param desc: the data descriptor to check
    :param state: the state that contains the node
    :param fsymbols: the free symbols to check against
    """
    if isinstance(desc, (dt.View)):
        # Views can be non-free symbol dependent due to the adjacent edges.
        e = get_view_edge(state, node)
        if e.data:
            src_subset = e.data.get_src_subset(e, state)
            dst_subset = e.data.get_dst_subset(e, state)
            free_symbols = set()
            if src_subset:
                free_symbols |= src_subset.free_symbols
            if dst_subset:
                free_symbols |= dst_subset.free_symbols
            if any(str(s) not in fsymbols for s in free_symbols):
                return True
        # If the viewed node/descriptor is non-free symbol dependent, then so
        # is the View.
        n = get_view_node(state, node)
        if n and isinstance(n, nd.AccessNode):
            d = state.parent.arrays[n.data]
            return is_nonfree_sym_dependent(n, d, state, fsymbols)
    elif isinstance(desc, dt.Array):
        if any(str(s) not in fsymbols for s in desc.free_symbols):
            return True
    return False


def _tswds_state(
    sdfg: SDFG,
    state: SDFGState,
    symbols: Dict[str, dtypes.typeclass],
    recursive: bool,
) -> Generator[Tuple[SDFGState, Node, Dict[str, dtypes.typeclass]], None, None]:
    """
    Helper function for ``traverse_sdfg_with_defined_symbols``.

    :see: traverse_sdfg_with_defined_symbols.
    """
    # Traverse state by scopes
    sdict = state.scope_children()

    def _traverse(scope: Node, symbols: Dict[str, dtypes.typeclass]):
        for node in sdict[scope]:
            yield state, node, symbols
            # Traverse inside scopes
            if node in sdict:
                inner_syms = {}
                inner_syms.update(symbols)
                inner_syms.update(node.new_symbols(sdfg, state, inner_syms))
                yield from _traverse(node, inner_syms)
            if isinstance(node, dace.sdfg.nodes.NestedSDFG) and recursive:
                yield from traverse_sdfg_with_defined_symbols(node.sdfg, recursive)

    # Start with top-level nodes
    yield from _traverse(None, symbols)


def _tswds_cf_region(
        sdfg: SDFG,
        cfg: AbstractControlFlowRegion,
        symbols: Dict[str, dtypes.typeclass],
        recursive: bool = False) -> Generator[Tuple[SDFGState, Node, Dict[str, dtypes.typeclass]], None, None]:
    sub_regions = cfg.sub_regions() or [cfg]
    for region in sub_regions:
        # Add symbols newly defined by this region, if present.
        region_symbols = region.new_symbols(symbols)
        symbols.update({k: v for k, v in region_symbols.items() if v is not None})

        # Add symbols from inter-state edges along the state machine
        start_region = region.start_block
        visited = set()
        visited_edges = set()
        for edge in region.dfs_edges(start_region):
            # Source -> inter-state definition -> Destination
            visited_edges.add(edge)
            # Source
            if edge.src not in visited:
                visited.add(edge.src)
                if isinstance(edge.src, SDFGState):
                    yield from _tswds_state(sdfg, edge.src, symbols, recursive)
                elif isinstance(edge.src, AbstractControlFlowRegion):
                    yield from _tswds_cf_region(sdfg, edge.src, symbols, recursive)

            # Add edge symbols into defined symbols
            issyms = edge.data.new_symbols(sdfg, symbols)
            symbols.update({k: v for k, v in issyms.items() if v is not None})

            # Destination
            if edge.dst not in visited:
                visited.add(edge.dst)
                if isinstance(edge.dst, SDFGState):
                    yield from _tswds_state(sdfg, edge.dst, symbols, recursive)
                elif isinstance(edge.dst, AbstractControlFlowRegion):
                    yield from _tswds_cf_region(sdfg, edge.dst, symbols, recursive)

        # If there is only one state, the DFS will miss it
        if start_region not in visited:
            if isinstance(start_region, SDFGState):
                yield from _tswds_state(sdfg, start_region, symbols, recursive)
            elif isinstance(start_region, AbstractControlFlowRegion):
                yield from _tswds_cf_region(sdfg, start_region, symbols, recursive)


def traverse_sdfg_with_defined_symbols(
        sdfg: SDFG,
        recursive: bool = False) -> Generator[Tuple[SDFGState, Node, Dict[str, dtypes.typeclass]], None, None]:
    """
    Traverses the SDFG, its states and nodes, yielding the defined symbols and their types at each node.

    :return: A generator that yields tuples of (state, node in state, currently-defined symbols)
    """
    # Start with global symbols and scalar constants
    symbols = copy.copy(sdfg.symbols)
    symbols.update({k: desc.dtype for k, (desc, _) in sdfg.constants_prop.items() if isinstance(desc, dt.Scalar)})
    for desc in sdfg.arrays.values():
        symbols.update({str(s): s.dtype for s in desc.free_symbols})

    yield from _tswds_cf_region(sdfg, sdfg, symbols, recursive)


CFBlockDictT = Dict[ControlFlowBlock, ControlFlowBlock]


def postdominators(
    cfg: ControlFlowRegion,
    return_alldoms: bool = False
) -> Optional[Union[CFBlockDictT, Tuple[CFBlockDictT, Dict[ControlFlowBlock, Set[ControlFlowBlock]]]]]:
    """
    Return the immediate postdominators of a CFG. This may require creating new nodes and removing them, which
    happens in-place on the CFG.

    :param cfg: The CFG to generate the postdominators from.
    :param return_alldoms: If True, returns the "all postdominators" dictionary as well.
    :return: Immediate postdominators, or a 2-tuple of (ipostdom, allpostdoms) if ``return_alldoms`` is True.
    """
    from dace.sdfg.analysis import cfg as cfg_analysis

    # Get immediate post-dominators
    sink_nodes = cfg.sink_nodes()
    if len(sink_nodes) > 1:
        sink = cfg.add_state()
        for snode in sink_nodes:
            cfg.add_edge(snode, sink, dace.InterstateEdge())
    elif len(sink_nodes) == 0:
        return None
    else:
        sink = sink_nodes[0]
    ipostdom: CFBlockDictT = nx.immediate_dominators(cfg._nx.reverse(), sink)

    if return_alldoms:
        allpostdoms = cfg_analysis.all_dominators(cfg, ipostdom)
        retval = (ipostdom, allpostdoms)
    else:
        retval = ipostdom

    # If a new sink was added for post-dominator computation, remove it
    if len(sink_nodes) > 1:
        cfg.remove_node(sink)

    return retval


def map_view_to_array(vdesc: dt.View, adesc: dt.Array,
                      subset: sbs.Range) -> Optional[Tuple[Dict[int, int], List[int], List[int]]]:
    """
    Finds the matching dimensions mapping between a data descriptor and a view reinterpreting it, if and only
    if the view represents a slice (with potential new, "unsqueezed" axes).
    Views have the following relationship (w.l.o.g.): (array) --subset--> (view). For every memlet that goes
    out of a view, we need to compose the subset with the new view dimensions and new subset.
    The precondition to this method is that the array has unique strides (if not, the process fails).
    The process works in three steps, as follows:
        * First, The degenerate (shape=1) dimensions are removed from both the array and the view for consideration.
        * The mapping between non-degenerate dimensions is done from the view to the array based on the strides.
            Note that in a slice, the strides can be expanded or squeezed, but never reordered. This fact is used
            during matching. If any non-degenerate dimension remains in the view, the process fails.
        * Second, we find the "unsqueezed" dimensions by looking at the remainder of the view dimensions:
            any dimension that is between the dimensions in the existing mapping is considered for strides. Dimensions
            that fall before or after the sizes, or between two consecutive dimensions, are considered new axes.
        * Third, the remainder of the dimensions of the original (non-view) data descriptor are considered
            "squeezed".

    For example, a scalar view ``A[i, j] -> v`` would return ``({}, [], [0, 1])``.
    Example 2: ``A[0:2, 3:5, i, j, 0:N] -> V[0:2, 0, 0:2, 0, 0:N, 0]`` would return
    ``({0: 0, 2: 1, 3: 2, 4: 4}, [1, 5], [3])``.
    :param vdesc: The data descriptor of the view.
    :param adesc: The data descriptor of the viewed data container.
    :return: A tuple of (mapping of view->array, expanded, squeezed) dimensions, or None if the process failed.
    """

    # Strides can be squeezed or expanded, but never reordered.
    # traverse both shapes and strides, ignoring shape-1 dimensions along the way
    dimension_mapping: Dict[int, int] = {}
    unsqueezed: List[int] = []
    squeezed: List[int] = []

    # First, remove shape=1 dimensions (unsqueezed or squeezed)
    non_squeeze_vdims = [i for i, s in enumerate(vdesc.shape) if s != 1]
    non_squeeze_adims = [i for i, s in enumerate(adesc.shape) if s != 1]
    astrides = [adesc.strides[i] for i in non_squeeze_adims]

    # Find matching strides
    last_adim = 0
    for i in non_squeeze_vdims:
        try:
            last_adim = astrides.index(vdesc.strides[i], last_adim)
            dimension_mapping[i] = non_squeeze_adims[last_adim]
        except ValueError:  # Index not found
            return None

    # Find degenerate dimension mapping (stride-matching and new axes / "unsqueezed")
    dims_iter = iter(sorted(dimension_mapping.items()))  # Sorted matched dimensions
    prev_dim = 0
    next_dim = next(dims_iter, (-1, -1))[1]  # First matched dimension in data container
    new_dims: Dict[int, int] = {}
    for i, vstride in enumerate(vdesc.strides):
        if i not in dimension_mapping:
            try:
                if next_dim < 0:
                    match = adesc.strides.index(vstride, prev_dim)
                else:
                    match = adesc.strides.index(vstride, prev_dim, next_dim)
                new_dims[i] = match
            except ValueError:  # No match found - new axis
                unsqueezed.append(i)
        else:
            prev_dim = dimension_mapping[i] + 1

            # If we are out of dimensions, return -1 so that anything further is unsqueezed
            next_dim = next(dims_iter, (-1, -1))[1]

    # Add new mappings after the loop to avoid interfering with it
    dimension_mapping.update(new_dims)

    # Find degenerate dimension mapping in remainder of data container (squeezed)
    subset_size = subset.size() if subset is not None else None
    inverse_dim_mapping = {v: k for k, v in dimension_mapping.items()}
    for i in range(len(adesc.shape)):
        if i not in inverse_dim_mapping:
            if subset_size is not None and subset_size[i] != 1:
                # A squeezed dimension must have a subset of size = 1 on the source data container
                return None
            squeezed.append(i)

    return dimension_mapping, unsqueezed, squeezed


def check_sdfg(sdfg: SDFG):
    """ Checks that the parent attributes of an SDFG are correct.

    :param sdfg: The SDFG to check.
    :raises AssertionError: If any of the parent attributes are incorrect.
    """
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                assert node.sdfg.parent_nsdfg_node is node
                assert node.sdfg.parent is state
                assert node.sdfg.parent_sdfg is sdfg
                assert node.sdfg.parent.parent is sdfg
                check_sdfg(node.sdfg)


def normalize_offsets(sdfg: SDFG):
    """
    Normalizes descriptor offsets to 0 and adjusts the Memlet subsets accordingly. This operation is done in-place.

    :param sdfg: The SDFG to be normalized.
    """

    import ast
    from dace.frontend.python import astutils

    for sd in sdfg.all_sdfgs_recursive():
        offsets = dict()
        for arrname, arrdesc in sd.arrays.items():
            if not isinstance(arrdesc, dt.Array):  # NOTE: Does this work with Views properly?
                continue
            if any(o != 0 for o in arrdesc.offset):
                offsets[arrname] = arrdesc.offset
                arrdesc.offset = [0] * len(arrdesc.shape)
        if offsets:
            for e in sd.edges():
                memlets = e.data.get_read_memlets(sd.arrays)
                for m in memlets:
                    if m.data in offsets:
                        m.subset.offset(offsets[m.data], False)
                for node in ast.walk(e.data.condition.code[0]):
                    if isinstance(node, ast.Subscript):
                        m = memlets.pop(0)
                        subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                        assert isinstance(node.value, ast.Name) and node.value.id == m.data
                        node.slice = ast.copy_location(subscript.slice, node.slice)
                e.data._cond_sympy = None
                for k, v in e.data.assignments.items():
                    vast = ast.parse(v)
                    for node in ast.walk(vast):
                        if isinstance(node, ast.Subscript):
                            m = memlets.pop(0)
                            subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                            assert isinstance(node.value, ast.Name) and node.value.id == m.data
                            node.slice = ast.copy_location(subscript.slice, node.slice)
                    newv = astutils.unparse(vast)
                    e.data.assignments[k] = newv
                assert not memlets
            for state in sd.states():
                # NOTE: Ideally, here we just want to iterate over the edges. However, we need to handle both the
                # subset and the other subset. Therefore, it is safer to traverse the Memlet paths.
                for node in state.nodes():
                    if isinstance(node, nd.AccessNode) and node.data in offsets:
                        off = offsets[node.data]
                        visited = set()
                        for e0 in state.all_edges(node):
                            for e1 in state.memlet_tree(e0):
                                if e1 in visited:
                                    continue
                                visited.add(e1)
                                if e1.data.data == node.data:
                                    e1.data.subset.offset(off, False)
                                else:
                                    e1.data.other_subset.offset(off, False)


def prune_symbols(sdfg: SDFG):
    """
    Prunes unused symbols from the SDFG and the NestedSDFG symbol mappings. This operation is done in place. See also
    `dace.transformation.interstate.PruneSymbols`.

    :param sdfg: The SDFG to have its symbols pruned.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG):
                prune_symbols(node.sdfg)
                declared_symbols = set(node.sdfg.symbols.keys())
                free_symbols = node.sdfg.free_symbols
                defined_symbols = declared_symbols - free_symbols
                for s in defined_symbols:
                    del node.sdfg.symbols[s]
                    if s in node.symbol_mapping:
                        del node.symbol_mapping[s]


def make_dynamic_map_inputs_unique(sdfg: SDFG):
    for sd in sdfg.all_sdfgs_recursive():
        dynamic_map_inputs = set(sd.arrays.keys())
        for state in sd.states():
            for node in state.nodes():
                repl_dict = {}
                if isinstance(node, nd.MapEntry):
                    # Find all dynamic map inputs
                    for e in state.in_edges(node):
                        if not e.dst_conn.startswith('IN_'):
                            if e.dst_conn in dynamic_map_inputs:
                                new_name = dt.find_new_name(e.dst_conn, dynamic_map_inputs)
                                dynamic_map_inputs.add(new_name)
                                repl_dict[e.dst_conn] = new_name
                                e._dst_conn = new_name
                            else:
                                dynamic_map_inputs.add(e.dst_conn)
                    if repl_dict:
                        in_connectors = {
                            repl_dict[n] if n in repl_dict else n: t
                            for n, t in node.in_connectors.items()
                        }
                        node.in_connectors = in_connectors
                        node.map.range.replace(repl_dict)
                        state.scope_subgraph(node).replace_dict(repl_dict)
                        propagation.propagate_memlets_scope(sd, state, state.scope_tree()[node])


def get_thread_local_data(sdfg: SDFG) -> List[str]:
    """ Returns a list of all data that are thread-local in the SDFG.

    This method DOES NOT apply recursively to nested SDFGs. It is also does not take into account outer Maps.

    :param sdfg: The SDFG to check.
    :return: A list of the names of all data that are thread-local in the SDFG.
    """
    # NOTE: We could exclude non-transient data here, but it is interesting to see if we find any non-transient data
    # only inside a Map.
    data_to_check = {name: None for name in sdfg.arrays.keys()}
    for state in sdfg.nodes():
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nd.AccessNode):
                # If the data was already removed from the candidated, continue
                if node.data not in data_to_check:
                    continue
                # If the data is not in a scope, i.e., cannot be thread-local, remove it from the candidates
                if scope_dict[node] is None:
                    del data_to_check[node.data]
                    continue
                # If the data is in a Map ...
                if isinstance(scope_dict[node], nd.MapEntry):
                    # ... if we haven't seen the data yet, note down the scope
                    if data_to_check[node.data] is None:
                        data_to_check[node.data] = scope_dict[node]
                    # ... if we have seen the data before, but in a different scope, remove it from the candidates
                    elif data_to_check[node.data] != scope_dict[node]:
                        del data_to_check[node.data]

    result = list(data_to_check.keys())
    for name in result:
        if not sdfg.arrays[name].transient:
            warnings.warn(f'Found thread-local data "{name}" that is not transient.')
    return result


def get_global_memlet_path_src(sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge) -> nd.Node:
    """
    Finds the global source node of an edge/memlet path, crossing nested SDFG scopes.

    :param sdfg: The SDFG containing the edge.
    :param state: The state containing the edge.
    :param edge: The edge to find the global source node for.
    :return: The global source node of the edge.
    """
    src = state.memlet_path(edge)[0].src
    if isinstance(src, nd.AccessNode) and not sdfg.arrays[src.data].transient and sdfg.parent is not None:
        psdfg = sdfg.parent_sdfg
        pstate = sdfg.parent
        pnode = sdfg.parent_nsdfg_node
        pedges = list(pstate.in_edges_by_connector(pnode, src.data))
        if len(pedges) > 0:
            pedge = pedges[0]
            return get_global_memlet_path_src(psdfg, pstate, pedge)
        else:
            pedges = list(pstate.out_edges_by_connector(pnode, src.data))
            if len(pedges) > 0:
                pedge = pedges[0]
                return get_global_memlet_path_dst(psdfg, pstate, pedge)
    return src


def get_global_memlet_path_dst(sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge) -> nd.Node:
    """
    Finds the global destination node of an edge/memlet path, crossing nested SDFG scopes.

    :param sdfg: The SDFG containing the edge.
    :param state: The state containing the edge.
    :param edge: The edge to find the global destination node for.
    :return: The global destination node of the edge.
    """
    dst = state.memlet_path(edge)[-1].dst
    if isinstance(dst, nd.AccessNode) and not sdfg.arrays[dst.data].transient and sdfg.parent is not None:
        psdfg = sdfg.parent_sdfg
        pstate = sdfg.parent
        pnode = sdfg.parent_nsdfg_node
        pedges = list(pstate.out_edges_by_connector(pnode, dst.data))
        if len(pedges) > 0:
            pedge = pedges[0]
            return get_global_memlet_path_dst(psdfg, pstate, pedge)
    return dst


def get_control_flow_block_dominators(sdfg: SDFG,
                                      idom: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                      all_dom: Optional[Dict[ControlFlowBlock, Set[ControlFlowBlock]]] = None,
                                      ipostdom: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                      all_postdom: Optional[Dict[ControlFlowBlock, Set[ControlFlowBlock]]] = None):
    """
    Find the dominator and postdominator relationship between control flow blocks of an SDFG.
    This transitively computes the domination relationship across control flow regions, as if the SDFG were to be
    inlined entirely.

    :param idom: A dictionary in which to store immediate dominator relationships. Not computed if None.
    :param all_dom: A dictionary in which to store all dominator relationships. Not computed if None.
    :param ipostdom: A dictionary in which to store immediate postdominator relationships. Not computed if None.
    :param all_postdom: A dictionary in which to all postdominator relationships. Not computed if None.
    """
    # Avoid cyclic import
    from dace.sdfg.analysis import cfg as cfg_analysis

    if idom is not None or all_dom is not None:
        added_sinks: Dict[AbstractControlFlowRegion, SDFGState] = {}
        if idom is None:
            idom = {}
        for cfg in sdfg.all_control_flow_regions(parent_first=True):
            if isinstance(cfg, ConditionalBlock):
                for _, b in cfg.branches:
                    idom[b] = cfg
            else:
                sinks = cfg.sink_nodes()
                if len(sinks) > 1:
                    added_sinks[cfg] = cfg.add_state()
                    for s in sinks:
                        cfg.add_edge(s, added_sinks[cfg], InterstateEdge())
                idom.update(nx.immediate_dominators(cfg.nx, cfg.start_block))
        # Compute the transitive relationship of immediate dominators:
        # - For every start state in a control flow region, the immediate dominator is the immediate dominator of the
        #   parent control flow region.
        # - If the immediate dominator is a conditional or a loop, change the immediate dominator to be the immediate
        #   dominator of that loop or conditional.
        # - If the immediate dominator is any other control flow region, change the immediate dominator to be the
        #   immediate dominator of that region's end / exit - or a virtual one if no single one exists.
        for k, _ in idom.items():
            if k.parent_graph is not sdfg and k is k.parent_graph.start_block:
                next_dom = idom[k.parent_graph]
                while next_dom.parent_graph is not sdfg and next_dom is next_dom.parent_graph.start_block:
                    next_dom = idom[next_dom.parent_graph]
                idom[k] = next_dom
        changed = True
        while changed:
            changed = False
            for k, v in idom.items():
                if isinstance(v, AbstractControlFlowRegion):
                    if isinstance(v, (LoopRegion, ConditionalBlock)):
                        idom[k] = idom[v]
                    else:
                        if v in added_sinks:
                            idom[k] = idom[added_sinks[v]]
                        else:
                            idom[k] = v.sink_nodes()[0]
                    if idom[k] is not v:
                        changed = True

        for cf, v in added_sinks.items():
            cf.remove_node(v)

        if all_dom is not None:
            all_dom.update(cfg_analysis.all_dominators(sdfg, idom))

    if ipostdom is not None or all_postdom is not None:
        added_sinks: Dict[AbstractControlFlowRegion, SDFGState] = {}
        sinks_per_cfg: Dict[AbstractControlFlowRegion, ControlFlowBlock] = {}
        if ipostdom is None:
            ipostdom = {}

        for cfg in sdfg.all_control_flow_regions(parent_first=True):
            if isinstance(cfg, ConditionalBlock):
                sinks_per_cfg[cfg] = cfg
                for _, b in cfg.branches:
                    ipostdom[b] = cfg
            else:
                # Get immediate post-dominators
                sink_nodes = cfg.sink_nodes()
                if len(sink_nodes) > 1:
                    sink = cfg.add_state()
                    added_sinks[cfg] = sink
                    sinks_per_cfg[cfg] = sink
                    for snode in sink_nodes:
                        cfg.add_edge(snode, sink, dace.InterstateEdge())
                elif len(sink_nodes) == 0:
                    return None
                else:
                    sink = sink_nodes[0]
                    sinks_per_cfg[cfg] = sink
                ipostdom.update(nx.immediate_dominators(cfg._nx.reverse(), sink))

        # Compute the transitive relationship of immediate postdominators, similar to how it works for immediate
        # dominators, but inverse.
        for k, _ in ipostdom.items():
            if k.parent_graph is not sdfg and (k is sinks_per_cfg[k.parent_graph]
                                               or isinstance(k.parent_graph, ConditionalBlock)):
                next_pdom = ipostdom[k.parent_graph]
                while next_pdom.parent_graph is not sdfg and (next_pdom is sinks_per_cfg[next_pdom.parent_graph]
                                                              or isinstance(next_pdom.parent_graph, ConditionalBlock)):
                    next_pdom = ipostdom[next_pdom.parent_graph]
                ipostdom[k] = next_pdom
        changed = True
        while changed:
            changed = False
            for k, v in ipostdom.items():
                if isinstance(v, AbstractControlFlowRegion):
                    if isinstance(v, (LoopRegion, ConditionalBlock)):
                        ipostdom[k] = ipostdom[v]
                    else:
                        ipostdom[k] = v.start_block
                    if ipostdom[k] is not v:
                        changed = True

        for cf, v in added_sinks.items():
            cf.remove_node(v)

        if all_postdom is not None:
            all_postdom.update(cfg_analysis.all_dominators(sdfg, ipostdom))


def set_nested_sdfg_parent_references(sdfg: SDFG):
    """
    Sets the parent_sdfg attribute for all NestedSDFGs recursively.
    """
    sdfg.reset_cfg_list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, NestedSDFG):
                node.sdfg.parent_sdfg = sdfg
                set_nested_sdfg_parent_references(node.sdfg)


def get_used_data(scope: Union[ControlFlowRegion, SDFGState, nd.MapEntry, nd.NestedSDFG],
                  parent_state: Union[SDFGState, None] = None) -> Set[str]:
    """
    Returns a set of all data names that are used in the given control flow region, state, map entry or nested SDFG node.
    Data is considered used if there is an access node within the scope to data or it appears in an interstate edge.

    :param cfg: The control flow region, state, or map entry node to check.
    :param parent_state: The parent state of the scope, used only for MapEntry nodes. Can't be None if scope is a MapEntry.
    :return: A set of used data names.
    """
    if isinstance(scope, SDFGState) or isinstance(scope, ControlFlowRegion):
        read_data, write_data = scope.read_and_write_sets()
        return read_data.union(write_data)
    elif isinstance(scope, nd.NestedSDFG):
        read_data, write_data = scope.sdfg.read_and_write_sets()
        return read_data.union(write_data)
    elif isinstance(scope, nd.MapEntry):
        assert parent_state is not None, "parent_state must be provided for MapEntry nodes"
        state: SDFGState = parent_state
        # How can data be accessed in an SDFG?:
        # Read interstate edges or access nodes using memlets
        # Written to access nodes using memlets
        # For map inputs the data might be not directly coming through an access node,
        # need to check the edges too
        #
        # To get all used data, within a state iterate access nodes
        # If data is passed to a nested SDFG (even if it is only used on an interstate edge),
        # the access node must be present in the parent graph.
        used_data = set()

        # All data used in the NestedSDFGs need to be connected through access nodes
        for node in state.all_nodes_between(scope, state.exit_node(scope)):
            if isinstance(node, nd.AccessNode):
                used_data.add(node.data)
        # Need to consider map inputs and outputs too
        for ie in state.in_edges(scope):
            if ie.data is not None and ie.data.data is not None:
                used_data.add(ie.data.data)
        for oe in state.out_edges(scope):
            if oe.data is not None and oe.data.data is not None:
                used_data.add(oe.data.data)

        return used_data
    else:
        raise Exception("Unsupported scope type for get_constant_data: {}".format(type(scope)))


def get_constant_data(scope: Union[ControlFlowRegion, SDFGState, nd.NestedSDFG, nd.MapEntry],
                      parent_state: Union[SDFGState, None] = None) -> Set[str]:
    """
    Returns a set of all constant data in the given control flow region, state, or with the map scope.
    Data is considered constant if there is any incoming edge to an access node of the data.
    Due to the semantics of SDFG, if a nested SDFG writes to the data container it needs to be
    visible in the parent graph as well, so the function does not need to be recursive.

    :param cfg: The control flow region, state or a map entry node to check.
    :param parent_state: The parent_state of the scope, used only for MapEntry nodes.
    :return: A set of constant data names.
    """

    def _incoming_memlet(state: SDFGState, node: nd.AccessNode) -> bool:
        return (state.in_degree(node) > 0 and any([e.data is not None for e in state.in_edges(node)]))

    if isinstance(scope, (SDFGState, ControlFlowRegion)):
        read_data, write_data = scope.read_and_write_sets()
        return read_data - write_data
    elif isinstance(scope, nd.NestedSDFG):
        read_data, write_data = scope.sdfg.read_and_write_sets()
        return read_data - write_data
    elif isinstance(scope, nd.MapEntry):
        state: SDFGState = parent_state

        # Which data are const:
        # All access nodes that have no incoming edges
        used_data = set()
        written_data = set()

        # All data used in the NestedSDFGs need to be connected through access nodes
        for node in state.all_nodes_between(scope, state.exit_node(scope)):
            if isinstance(node, nd.AccessNode):
                # Either no incoming edge, or no incoming edge has a Memlet (dependency edge only)
                if _incoming_memlet(state, node):
                    written_data.add(node.data)

        # Need to consider map inputs and outputs too
        for ie in state.in_edges(scope):
            if ie.data is not None and ie.data.data is not None:
                used_data.add(ie.data.data)
        for oe in state.out_edges(state.exit_node(scope)):
            if oe.data is not None and oe.data.data is not None:
                written_data.add(oe.data.data)
            used_data.add(oe.data.data)

        return used_data - written_data
    else:
        raise Exception("Unsupported scope type for get_constant_data: {}".format(type(scope)))


def get_used_symbols(
    scope: Union[SDFG, ControlFlowRegion, SDFGState, nd.MapEntry, nd.NestedSDFG],
    parent_state: Union[SDFGState, None] = None,
    include_symbols_for_offset_calculations: bool = False,
) -> Set[str]:
    """
    Returns a set of all used symbols, that have been defined by the scope or were already defined for the duration of the
    scope in the given control flow region, state, or with the map scope.

    :param cfg: The control flow region, state or a map entry node to check.
    :param parent_state: The parent graph of the scope, used only for MapEntry nodes.
    :return: A set of symbol names.
    """
    return _get_used_symbols_impl(scope=scope,
                                  constant_syms_only=False,
                                  parent_state=parent_state,
                                  include_symbols_for_offset_calculations=include_symbols_for_offset_calculations)


def get_constant_symbols(scope: Union[SDFG, ControlFlowRegion, SDFGState, nd.MapEntry, nd.NestedSDFG],
                         parent_state: Union[SDFGState, None] = None,
                         include_symbols_for_offset_calculations: bool = False) -> Set[str]:
    """
    Returns a set of all constant symbols in the given control flow region, state, or with the map scope,
    which have been defined by the scope (e.g. map) or defined for the duration of the scope.
    A symbol is considered constant if no interstate edge within the scope writes to it.

    :param cfg: The control flow region, state or a map entry node to check.
    :param parent_state: The parent graph of the scope, used only for MapEntry nodes.
    :return: A set of constant symbol names.
    """
    return _get_used_symbols_impl(scope=scope,
                                  constant_syms_only=True,
                                  parent_state=parent_state,
                                  include_symbols_for_offset_calculations=include_symbols_for_offset_calculations)


def _get_used_symbols_impl(scope: Union[SDFG, ControlFlowRegion, SDFGState, nd.MapEntry,
                                        nd.NestedSDFG], constant_syms_only: bool, parent_state: Union[SDFGState, None],
                           include_symbols_for_offset_calculations: bool) -> Set[str]:
    """
    Returns a set of all constant symbols in the given control flow region, state, or with the map scope.
    A symbol is considered constant if no interstate edge writes to it.

    :param cfg: The control flow region, state or a map entry node to check.
    :param parent_state: The parent graph of the scope, used only for MapEntry nodes.
    :return: A set of constant symbol names.
    """

    def _get_assignments(cfg: Union[ControlFlowRegion, SDFG]) -> Set[str]:
        written_symbols = set()
        for edge in cfg.all_interstate_edges():
            if edge.data is not None:
                written_symbols = written_symbols.union(edge.data.assignments.keys())
        return written_symbols

    offset_symbols = set()
    if include_symbols_for_offset_calculations:
        used_data = get_used_data(scope=scope, parent_state=parent_state)
        for data in used_data:
            parent_graph = parent_state if isinstance(scope, nd.MapEntry) else scope
            if data in parent_graph.sdfg.arrays:
                desc = parent_graph.sdfg.arrays[data]
                offset_symbols.update(str(sym) for sym in desc.free_symbols)

    if isinstance(scope, SDFGState):
        symbols = scope.used_symbols(all_symbols=False)
        # Since no symbol can change within a state we are good to go
        return offset_symbols | symbols
    elif isinstance(scope, (SDFG, ControlFlowRegion)):
        # Need to get all used symbols within the SDFG or CFG
        used_symbols = scope.used_symbols(all_symbols=False)
        # Get all symbols that are written to
        written_symbols = _get_assignments(scope)
        if constant_syms_only:
            return (offset_symbols | used_symbols) - written_symbols
        else:
            return offset_symbols | used_symbols
    elif isinstance(scope, nd.NestedSDFG):
        used_symbols = scope.sdfg.used_symbols(all_symbols=False)
        # Can't pass them as const if they are written to in the nested SDFG
        written_symbols = _get_assignments(scope.sdfg)
        if constant_syms_only:
            return (offset_symbols | used_symbols) - written_symbols
        else:
            return offset_symbols | used_symbols
    elif isinstance(scope, nd.MapEntry):
        used_symbols = scope.used_symbols_within_scope(parent_state=parent_state)
        return offset_symbols | used_symbols
    else:
        raise Exception("Unsupported scope type for get_constant_data: {}".format(type(scope)))


def _specialize_scalar_impl(root: 'dace.SDFG', sdfg: 'dace.SDFG', scalar_name: str, scalar_val: Union[float, int, str]):
    # This function replaces a scalar with the name <scalar_name> with a constant
    # A scalar can appear on:
    # 1. Interstate Edge
    # -> For 1: Replace occurence on the interstate edge with scalar_name
    # 2. Dynamic Input to a Map
    # -> For 2: Rm. dynamic in connector, remove the edge and the node if the degree is None
    # 3. Access Node
    # -> If access node is used then e.g. [scalar] -> [tasklet]
    # -> then [tasklet(assign const value)] -> [access node] -> [tasklet]

    def repl_code_block_or_str(input: Union[CodeBlock, str], src: str, dst: str):
        if isinstance(input, CodeBlock):
            return CodeBlock(input.as_string.replace(src, dst))
        else:
            return input.replace(src, dst)

    # If we are the root SDFG then we need can't remove non-transient scalar (but will just not use it)
    # For nestedSDFGs we will remove
    if root != sdfg:
        if scalar_name in sdfg.arrays:
            sdfg.remove_data(scalar_name, validate=False)

        if scalar_name in sdfg.symbols:
            sdfg.remove_symbol(scalar_name)

    nsdfgs = set()
    c = 0
    for state in sdfg.all_states():
        # Check dynamic inputs
        for e in state.edges():
            if e not in state.edges():
                continue
            if e.data is None or e.data.data != scalar_name:
                continue

            # Now we know we have an edge where memlet.data is the scalar

            src = e.src
            dst = e.dst

            assert e.data.data == scalar_name

            if isinstance(e.dst, nd.Tasklet):
                assign_tasklet = state.add_tasklet(f"assign_{scalar_name}",
                                                   inputs={},
                                                   outputs={"_out"},
                                                   code=f"_out = {scalar_val}")
                tmp_name = f"__tmp_{scalar_name}_{c}"
                c += 1
                copydesc = copy.deepcopy(sdfg.arrays[scalar_name])
                copydesc.transient = True
                copydesc.storage = dace.StorageType.Register
                sdfg.add_datadesc(tmp_name, copydesc)
                scl_an = state.add_access(tmp_name)
                state.remove_edge(e)
                state.add_edge(assign_tasklet, "_out", scl_an, None, dace.memlet.Memlet.from_array(tmp_name, copydesc))
                state.add_edge(scl_an, None, dst, e.dst_conn, dace.memlet.Memlet.from_array(tmp_name, copydesc))
                if e.src_conn is not None:
                    src.remove_out_connector(e.src_conn)
            else:
                state.remove_edge(e)
                if e.src_conn is not None:
                    src.remove_out_connector(e.src_conn)
                if e.dst_conn is not None:
                    dst.remove_in_connector(e.dst_conn)

            if state.out_degree(src) == 0:
                if isinstance(src, nd.MapEntry):
                    # Add a dep edge, to not invalidate the map
                    state.add_edge(src, None, dst, None, dace.memlet.Memlet())
                else:
                    if state.degree(src) == 0:
                        state.remove_node(src)
            if state.in_degree(dst) == 0:
                if isinstance(dst, nd.MapExit):
                    state.add_edge(src, None, dst, None, dace.memlet.Memlet())
                else:
                    if state.degree(dst) == 0:
                        state.remove_node(dst)

        for node in state.nodes():
            if isinstance(node, nd.MapEntry):
                new_range_list = []

                for (b, e, s) in node.map.range:
                    _b = b.subs(scalar_name, scalar_val)
                    _e = e.subs(scalar_name, scalar_val)
                    _s = s.subs(scalar_name, scalar_val)
                    new_range_list.append((_b, _e, _s))
                node.map.range = dace.subsets.Range(new_range_list)
            elif isinstance(node, nd.NestedSDFG):
                nsdfgs.add(node.sdfg)

    # Replace on for CFGs as
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            cfg.loop_condition = repl_code_block_or_str(cfg.loop_condition, scalar_name, str(scalar_val))
            cfg.init_statement = repl_code_block_or_str(cfg.init_statement, scalar_name, str(scalar_val))
            cfg.update_statement = repl_code_block_or_str(cfg.update_statement, scalar_name, str(scalar_val))
            assert cfg.loop_variable != scalar_name, (
                f"Loop variable {cfg.loop_variable} cannot be the same as the scalar {scalar_name}")
        if isinstance(cfg, ConditionalBlock):
            for i, (n_cond, n_body) in enumerate(cfg.branches):
                if n_cond is not None:
                    cfg.branches[0] = (repl_code_block_or_str(n_cond, scalar_name, str(scalar_val)), n_body)

    for edge in sdfg.all_interstate_edges(recursive=True):
        edge.data.replace_dict({f"{scalar_name}": f"{scalar_val}"})

    if root != sdfg:
        if scalar_name in sdfg.parent_nsdfg_node.symbol_mapping:
            del sdfg.parent_nsdfg_node.symbol_mapping[scalar_name]

    for nsdfg in nsdfgs:
        _specialize_scalar_impl(root, nsdfg, scalar_name, scalar_val)


def specialize_scalar(sdfg: 'dace.SDFG', scalar_name: str, scalar_val: Union[float, int, str]):
    assert isinstance(scalar_name, str)
    assert isinstance(scalar_val, (float, int, str))
    _specialize_scalar_impl(sdfg, sdfg, scalar_name, scalar_val)


def in_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """
    Find the edge that connects to input connector `name` on `node`.

    :param node: the node.
    :param state: the state.
    :param name: the input connector name.
    :return: the edge that connects to connector `name`.
    """
    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """
    Find the edge that connects to output connector `name` on `node`.

    :param node: the node.
    :param state: the state.
    :param name: the output connector name.
    :return: the edge that connects to connector `name`.
    """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """
    Find the descriptor of the data that connects to input connector `name`.

    :param node: the node.
    :param state: the state.
    :param sdfg: the sdfg.
    :param name: the input connector name.
    :return: the descriptor of the data that connects to connector `name`.
    """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """
    Find the descriptor of the data that connects to output connector `name`.

    :param node: the node.
    :param state: the state.
    :param sdfg: the sdfg.
    :param name: the output connector name.
    :return: the descriptor of the data that connects to connector `name`.
    """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def expand_nodes(sdfg: SDFG, predicate: Callable[[nd.Node], bool]):
    """
    Recursively expand library nodes in the SDFG using a given predicate.

    :param sdfg: the sdfg to expand nodes on.
    :param predicate: a predicate that will be called to check if a node should be expanded.
    """
    if sdfg is None:
        return
    states = list(sdfg.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):
            if isinstance(node, nd.NestedSDFG):
                expand_nodes(node.sdfg, predicate=predicate)
            elif isinstance(node, nd.LibraryNode):
                if predicate(node):
                    impl_name = node.expand(state)
                    if config.Config.get_bool('debugprint'):
                        print("Automatically expanded library node \"{}\" with implementation \"{}\".".format(
                            str(node), impl_name))
                    expanded_something = True

        if expanded_something:
            states.append(state)
