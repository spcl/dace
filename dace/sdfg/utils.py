# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
from dace.sdfg.sdfg import SDFG
from dace.sdfg.nodes import Node, NestedSDFG
from dace.sdfg.state import SDFGState, StateSubgraphView
from dace.sdfg.scope import ScopeSubgraphView
from dace.sdfg import nodes as nd, graph as gr
from dace import config, data as dt, dtypes, memlet as mm, subsets as sbs, symbolic
from string import ascii_uppercase
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union


def node_path_graph(*args):
    """ Generates a path graph passing through the input nodes.

        The function generates a graph using as nodes the input arguments.
        Subsequently, it creates a path passing through all the nodes, in
        the same order as they were given in the function input.

        :param *args: Variable number of nodes or a list of nodes.
        :return: A directed graph based on the input arguments.
        @rtype: gr.OrderedDiGraph
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
    """ Return best node and its value using a limited-depth Search (depth-
        limited DFS). """
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
    """ Produce nodes in a depth-first topological ordering.

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

    :note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    by D. Eppstein, July 2004.

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
    

    :param G: An input DiGraph (assumed acyclic).
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

    :note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    by D. Eppstein, July 2004.

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


def nodes_in_all_simple_paths(G, source, target, condition: Callable[[Any], bool] = None) -> Set[Any]:
    """
    Returns a set of nodes that appear in any of the paths from ``source``
    to ``targets``. Optionally, a condition can be given to control traversal.
    
    :param G: The graph to traverse.
    :param source: Source node.
    :param targets: 

    Notes
    -----
    This algorithm uses a modified depth-first search, adapted from 
    networkx.all_simple_paths.
    
    The algorithm is written for directed _graphs_. For multigraphs, use
    networkx.all_simple_paths!

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.
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
) -> (nd.MapEntry, nd.MapExit):
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
        remove_outer_connector = scope_node.remove_in_connector
        remove_inner_connector = scope_node.remove_out_connector
        prefix, oprefix = 'IN_', 'OUT_'
    else:
        outer_edges = state.out_edges
        inner_edges = state.in_edges
        remove_outer_connector = scope_node.remove_out_connector
        remove_inner_connector = scope_node.remove_in_connector
        prefix, oprefix = 'OUT_', 'IN_'

    edges_by_connector = collections.defaultdict(list)
    connectors_to_remove = set()
    for e in inner_edges(scope_node):
        edges_by_connector[e.src_conn].append(e)
        if e.data.data not in data_to_conn:
            data_to_conn[e.data.data] = e.src_conn
        elif data_to_conn[e.data.data] != e.src_conn:  # Need to consolidate
            connectors_to_remove.add(e.src_conn)

    for conn in connectors_to_remove:
        e = edges_by_connector[conn][0]
        offset = 3 if conn.startswith('IN_') else (4 if conn.startswith('OUT_') else len(oprefix))
        # Outer side of the scope - remove edge and union subsets
        target_conn = prefix + data_to_conn[e.data.data][offset:]
        conn_to_remove = prefix + conn[offset:]
        remove_outer_connector(conn_to_remove)
        if isinstance(scope_node, nd.EntryNode):
            out_edge = next(ed for ed in outer_edges(scope_node) if ed.dst_conn == target_conn)
            edge_to_remove = next(ed for ed in outer_edges(scope_node) if ed.dst_conn == conn_to_remove)
        else:
            out_edge = next(ed for ed in outer_edges(scope_node) if ed.src_conn == target_conn)
            edge_to_remove = next(ed for ed in outer_edges(scope_node) if ed.src_conn == conn_to_remove)
        out_edge.data.subset = sbs.union(out_edge.data.subset, edge_to_remove.data.subset)

        # Check if dangling connectors have been created and remove them,
        # as well as their parent edges
        remove_edge_and_dangling_path(state, edge_to_remove)

        consolidated += 1
        # Inner side of the scope - remove and reconnect
        if isinstance(scope_node, nd.EntryNode):
            remove_inner_connector(e.src_conn)
            for e in edges_by_connector[conn]:
                e._src_conn = data_to_conn[e.data.data]
        else:
            remove_inner_connector(e.dst_conn)
            for e in edges_by_connector[conn]:
                e._dst_conn = data_to_conn[e.data.data]

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
            neighbors = [neighbor for neighbor in state.out_edges(e.src) if e.src_conn == neighbor.src_conn]
        else:
            neighbors = [neighbor for neighbor in state.in_edges(e.dst) if e.dst_conn == neighbor.dst_conn]
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
                e.src.remove_out_connector('OUT' + e.dst_conn[2:])

        # Continue traversing upwards
        curedge = curedge.parent
    else:
        # Check if an isolated node have been created at the root and remove
        root_edge = mtree.root().edge
        root_node: nd.Node = root_edge.src if inwards else root_edge.dst
        if state.degree(root_node) == 0:
            state.remove_node(root_node)


def consolidate_edges(sdfg: SDFG, starting_scope=None) -> int:
    """
    Union scope-entering memlets relating to the same data node in all states.
    This effectively reduces the number of connectors and allows more
    transformations to be performed, at the cost of losing the individual
    per-tasklet memlets.
    :param sdfg: The SDFG to consolidate.
    :return: Number of edges removed.
    """
    from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlets_scope

    consolidated = 0
    for state in sdfg.nodes():
        # Start bottom-up
        if starting_scope and starting_scope.entry not in state.nodes():
            continue

        queue = [starting_scope] if starting_scope else state.scope_leaves()
        next_queue = []
        while len(queue) > 0:
            for scope in queue:
                consolidated += consolidate_edges_scope(state, scope.entry)
                consolidated += consolidate_edges_scope(state, scope.exit)
                if scope.parent is not None:
                    next_queue.append(scope.parent)
            queue = next_queue
            next_queue = []

        if starting_scope is not None:
            # Repropagate memlets from this scope outwards
            propagate_memlets_scope(sdfg, state, starting_scope)

            # No need to traverse other states
            break

    # Repropagate memlets
    if starting_scope is None:
        propagate_memlets_sdfg(sdfg)

    return consolidated


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
    out_edges = state.out_edges(view)

    # Invalid case: No data to view
    if len(in_edges) == 0 or len(out_edges) == 0:
        return None

    # If there is one edge (in/out) that leads (via memlet path) to an access
    # node, and the other side (out/in) has a different number of edges.
    if len(in_edges) == 1 and len(out_edges) != 1:
        return in_edges[0]
    if len(out_edges) == 1 and len(in_edges) != 1:
        return out_edges[0]
    if len(out_edges) == len(in_edges) and len(out_edges) != 1:
        return None

    in_edge = in_edges[0]
    out_edge = out_edges[0]

    # If there is one incoming and one outgoing edge, and one leads to a code
    # node, the one that leads to an access node is the viewed data.
    inmpath = state.memlet_path(in_edge)
    outmpath = state.memlet_path(out_edge)
    src_is_data, dst_is_data = False, False
    if isinstance(inmpath[0].src, nd.AccessNode):
        src_is_data = True
    if isinstance(outmpath[-1].dst, nd.AccessNode):
        dst_is_data = True

    if src_is_data and not dst_is_data:
        return in_edge
    if not src_is_data and dst_is_data:
        return out_edge
    if not src_is_data and not dst_is_data:
        return None

    # If both sides lead to access nodes, if one memlet's data points to the
    # view it cannot point to the viewed node.
    if in_edge.data.data == view.data and out_edge.data.data != view.data:
        return out_edge
    if in_edge.data.data != view.data and out_edge.data.data == view.data:
        return in_edge
    if in_edge.data.data == view.data and out_edge.data.data == view.data:
        return None

    # Check if there is a 'views' connector
    if in_edge.dst_conn and in_edge.dst_conn == 'views':
        return in_edge
    if out_edge.src_conn and out_edge.src_conn == 'views':
        return out_edge

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
    from dace.transformation.interstate import StateFusion  # Avoid import loop

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
        for sd in sdfg.all_sdfgs_recursive():
            fusible_states += sd.number_of_edges()

    if progress is True:
        pbar = tqdm(total=fusible_states, desc='Fusing states')

    start = time.time()

    for sd in sdfg.all_sdfgs_recursive():
        id = sd.sdfg_id

        while True:
            edges = list(sd.nx.edges)
            applied = 0
            skip_nodes = set()
            for u, v in edges:
                if (progress is None and tqdm is not None and (time.time() - start) > 5):
                    progress = True
                    pbar = tqdm(total=fusible_states, desc='Fusing states', initial=counter)

                if u in skip_nodes or v in skip_nodes:
                    continue
                candidate = {StateFusion.first_state: u, StateFusion.second_state: v}
                sf = StateFusion()
                sf.setup_match(sd, id, -1, candidate, 0, override=True)
                if sf.can_be_applied(sd, 0, sd, permissive=permissive):
                    sf.apply(sd, sd)
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

    if progress is None and not config.Config.get_bool('progress'):
        progress = False

    if progress is True or progress is None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

    counter = 0
    nsdfgs = [(n, p) for n, p in sdfg.all_nodes_recursive() if isinstance(n, NestedSDFG)]
    if progress is True:
        pbar = tqdm(total=len(nsdfgs), desc='Inlining SDFGs')

    start = time.time()

    for ctr, (node, state) in enumerate(reversed(nsdfgs)):
        id = node.sdfg.sdfg_id
        sd = state.parent
        if (progress is None and tqdm is not None and (time.time() - start) > 5):
            progress = True
            pbar = tqdm(total=len(nsdfgs), desc='Inlining SDFG', initial=ctr)

        # We have to reevaluate every time due to changing IDs
        state_id = sd.node_id(state)
        if multistate:
            candidate = {
                InlineMultistateSDFG.nested_sdfg: node,
            }
            inliner = InlineMultistateSDFG()
            inliner.setup_match(sd, id, state_id, candidate, 0, override=True)
            if inliner.can_be_applied(state, 0, sd, permissive=permissive):
                inliner.apply(state, sd)
                counter += 1
                if progress:
                    pbar.update(1)
                continue

        candidate = {
            InlineSDFG.nested_sdfg: node,
        }
        inliner = InlineSDFG()
        inliner.setup_match(sd, id, state_id, candidate, 0, override=True)
        if inliner.can_be_applied(state, 0, sd, permissive=permissive):
            inliner.apply(state, sd)
            counter += 1
        if progress:
            pbar.update(1)

    if progress:
        pbar.close()
    return counter


def load_precompiled_sdfg(folder: str):
    """
    Loads a pre-compiled SDFG from an output folder (e.g. ".dacecache/program").
    Folder must contain a file called "program.sdfg" and a subfolder called
    "build" with the shared object.

    :param folder: Path to SDFG output folder.
    :return: A callable CompiledSDFG object.
    """
    sdfg = SDFG.from_file(os.path.join(folder, 'program.sdfg'))
    suffix = config.Config.get('compiler', 'library_extension')
    return csdfg.CompiledSDFG(sdfg,
                              csdfg.ReloadableDLL(os.path.join(folder, 'build', f'lib{sdfg.name}.{suffix}'), sdfg.name))


def distributed_compile(sdfg: SDFG, comm: "Intracomm") -> csdfg.CompiledSDFG:
    """
    Compiles an SDFG in rank 0 of MPI communicator `comm`. Then, the compiled SDFG is loaded in all other ranks.
    NOTE: This method can be used only if the module mpi4py is installed.
    :param sdfg: SDFG to be compiled.
    :param comm: MPI communicator. "Intracomm" is the base mpi4py communicator class.
    :return: Compiled SDFG.
    """

    rank = comm.Get_rank()
    func = None
    folder = None

    # Rank 0 compiles SDFG.
    if rank == 0:
        func = sdfg.compile()
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
    for succ in sdfg.successors(state):
        result |= set(dfs_conditional(sdfg, sources=[succ], condition=lambda parent, _: parent.is_empty()))

    # Filter out empty states
    result = {s for s in result if not s.is_empty()}

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
    return str(sdfg.sdfg_id) + "_" + str(sdfg.node_id(state)) + "_" + str(state.node_id(node))


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
    if isinstance(desc, dt.View):
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


def traverse_sdfg_with_defined_symbols(
        sdfg: SDFG,
        recursive: bool = False) -> Generator[Tuple[SDFGState, Node, Dict[str, dtypes.typeclass]], None, None]:
    """
    Traverses the SDFG, its states and nodes, yielding the defined symbols and their types at each node.
    :return: A generator that yields tuples of (state, node in state, currently-defined symbols)
    """
    # Start with global symbols
    symbols = copy.copy(sdfg.symbols)
    symbols.update({k: dt.create_datadescriptor(v).dtype for k, v in sdfg.constants.items()})
    for desc in sdfg.arrays.values():
        symbols.update({str(s): s.dtype for s in desc.free_symbols})

    # Add symbols from inter-state edges along the state machine
    start_state = sdfg.start_state
    visited = set()
    visited_edges = set()
    for edge in sdfg.dfs_edges(start_state):
        # Source -> inter-state definition -> Destination
        visited_edges.add(edge)
        # Source
        if edge.src not in visited:
            visited.add(edge.src)
            yield from _tswds_state(sdfg, edge.src, symbols, recursive)

        # Add edge symbols into defined symbols
        issyms = edge.data.new_symbols(sdfg, symbols)
        symbols.update({k: v for k, v in issyms.items() if v is not None})

        # Destination
        if edge.dst not in visited:
            visited.add(edge.dst)
            yield from _tswds_state(sdfg, edge.dst, symbols, recursive)

    # If there is only one state, the DFS will miss it
    if start_state not in visited:
        yield from _tswds_state(sdfg, start_state, symbols, recursive)


def is_fpga_kernel(sdfg, state):
    """
    Returns whether the given state is an FPGA kernel and should be dispatched
    to the FPGA code generator.
    :return: True if this is an FPGA kernel, False otherwise.
    """
    if ("is_FPGA_kernel" in state.location and state.location["is_FPGA_kernel"] == False):
        return False
    data_nodes = state.data_nodes()
    if len(data_nodes) == 0:
        return False
    for n in data_nodes:
        if n.desc(sdfg).storage not in (dtypes.StorageType.FPGA_Global, dtypes.StorageType.FPGA_Local,
                                        dtypes.StorageType.FPGA_Registers, dtypes.StorageType.FPGA_ShiftRegister):
            return False
    return True


def postdominators(
    sdfg: SDFG,
    return_alldoms: bool = False
) -> Union[Dict[SDFGState, SDFGState], Tuple[Dict[SDFGState, SDFGState], Dict[SDFGState, Set[SDFGState]]]]:
    """
    Return the immediate postdominators of an SDFG. This may require creating new nodes and removing them, which
    happens in-place on the SDFG.

    :param sdfg: The SDFG to generate the postdominators from.
    :param return_alldoms: If True, returns the "all postdominators" dictionary as well.
    :return: Immediate postdominators, or a 2-tuple of (ipostdom, allpostdoms) if ``return_alldoms`` is True.
    """
    from dace.sdfg.analysis import cfg

    # Get immediate post-dominators
    sink_nodes = sdfg.sink_nodes()
    if len(sink_nodes) > 1:
        sink = sdfg.add_state()
        for snode in sink_nodes:
            sdfg.add_edge(snode, sink, dace.InterstateEdge())
    else:
        sink = sink_nodes[0]
    ipostdom: Dict[SDFGState, SDFGState] = nx.immediate_dominators(sdfg._nx.reverse(), sink)

    if return_alldoms:
        allpostdoms = cfg.all_dominators(sdfg, ipostdom)
        retval = (ipostdom, allpostdoms)
    else:
        retval = ipostdom

    # If a new sink was added for post-dominator computation, remove it
    if len(sink_nodes) > 1:
        sdfg.remove_node(sink)

    return retval
