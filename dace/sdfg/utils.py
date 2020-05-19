""" Various utility functions to create, traverse, and modify SDFGs. """

import collections
import copy
from dace.sdfg.sdfg import SDFG, SDFGState
from dace.graph import nodes as nd
from dace import subsets as sbs
from typing import Union

from typing import Callable, List, Union
from string import ascii_uppercase

from dace.graph import graph as gr


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


def dfs_topological_sort(G, sources=None, condition=None):
    """ Produce nodes in a depth-first topological ordering.

    The function produces nodes in a depth-first topological ordering
    (DFS to make sure maps are visited properly), with the condition
    that each node visited had all its predecessors visited. Applies
    for DAGs only, but works on any directed graph.

    :param G: An input DiGraph (assumed acyclic).
    :param sources: (optional) node or list of nodes that
                    specify starting point(s) for depth-first search and return
                    edges in the component reachable from source.
    :return: A generator of edges in the lastvisit depth-first-search.

    @note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
    by D. Eppstein, July 2004.

    @note: If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    """
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
        yield start
        visited.add(start)
        stack = [(start, iter(G.neighbors(start)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # Make sure that all predecessors have been visited
                    skip = False
                    for pred in G.predecessors(child):
                        if pred not in visited:
                            skip = True
                            break
                    if skip:
                        continue

                    visited.add(child)
                    if condition is None or condition(parent, child):
                        yield child
                        stack.append((child, iter(G.neighbors(child))))
            except StopIteration:
                stack.pop()


def change_edge_dest(graph: gr.OrderedDiGraph,
                     node_a: Union[nd.Node, gr.OrderedMultiDiConnectorGraph],
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


def change_edge_src(graph: gr.OrderedDiGraph,
                    node_a: Union[nd.Node, gr.OrderedMultiDiConnectorGraph],
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


def find_source_nodes(graph):
    """ Finds the source nodes of a graph.

        The function finds the source nodes of a graph, i.e. the nodes with 
        zero in-degree.

        :param graph: The graph whose source nodes are being searched for.
        :return: A list of the source nodes found.
    """
    return [n for n in graph.nodes() if graph.in_degree(n) == 0]


def find_sink_nodes(graph):
    """ Finds the sink nodes of a graph.

        The function finds the sink nodes of a graph, i.e. the nodes with zero out-degree.

        :param graph: The graph whose sink nodes are being searched for.
        :return: A list of the sink nodes found.
    """
    return [n for n in graph.nodes() if graph.out_degree(n) == 0]


ParamsType = List['dace.symbolic.symbol']
RangesType = List[sbs.Subset]


def merge_maps(
    graph: gr.OrderedMultiDiConnectorGraph,
    outer_map_entry: nd.MapEntry,
    outer_map_exit: nd.MapExit,
    inner_map_entry: nd.MapEntry,
    inner_map_exit: nd.MapExit,
    param_merge: Callable[[ParamsType, ParamsType],
                          ParamsType] = lambda p1, p2: p1 + p2,
    range_merge: Callable[[RangesType, RangesType],
                          RangesType] = lambda r1, r2: type(r1)
    (r1.ranges + r2.ranges)
) -> (nd.MapEntry, nd.MapExit):
    """ Merges two maps (their entries and exits). It is assumed that the
    operation is valid. """

    outer_map = outer_map_entry.map
    inner_map = inner_map_entry.map

    # Create merged map by inheriting attributes from outer map and using
    # the merge functions for parameters and ranges.
    merged_map = copy.deepcopy(outer_map)
    merged_map.label = 'merged_' + outer_map.label
    merged_map.params = param_merge(outer_map.params, inner_map.params)
    merged_map.range = range_merge(outer_map.range, inner_map.range)

    merged_entry = nd.MapEntry(merged_map)
    merged_entry.in_connectors = outer_map_entry.in_connectors
    merged_entry.out_connectors = outer_map_entry.out_connectors

    merged_exit = nd.MapExit(merged_map)
    merged_exit.in_connectors = outer_map_exit.in_connectors
    merged_exit.out_connectors = outer_map_exit.out_connectors

    graph.add_nodes_from([merged_entry, merged_exit])

    # Redirect inner in edges.
    inner_in_edges = graph.out_edges(inner_map_entry)
    for edge in graph.edges_between(outer_map_entry, inner_map_entry):
        if edge.dst_conn is None:  # Empty memlets
            out_conn = None
        else:
            out_conn = 'OUT_' + edge.dst_conn[3:]
        inner_edge = [e for e in inner_in_edges if e.src_conn == out_conn][0]
        graph.remove_edge(edge)
        graph.remove_edge(inner_edge)
        graph.add_edge(merged_entry, edge.src_conn, inner_edge.dst,
                       inner_edge.dst_conn, inner_edge.data)

    # Redirect inner out edges.
    inner_out_edges = graph.in_edges(inner_map_exit)
    for edge in graph.edges_between(inner_map_exit, outer_map_exit):
        if edge.src_conn is None:  # Empty memlets
            in_conn = None
        else:
            in_conn = 'IN_' + edge.src_conn[4:]
        inner_edge = [e for e in inner_out_edges if e.dst_conn == in_conn][0]
        graph.remove_edge(edge)
        graph.remove_edge(inner_edge)
        graph.add_edge(inner_edge.src, inner_edge.src_conn, merged_exit,
                       edge.dst_conn, inner_edge.data)

    # Redirect outer edges.
    change_edge_dest(graph, outer_map_entry, merged_entry)
    change_edge_src(graph, outer_map_exit, merged_exit)

    # Clean-up
    graph.remove_nodes_from(
        [outer_map_entry, outer_map_exit, inner_map_entry, inner_map_exit])

    return merged_entry, merged_exit


def consolidate_edges_scope(
        state: SDFGState, scope_node: Union[nd.EntryNode, nd.ExitNode]) -> int:
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
        # Outer side of the scope - remove edge and union subsets
        target_conn = prefix + data_to_conn[e.data.data][len(oprefix):]
        conn_to_remove = prefix + conn[len(oprefix):]
        remove_outer_connector(conn_to_remove)
        out_edge = next(ed for ed in outer_edges(scope_node)
                        if ed.dst_conn == target_conn)
        edge_to_remove = next(ed for ed in outer_edges(scope_node)
                              if ed.dst_conn == conn_to_remove)
        out_edge.data.subset = sbs.union(out_edge.data.subset,
                                         edge_to_remove.data.subset)
        state.remove_edge(edge_to_remove)
        consolidated += 1
        # Inner side of the scope - remove and reconnect
        remove_inner_connector(e.src_conn)
        for e in edges_by_connector[conn]:
            e._src_conn = data_to_conn[e.data.data]

    return consolidated


def consolidate_edges(sdfg: SDFG) -> int:
    """
    Union scope-entering memlets relating to the same data node in all states.
    This effectively reduces the number of connectors and allows more
    transformations to be performed, at the cost of losing the individual
    per-tasklet memlets.
    :param sdfg: The SDFG to consolidate.
    :return: Number of edges removed.
    """
    consolidated = 0
    for state in sdfg.nodes():
        # Start bottom-up
        queue = state.scope_leaves()
        next_queue = []
        while len(queue) > 0:
            for scope in queue:
                consolidated += consolidate_edges_scope(state, scope.entry)
                consolidated += consolidate_edges_scope(state, scope.exit)
                if scope.parent is not None:
                    next_queue.append(scope.parent)
            queue = next_queue
            next_queue = []

    return consolidated
