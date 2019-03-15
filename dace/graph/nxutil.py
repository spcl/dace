from ast import Subscript
from collections import deque
import copy
import itertools
import re
import os
from typing import Callable, List, Union
from string import ascii_uppercase
import networkx as nx

import dace
from dace import sdfg, types, symbolic
from dace.config import Config
from dace.graph import nodes, graph as gr

params = List[dace.symbolic.symbol]
ranges = List[Union[dace.subsets.Range, dace.subsets.Indices]]


class CannotExpand(Exception):
    pass


def node_path_graph(*args):
    """ Generates a path graph passing through the input nodes.

        The function generates a graph using as nodes the input arguments.
        Subsequently, it creates a path passing through all the nodes, in
        the same order as they were given in the function input.

        @param *args: Variable number of nodes or a list of nodes.
        @return: A directed graph based on the input arguments.
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


def dfs_topological_sort(G, sources=None, parent=False, condition=None):
    """ Produce nodes in a depth-first topological ordering.

    The function produces nodes in a depth-first topological ordering
    (DFS to make sure maps are visited properly), with the condition
    that each node visited had all its predecessors visited. Applies
    for DAGs only.

    @param G: An input DiGraph (assumed acyclic).
    @param sources: (optional) node or list of nodes that
                    specify starting point(s) for depth-first search and return
                    edges in the component reachable from source.
    @return: A generator of edges in the lastvisit depth-first-search.

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


def dfs_conditional(G, source, condition, reversed=False):
    """ Traverse a graph (DFS) only through edges that match a condition. """
    if isinstance(source, list): nodes = source
    else: nodes = [source]

    def in_edges_reversed(graph):
        def _in_edges_reversed(node):
            for e in graph.in_edges(node):
                ecpy = copy.copy(e)
                ecpy.reverse()
                yield ecpy

        return _in_edges_reversed

    get_children = G.out_edges if not reversed else in_edges_reversed(G)

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, get_children(start).__iter__())]
        while stack:
            parent, children = stack[-1]
            try:
                e = next(children)
                if e.dst not in visited:
                    visited.add(e.dst)
                    if condition(e.src, e.dst, e.data):
                        yield e
                        stack.append((e.dst, get_children(e.dst).__iter__()))
            except StopIteration:
                stack.pop()


def bfs_conditional(G, source, condition):
    """ Traverse a graph (BFS) only through edges that match a condition. """

    visited = set([source])
    queue = deque([(source, G.out_edges(source).__iter__())])
    while queue:
        parent, children = queue[0]
        try:
            e = next(children)
            if e.dst not in visited:
                visited.add(e.dst)
                if condition(e.src, e.dst, e.data):
                    yield e
                    queue.append((e.dst, G.out_edges(child).__iter__()))
        except StopIteration:
            queue.popleft()


def traverse_sdfg_scope(G, source, yield_edges=True):
    """ Traverse an SDFG scope (nodes dominated by a ScopeEntry and 
        post-dominated by a ScopeExit). 
        @param G: Input graph (assumed SDFGState).
        @param source: Source node.
        @param yield_edges: If True, returned generator yields edges
                            as well as nodes.
        @return: A generator that iterates over the scope nodes (or edges).
    """

    if not isinstance(source, nodes.EntryNode):
        raise SyntaxError('Source should be an entry node')

    visited = set()
    visited.add(source)

    if yield_edges:
        for e in G.out_edges(source):
            yield tuple(e) + (1, )
    else:
        yield source, 1

    stack = [(1, source, iter(G.out_edges(source)))]
    while stack:
        scope, parent, children = stack[-1]
        try:
            e = next(children)
            child = e.dst
            if child not in visited:
                # Make sure that all predecessors have been visited
                skip = False
                for pred in G.predecessors(child):
                    if pred not in visited:
                        skip = True
                        break
                if skip:
                    continue

                if yield_edges:
                    if not (isinstance(child, nodes.ExitNode) and scope == 1):
                        for e in G.out_edges(child):
                            yield tuple(e) + (scope, )
                else:
                    yield child, scope

                visited.add(child)
                if isinstance(child, nodes.EntryNode):
                    stack.append((scope + 1, child, iter(G.out_edges(child))))
                elif isinstance(child, nodes.ExitNode):
                    if scope > 1:  # Don't traverse beyond scope
                        stack.append((scope - 1, child, iter(
                            G.out_edges(child))))
                else:
                    stack.append((scope, child, iter(G.out_edges(child))))
        except StopIteration:
            stack.pop()


def gen_label(prefix=""):
    """ Generates a label as A,B,C,...,Z,AA,AB,... """
    indices = [0]
    while True:
        label = "".join([ascii_uppercase[i] for i in indices])
        yield prefix + label
        indices[0] += 1
        for pos, val in enumerate(indices):
            if val == len(ascii_uppercase):
                indices[pos] = 0
            if len(indices) == pos + 1:
                indices.append(1)
            else:
                indices[pos + 1] += 1


def indstr(x):
    try:
        return int(x)
    except TypeError:  # int() argument must be a string, a bytes-like object or a number, not [X]
        return str(x)


def range_to_str(ranges, limit_length=50):
    """ Converts one or multiple range tuples to a string. """

    try:
        len(ranges[0])
    except TypeError:
        ranges = [ranges]

    def convert_index(r):
        if len(r) == 3:
            if r[2] != 1:
                return "{}:{}:{}".format(
                    symbolic.symstr(r[0]), symbolic.symstr(r[1]),
                    symbolic.symstr(r[2]))
            else:
                return "{}:{}".format(
                    symbolic.symstr(r[0]), symbolic.symstr(r[1]))
        else:
            raise ValueError("Unsupported range: " + str(r))

    s = []
    for r in ranges:
        s.append(convert_index(r))

    res = ', '.join(s)
    if limit_length is not None:
        if not Config.get_bool('renderer', 'fulledges') and \
           len(res) > limit_length:
            res = '...'

    return "[" + res + "]"


def str_to_range(rangeStr):
    """ Converts a range string into a range tuple. """
    if rangeStr[0] != "[" or rangeStr[-1] != "]":
        raise ValueError("Invalid range " + rangeStr)
    rangeStr = re.sub("[\[\] ]", "", rangeStr)
    dimensions = rangeStr.split(",")
    ranges = [None] * len(dimensions)
    for i, r in enumerate(dimensions):
        entries = r.split(":")
        numArgs = len(entries)
        if numArgs < 2 or numArgs > 3:
            raise ValueError(
                "Range string should contain one or two separators (received "
                + str(r) + ")")
        iMin = None
        iMax = None
        step = None
        if entries[0]:
            iMin = entries[0]
        if entries[1]:
            iMax = entries[1]
        if numArgs == 3:
            if not entries[2]:
                raise ValueError("Stride for range cannot be empty")
            step = entries[2]
        ranges[i] = (iMin, iMax, step)
    return ranges


def make_list(val):
    """ If a scalar or string is passed make it a list, otherwise do nothing. 
    """
    try:
        len(val)
        if not isinstance(val, str):
            return val
    except TypeError:
        pass
    return [val]


def make_2d(ranges):
    """ If a 1D list is passed, make it 2D, otherwise do nothing. """
    if isinstance(ranges, Subscript):
        return [ranges]
    firstElem = ranges[0]
    try:
        if isinstance(firstElem, Subscript):
            return ranges
        len(firstElem)
        if not isinstance(firstElem, str):
            return ranges
    except TypeError:
        pass
    return [ranges]


def label_of(obj):
    """ Fetches the label of an object, or generates one if it doesn't exist. 
    """
    try:
        return obj.label
    except AttributeError:
        try:
            return obj.name
        except AttributeError:
            try:
                return next(type(obj)._nameGen)
            except AttributeError:
                type(obj)._nameGen = gen_label(type(obj).__name__ + " ")
                obj.label = next(type(obj)._nameGen)
                return obj.label


def fullrange(ndslice, var_size):
    """ Returns True iff the ND-slice represents the full array size. """
    for dim, (b, e, s) in zip(var_size, ndslice):
        if b != 0 or e != symbolic.pystr_to_symbolic(
                types.symbol_name_or_value(dim)) or s != 1:
            return False
    return True


def change_edge_dest(
        graph: dace.graph.graph.OrderedDiGraph,
        node_a: Union[dace.graph.nodes.Node,
                      dace.graph.graph.OrderedMultiDiConnectorGraph],
        node_b: Union[dace.graph.nodes.Node,
                      dace.graph.graph.OrderedMultiDiConnectorGraph]):
    """ Changes the destination of edges from node A to node B.

        The function finds all edges in the graph that have node A as their
        destination. It then creates a new edge for each one found,
        using the same source nodes and data, but node B as the destination.
        Afterwards, it deletes the edges found and inserts the new ones into 
        the graph.

        @param graph: The graph upon which the edge transformations will be 
                      applied.  
        @param node_a: The original destination of the edges.  
        @param node_b: The new destination of the edges to be transformed. 
    """

    # Create new incoming edges to node B, by copying the incoming edges to
    # node A and setting their destination to node B.
    edges = list(graph.in_edges(node_a))
    for e in edges:
        # Delete the incoming edges to node A from the graph.
        graph.remove_edge(e)
        # Insert the new edges to the graph.
        if isinstance(e, gr.MultiConnectorEdge):
            # dst_conn = e.dst_conn
            # if e.dst_conn is not None:
            #     # Remove connector from node A.
            #     node_a.remove_in_connector(e.dst_conn)
            #     # Insert connector to node B.
            #     if (not node_b.add_in_connector(dst_conn) and isinstance(
            #             node_b, (dace.graph.nodes.CodeNode,
            #                      dace.graph.nodes.MapEntry))):
            #         while not node_b.add_in_connector(dst_conn):
            #             dst_conn = dst_conn + '_'
            # graph.add_edge(e.src, e.src_conn, node_b, dst_conn, e.data)
            graph.add_edge(e.src, e.src_conn, node_b, e.dst_conn, e.data)
        else:
            graph.add_edge(e.src, node_b, e.data)


def change_edge_src(
        graph: dace.graph.graph.OrderedDiGraph,
        node_a: Union[dace.graph.nodes.Node,
                      dace.graph.graph.OrderedMultiDiConnectorGraph],
        node_b: Union[dace.graph.nodes.Node,
                      dace.graph.graph.OrderedMultiDiConnectorGraph]):
    """ Changes the sources of edges from node A to node B.

        The function finds all edges in the graph that have node A as their 
        source. It then creates a new edge for each one found, using the same 
        destination nodes and data, but node B as the source. Afterwards, it 
        deletes the edges
        found and inserts the new ones into the graph.

        @param graph: The graph upon which the edge transformations will be 
                      applied.
        @param node_a: The original source of the edges to be transformed.
        @param node_b: The new source of the edges to be transformed.
    """

    # Create new outgoing edges from node B, by copying the outgoing edges from
    # node A and setting their source to node B.
    edges = list(graph.out_edges(node_a))
    for e in edges:
        # Delete the outgoing edges from node A from the graph.
        graph.remove_edge(e)
        # Insert the new edges to the graph.
        if isinstance(e, gr.MultiConnectorEdge):
            # src_conn = e.src_conn
            # if e.src_conn is not None:
            #     # Remove connector from node A.
            #     node_a.remove_out_connector(e.src_conn)
            #     # Insert connector to node B.
            #     if (not node_b.add_out_connector(src_conn) and isinstance(
            #             node_b, (dace.graph.nodes.CodeNode,
            #                      dace.graph.nodes.MapExit))):
            #         while not node_b.add_out_connector(src_conn):
            #             src_conn = src_conn + '_'
            # graph.add_edge(node_b, src_conn, e.dst, e.dst_conn, e.data)
            graph.add_edge(node_b, e.src_conn, e.dst, e.dst_conn, e.data)
        else:
            graph.add_edge(node_b, e.dst, e.data)


def find_source_nodes(graph):
    """ Finds the source nodes of a graph.

        The function finds the source nodes of a graph, i.e. the nodes with 
        zero in-degree.

        @param graph: The graph whose source nodes are being searched for.
        @return: A list of the source nodes found.
    """
    return [n for n in graph.nodes() if graph.in_degree(n) == 0]


def find_sink_nodes(graph):
    """ Finds the sink nodes of a graph.

        The function finds the sink nodes of a graph, i.e. the nodes with zero out-degree.

        @param graph: The graph whose sink nodes are being searched for.
        @return: A list of the sink nodes found.
    """
    return [n for n in graph.nodes() if graph.out_degree(n) == 0]


def replace_subgraph(graph: dace.graph.graph.OrderedDiGraph,
                     old: dace.graph.graph.OrderedDiGraph,
                     new: dace.graph.graph.OrderedDiGraph):
    """ Replaces a subgraph of a graph with a new one. If replacement is not
        possible, it returns False.

        The function replaces the 'old' subgraph of the input graph with the 
        'new' subgraph. Both the 'old' and the 'new' subgraphs must have 
        unique source and sink nodes. Graph edges incoming to the source of 
        the 'old' subgraph have their destination changed to the source of 
        the 'new subgraph. Likewise, graph edges outgoing from the sink of 
        the 'old subgraph have their source changed to the sink of the 'new' 
        subgraph.

        @param graph: The graph upon which the replacement will be applied.
        @param old: The subgraph to be replaced.
        @param new: The replacement subgraph.

        @return: True if the replacement succeeded, otherwise False.
    """

    # 1. Find the source node of 'old' subgraph.
    # 1.1. Retrieve the source nodes of the 'old' subgraph.
    old_source_nodes = find_source_nodes(old)
    # 1.2. Verify the existence of a unique source in the 'old' subgraph.
    if len(old_source_nodes) != 1:
        return False
    old_source = old_source_nodes[0]

    # 2. Find the sink node of the 'old' subgraph.
    # 2.1. Retrieve the sink nodes of the 'old' subgraph.
    old_sink_nodes = find_sink_nodes(old)
    # 2.2. Verify the existence of a unique sink in the 'old' subgraph.
    if len(old_sink_nodes) != 1:
        return False
    old_sink = old_sink_nodes[0]

    # 3. Find the source node of 'new' subgraph.
    # 3.1. Retrieve the source nodes of the 'new' subgraph.
    new_source_nodes = find_source_nodes(new)
    # 3.2. Verify the existence of a unique source in the 'new' subgraph.
    if len(new_source_nodes) != 1:
        return False
    new_source = new_source_nodes[0]

    # 4. Find the sink node of the 'new' subgraph.
    # 4.1. Retrieve the sink nodes of the 'new' subgraph.
    new_sink_nodes = find_sink_nodes(new)
    # 4.2. Verify the existence of a unique sink in the 'new' subgraph.
    if len(new_sink_nodes) != 1:
        return False
    new_sink = new_sink_nodes[0]

    # 5. Add the 'new' subgraph to the graph.
    # 5.1. Add the nodes of the 'new' subgraph to the graph.
    graph.add_nodes_from(new.nodes())
    # 5.2. Add the edges of the 'new' subgraph to the graph.
    for e in new.edges():
        graph.add_edge(*e)

    # 6. Create new incoming edges to the source of the 'new' subgraph.
    change_edge_dest(graph, old_source, new_source)

    # 7. Create new outgoing edges from the sink of the 'new' subgraph.
    change_edge_src(graph, old_sink, new_sink)

    # 8. Remove all nodes of the 'old' subgraph from the graph.
    graph.remove_nodes_from(old.nodes())

    # 10. Subgraph replacement has succeeded. Return true.
    return True


def merge_maps(graph: dace.graph.graph.OrderedMultiDiConnectorGraph,
               outer_map_entry: dace.graph.nodes.MapEntry,
               outer_map_exit: dace.graph.nodes.MapExit,
               inner_map_entry: dace.graph.nodes.MapEntry,
               inner_map_exit: dace.graph.nodes.MapExit,
               param_merge: Callable[[params, params],
                                     params] = lambda p1, p2: p1 + p2,
               range_merge: Callable[[
                   ranges, ranges
               ], ranges] = lambda r1, r2: type(r1)(r1.ranges + r2.ranges)
               ) -> (dace.graph.nodes.MapEntry, dace.graph.nodes.MapExit):
    """ Merges two maps (their entries and exits). It is assumed that the
    operation is valid. """

    outer_map = outer_map_entry.map
    inner_map = inner_map_entry.map

    # Create merged map by inheriting attributes from outer map and using
    # the merge functions for parameters and ranges.
    merged_map = dace.graph.nodes.Map(
        label='_merged_' + outer_map.label + '_' + inner_map.label,
        params=param_merge(outer_map.params, inner_map.params),
        ndrange=range_merge(outer_map.range, inner_map.range),
        schedule=outer_map.schedule,
        unroll=outer_map.unroll,
        is_async=outer_map.is_async,
        flatten=outer_map.flatten,
        debuginfo=outer_map.debuginfo)

    merged_entry = dace.graph.nodes.MapEntry(merged_map)
    merged_entry.in_connectors = outer_map_entry.in_connectors
    merged_entry.out_connectors = outer_map_entry.out_connectors

    merged_exit = dace.graph.nodes.MapExit(merged_map)
    merged_exit.in_connectors = outer_map_exit.in_connectors
    merged_exit.out_connectors = outer_map_exit.out_connectors

    graph.add_nodes_from([merged_entry, merged_exit])

    # Redirect inner in edges.
    inner_in_edges = graph.out_edges(inner_map_entry)
    for edge in graph.edges_between(outer_map_entry, inner_map_entry):
        in_conn_num = edge.dst_conn[3:]
        out_conn = 'OUT_' + in_conn_num
        inner_edge = [e for e in inner_in_edges if e.src_conn == out_conn][0]
        graph.remove_edge(edge)
        graph.remove_edge(inner_edge)
        graph.add_edge(merged_entry, edge.src_conn, inner_edge.dst,
                       inner_edge.dst_conn, inner_edge.data)

    # Redirect inner out edges.
    inner_out_edges = graph.in_edges(inner_map_exit)
    for edge in graph.edges_between(inner_map_exit, outer_map_exit):
        out_conn_num = edge.src_conn[4:]
        in_conn = 'IN_' + out_conn_num
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
