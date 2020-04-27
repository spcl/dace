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
from dace import sdfg, dtypes, symbolic
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
    for DAGs only.

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


def traverse_sdfg_scope(G, source, yield_edges=True):
    """ Traverse an SDFG scope (nodes dominated by a ScopeEntry and 
        post-dominated by a ScopeExit). 
        :param G: Input graph (assumed SDFGState).
        :param source: Source node.
        :param yield_edges: If True, returned generator yields edges
                            as well as nodes.
        :return: A generator that iterates over the scope nodes (or edges).
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
                        stack.append(
                            (scope - 1, child, iter(G.out_edges(child))))
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


def range_to_str(ranges, limit_length=50):
    """ Converts one or multiple range tuples to a string. """

    try:
        len(ranges[0])
    except TypeError:
        ranges = [ranges]

    def convert_index(r):
        if len(r) == 3:
            if r[2] != 1:
                return "{}:{}:{}".format(symbolic.symstr(r[0]),
                                         symbolic.symstr(r[1]),
                                         symbolic.symstr(r[2]))
            else:
                return "{}:{}".format(symbolic.symstr(r[0]),
                                      symbolic.symstr(r[1]))
        else:
            raise ValueError("Unsupported range: " + str(r))

    s = []
    for r in ranges:
        s.append(convert_index(r))

    res = ', '.join(s)

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


def change_edge_dest(graph: dace.graph.graph.OrderedDiGraph,
                     node_a: Union[dace.graph.nodes.Node, dace.graph.graph.
                                   OrderedMultiDiConnectorGraph],
                     node_b: Union[dace.graph.nodes.Node, dace.graph.graph.
                                   OrderedMultiDiConnectorGraph]):
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


def change_edge_src(graph: dace.graph.graph.OrderedDiGraph,
                    node_a: Union[dace.graph.nodes.Node, dace.graph.graph.
                                  OrderedMultiDiConnectorGraph],
                    node_b: Union[dace.graph.nodes.Node, dace.graph.graph.
                                  OrderedMultiDiConnectorGraph]):
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


def merge_maps(graph: dace.graph.graph.OrderedMultiDiConnectorGraph,
               outer_map_entry: dace.graph.nodes.MapEntry,
               outer_map_exit: dace.graph.nodes.MapExit,
               inner_map_entry: dace.graph.nodes.MapEntry,
               inner_map_exit: dace.graph.nodes.MapExit,
               param_merge: Callable[[params, params], params] = lambda p1, p2:
               p1 + p2,
               range_merge: Callable[[ranges, ranges], ranges] = lambda r1, r2:
               type(r1)(r1.ranges + r2.ranges)
               ) -> (dace.graph.nodes.MapEntry, dace.graph.nodes.MapExit):
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
