# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Various utility functions to create, traverse, and modify SDFGs. """

import collections
import copy
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import nodes as nd, graph as gr
from dace import data as dt, dtypes, subsets as sbs
from string import ascii_uppercase
from typing import Callable, List, Optional, Union


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


def dfs_conditional(G, sources=None, condition=None):
    """ Produce nodes in a depth-first ordering.

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
    graph: SDFGState,
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
    for edge in graph.out_edges(inner_map_entry):
        if edge.src_conn is None:  # Empty memlets
            graph.add_edge(merged_entry, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)
            continue

        # Get memlet path and edge
        path = graph.memlet_path(edge)
        ind = path.index(edge)
        # Add an edge directly from the previous source connector to the
        # destination
        graph.add_edge(merged_entry, path[ind - 1].src_conn, edge.dst,
                       edge.dst_conn, edge.data)

    # Redirect inner out edges.
    for edge in graph.in_edges(inner_map_exit):
        if edge.dst_conn is None:  # Empty memlets
            graph.add_edge(edge.src, edge.src_conn, merged_exit, edge.dst_conn,
                           edge.data)
            continue

        # Get memlet path and edge
        path = graph.memlet_path(edge)
        ind = path.index(edge)
        # Add an edge directly from the source to the next destination
        # connector
        graph.add_edge(edge.src, edge.src_conn, merged_exit,
                       path[ind + 1].dst_conn, edge.data)

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

        # Check if dangling connectors have been created and remove them,
        # as well as their parent edges
        remove_edge_and_dangling_path(state, edge_to_remove)

        consolidated += 1
        # Inner side of the scope - remove and reconnect
        remove_inner_connector(e.src_conn)
        for e in edges_by_connector[conn]:
            e._src_conn = data_to_conn[e.data.data]

    return consolidated


def remove_edge_and_dangling_path(state: SDFGState, edge: MultiConnectorEdge):
    """
    Removes an edge and all of its parent edges in a memlet path, cleaning
    dangling connectors and isolated nodes resulting from the removal.
    :param state: The state in which the edge exists.
    :param edge: The edge to remove.
    """
    mtree = state.memlet_tree(edge)
    inwards = (isinstance(edge.src, nd.EntryNode)
               or isinstance(edge.dst, nd.EntryNode))

    # Traverse tree upwards, removing edges and connectors as necessary
    curedge = mtree
    while curedge is not None:
        e = curedge.edge
        state.remove_edge(e)
        if inwards:
            neighbors = [
                neighbor for neighbor in state.out_edges(e.src)
                if e.src_conn == neighbor.src_conn
            ]
        else:
            neighbors = [
                neighbor for neighbor in state.in_edges(e.dst)
                if e.dst_conn == neighbor.dst_conn
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
        if isinstance(src_node, nd.AccessNode) and isinstance(
                src_node.desc(sdfg), dt.Array):
            source_paths.append(src_node)
    for e in dfg.out_edges(node):
        sink_node = dfg.memlet_path(e)[-1].dst

        # Append empty path to differentiate between a copy and an array-view
        if isinstance(sink_node, nd.CodeNode):
            all_sink_paths.append(None)
        # Append path to sink node
        if isinstance(sink_node, nd.AccessNode) and isinstance(
                sink_node.desc(sdfg), dt.Array):
            sink_paths.append(sink_node)

    all_sink_paths.extend(sink_paths)
    all_source_paths.extend(source_paths)

    # Special case: stream can be represented as a view of an array
    if ((len(all_source_paths) > 0 and len(sink_paths) == 1)
            or (len(all_sink_paths) > 0 and len(source_paths) == 1)):
        # TODO: What about a source path?
        arrnode = sink_paths[0]
        # Only works if the stream itself is not an array of streams
        if list(node.desc(sdfg).shape) == [1]:
            node.desc(sdfg).sink = arrnode.data  # For memlet generation
            arrnode.desc(
                sdfg).src = node.data  # TODO: Move src/sink to node, not array
            return True
    return False


def dynamic_map_inputs(state: SDFGState,
                       map_entry: nd.MapEntry) -> List[gr.MultiConnectorEdge]:
    """
    For a given map entry node, returns a list of dynamic-range input edges.
    :param state: The state in which the map entry node resides.
    :param map_entry: The given node.
    :return: A list of edges in state whose destination is map entry and denote
             dynamic-range input memlets.
    """
    return [
        e for e in state.in_edges(map_entry)
        if e.dst_conn and not e.dst_conn.startswith('IN_')
    ]


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
        nsdfg_node = next(
            n for n in state.parent.parent
            if isinstance(n, nd.NestedSDFG) and n.sdfg == state.parent)
        return is_parallel(state.parent.parent, nsdfg_node)

    return False


def find_input_arraynode(graph, edge):
    result = graph.memlet_path(edge)[0]
    if not isinstance(result.src, nd.AccessNode):
        raise RuntimeError("Input array node not found for memlet " +
                           str(edge.data))
    return result.src


def find_output_arraynode(graph, edge):
    result = graph.memlet_path(edge)[-1]
    if not isinstance(result.dst, nd.AccessNode):
        raise RuntimeError("Output array node not found for memlet " +
                           str(edge.data))
    return result.dst


def concurrent_subgraphs(graph):
    """ Finds subgraphs of an SDFGState or ScopeSubgraphView that can
        run concurrently. """
    from dace.sdfg.scope import ScopeSubgraphView

    if not isinstance(graph, (SDFGState, ScopeSubgraphView)):
        raise TypeError(
            "Expected SDFGState or ScopeSubgraphView, got: {}".format(
                type(graph).__name__))
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
    return [
        ScopeSubgraphView(graph, [n for n in all_nodes if n in sg], None)
        for sg in subgraphs
    ]


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


def _transients_in_scope(sdfg, scope, scope_dict):
    return set(
        node.data for node in scope_dict[scope.entry if scope else scope]
        if isinstance(node, nd.AccessNode) and sdfg.arrays[node.data].transient)


def local_transients(sdfg, dfg, entry_node):
    """ Returns transients local to the scope defined by the specified entry
        node in the dataflow graph. """
    state: SDFGState = dfg._graph
    scope_dict = state.scope_dict(node_to_children=True)
    scope_tree = state.scope_tree()
    current_scope = scope_tree[entry_node]

    # Start by setting shared transients as defined
    defined_transients = set(sdfg.shared_transients())

    # Get access nodes in current scope
    transients = _transients_in_scope(sdfg, current_scope, scope_dict)

    # Add transients defined in parent scopes
    while current_scope.parent is not None:
        current_scope = current_scope.parent
        defined_transients.update(
            _transients_in_scope(sdfg, current_scope, scope_dict))

    return sorted(list(transients - defined_transients))


def trace_nested_access(node, state, sdfg):
    """ 
    Given an AccessNode in a nested SDFG, trace the accessed memory
    back to the outermost scope in which it is defined.

    :param node: An access node.
    :param state: State in which the access node is located.
    :param sdfg: SDFG in which the access node is located.
    :return: A list of scopes (node, state, sdfg) in which the given
                data is accessed, from outermost scope to innermost scope.
    """
    if node.access == dtypes.AccessType.ReadWrite:
        raise NotImplementedError("Access node must be read or write only")
    curr_node = node
    curr_sdfg = sdfg
    trace = [(node, state, sdfg)]
    while curr_sdfg.parent is not None:
        curr_state = curr_sdfg.parent
        # Find the nested SDFG containing ourself in the parent state
        for nested_sdfg in curr_state.nodes():
            if isinstance(nested_sdfg,
                          nd.NestedSDFG) and nested_sdfg.sdfg == curr_sdfg:
                break
        else:
            raise ValueError("{} not found in its parent state {}".format(
                curr_sdfg.name, curr_state.label))
        if node.access == dtypes.AccessType.ReadOnly:
            for e in curr_state.in_edges(nested_sdfg):
                if e.dst_conn == curr_node.data:
                    # See if the input to this connector traces back to an
                    # access node. If not, just give up here
                    curr_node = find_input_arraynode(curr_state, e)
                    curr_sdfg = curr_state.parent  # Exit condition
                    if isinstance(curr_node, nd.AccessNode):
                        trace.append((curr_node, curr_state, curr_sdfg))
                    break
        if node.access == dtypes.AccessType.WriteOnly:
            for e in curr_state.out_edges(nested_sdfg):
                if e.src_conn == curr_node.data:
                    # See if the output of this connector traces back to an
                    # access node. If not, just give up here
                    curr_node = find_output_arraynode(curr_state, e)
                    curr_sdfg = curr_state.parent  # Exit condition
                    if isinstance(curr_node, nd.AccessNode):
                        trace.append((curr_node, curr_state, curr_sdfg))
                    break
    return list(reversed(trace))
