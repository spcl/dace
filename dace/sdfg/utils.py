""" Various utility functions to traverse and modify SDFGs. """

import collections
from dace.sdfg.sdfg import SDFG, SDFGState
from dace.graph import nodes as nd
from dace import subsets as sbs
from typing import Union


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
