# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resolve a hybrid state by wrapping its host code that touches arrays into size-1 GPU maps."""

from collections import deque
from copy import deepcopy

import networkx as nx
from ordered_set import OrderedSet

from dace import data, dtypes, Memlet
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.graph import MultiConnectorEdge

import dace.transformation.passes.offloading.offloading_helpers as helpers


def wrap_host_code(sdfg: SDFG, state: SDFGState, host_maps: OrderedSet[nodes.MapEntry]) -> None:
    """Wrap every partition of ``state`` between its device work that touches an array in a size-1 map.

    Kernels, host maps and nested SDFGs launching kernels bound the partitions, and so does a callback, which
    only the host can run.
    """
    top_level = state.scope_children()[None]
    boundary = OrderedSet(node for node in top_level if helpers.is_device_work(node) or node in host_maps)
    boundary |= OrderedSet(state.exit_node(node) for node in boundary if isinstance(node, nodes.MapEntry))
    callbacks = helpers.callback_symbol_names(sdfg)
    separators = boundary | OrderedSet(node for node in top_level if helpers.is_callback_tasklet(node, callbacks))

    for partition in partitions(state, separators):
        if not any(
            isinstance(node, nodes.AccessNode) and not isinstance(sdfg.arrays[node.data], data.Scalar)
            for node in partition
        ):
            continue
        peel_outer_access_nodes(state, partition)
        closed = close_scopes(state, partition, separators)
        if closed:
            map_entry, map_exit = wrap_in_size1_map(state, closed)
            insert_access_between_adjacent_maps(state, map_exit)
            forward_input_only_map_data(state, map_entry, map_exit)


def partitions(state: SDFGState, separators: OrderedSet[nodes.Node]) -> list[OrderedSet[nodes.Node]]:
    """The weakly connected components of ``state`` without ``separators`` that hold a top-level node, in state
    order. A component spans the scopes of its maps."""
    top_level = state.scope_children()[None]
    rest = state.nx.subgraph(node for node in state.nodes() if node not in separators)
    components = [component for component in nx.weakly_connected_components(rest) if component & set(top_level)]
    order = {node: index for index, node in enumerate(state.nodes())}
    return [OrderedSet(sorted(component, key=order.__getitem__)) for component in components]


def peel_outer_access_nodes(state: SDFGState, group: OrderedSet[nodes.Node]) -> None:
    """Peel access nodes off the group's boundary until none is left there: they stay outside the wrapper."""
    while True:
        outer = OrderedSet(
            node
            for node in group
            if isinstance(node, nodes.AccessNode)
            and (
                all(edge.src not in group for edge in state.in_edges(node))
                or all(edge.dst not in group for edge in state.out_edges(node))
            )
        )
        if not outer:
            return
        group -= outer


def close_scopes(
    state: SDFGState, region: OrderedSet[nodes.Node], separators: OrderedSet[nodes.Node]
) -> OrderedSet[nodes.Node] | None:
    """``region`` grown until every map scope it touches lies wholly inside it, or None.

    A size-1 map around half a scope is not a scope. None when closing would swallow a node of ``separators``
    (a kernel or a device-wide call): such a group is left alone.
    """
    closed = OrderedSet(region)
    scope_children = state.scope_children()
    queue = list(region)
    while queue:
        node = queue.pop()
        if isinstance(node, nodes.MapEntry):
            entry = node
        elif isinstance(node, nodes.MapExit):
            entry = state.entry_node(node)
        else:
            continue
        for extra in [entry, state.exit_node(entry), *scope_children[entry]]:
            if extra in closed:
                continue
            if extra in separators:
                return None
            closed.add(extra)
            queue.append(extra)
    return closed


def wrap_in_size1_map(state: SDFGState, region: OrderedSet[nodes.Node]) -> tuple[nodes.MapEntry, nodes.MapExit]:
    label, param = helpers.new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
    map_entry, map_exit = state.add_map(name=label, ndrange={param: "0:1"}, schedule=dtypes.ScheduleType.GPU_Device)

    # Lists, not sets: the connector numbering follows this order.
    boundary_in = [edge for node in region for edge in state.in_edges(node) if edge.src not in region]
    boundary_out = [edge for node in region for edge in state.out_edges(node) if edge.dst not in region]
    route_through(state, boundary_in, map_entry, "REGION_IN")
    route_through(state, boundary_out, map_exit, "REGION_OUT")

    # A root or leaf no rewired edge reaches would sit outside the scope while its neighbors are inside.
    for node in region:
        if state.in_degree(node) == 0:
            state.add_nedge(map_entry, node, Memlet())
        if state.out_degree(node) == 0:
            state.add_nedge(node, map_exit, Memlet())
    return map_entry, map_exit


def route_through(state: SDFGState, edges: list[MultiConnectorEdge], scope_node: nodes.Node, prefix: str) -> None:
    """Route each edge crossing into or out of a new scope through ``scope_node``'s connectors.

    An empty memlet only orders, so it takes no connector.
    """
    index = 0
    for edge in edges:
        state.remove_edge(edge)
        if edge.data.is_empty():
            state.add_nedge(edge.src, scope_node, deepcopy(edge.data))
            state.add_nedge(scope_node, edge.dst, deepcopy(edge.data))
            continue
        in_conn, out_conn = f"IN_{prefix}_{index}", f"OUT_{prefix}_{index}"
        index += 1
        scope_node.add_in_connector(in_conn)
        scope_node.add_out_connector(out_conn)
        state.add_edge(edge.src, edge.src_conn, scope_node, in_conn, deepcopy(edge.data))
        state.add_edge(scope_node, out_conn, edge.dst, edge.dst_conn, deepcopy(edge.data))


def insert_access_between_adjacent_maps(state: SDFGState, map_exit: nodes.MapExit) -> None:
    """Route a direct map-exit to map-entry edge through an access node."""
    for edge in list(state.out_edges(map_exit)):
        if not isinstance(edge.dst, nodes.MapEntry) or edge.data.is_empty():
            continue
        access = state.add_access(edge.data.data)
        state.remove_edge(edge)
        state.add_edge(edge.src, edge.src_conn, access, None, deepcopy(edge.data))
        state.add_edge(access, None, edge.dst, edge.dst_conn, deepcopy(edge.data))


def forward_input_only_map_data(state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> None:
    """Route the last in-map access of each input that is not an output through the exit.

    Otherwise DaCe labels such an input as constant, which it is not once the wrapper writes it.
    """
    input_memlets = [edge.data for edge in state.in_edges(map_entry) if not edge.data.is_empty()]
    written = OrderedSet(edge.data.data for edge in state.out_edges(map_exit) if not edge.data.is_empty())
    input_only = OrderedSet(memlet.data for memlet in input_memlets if memlet.data not in written)
    last_accesses = last_access_nodes_in_map(state, map_entry, map_exit, input_only)

    for memlet in input_memlets:
        if memlet.data not in last_accesses:
            continue
        connector = map_exit.next_connector(memlet.data)
        in_conn, out_conn = f"IN_{connector}", f"OUT_{connector}"
        map_exit.add_in_connector(in_conn)
        map_exit.add_out_connector(out_conn)
        state.add_edge(last_accesses[memlet.data], None, map_exit, in_conn, deepcopy(memlet))
        state.add_edge(map_exit, out_conn, state.add_access(memlet.data), None, deepcopy(memlet))


def last_access_nodes_in_map(
    state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, names: OrderedSet[str]
) -> dict[str, nodes.AccessNode]:
    """The access node of each name in ``names`` that a breadth-first walk from the entry reaches last."""
    last_access: dict[str, nodes.AccessNode] = {}
    queue = deque([map_entry])
    visited = OrderedSet([map_entry])
    while names and queue:
        node = queue.popleft()
        if isinstance(node, nodes.AccessNode) and node.data in names:
            last_access[node.data] = node
        if node is map_exit:
            continue
        for child in state.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return last_access
