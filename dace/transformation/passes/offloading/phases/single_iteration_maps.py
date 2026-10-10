# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resolve a hybrid state by wrapping its host code that touches arrays into size-1 GPU maps."""

from collections import deque
from copy import deepcopy

from ordered_set import OrderedSet

import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace import Memlet, data, dtypes
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import SDFGState


def make_size1_map_wrappers(sdfg: SDFG, state: SDFGState, host_maps: OrderedSet[nodes.MapEntry]) -> None:
    """Wrap every top-level partition of ``state`` between its device work that touches an array in a size-1 map.

    Kernels, host maps and nested SDFGs launching kernels bound the partitions, and so does a callback, which
    only the host can run.
    """
    top_level = state.scope_children()[None]
    boundary = OrderedSet(node for node in top_level if helpers.is_device_work(node) or node in host_maps)
    boundary |= OrderedSet(state.exit_node(node) for node in boundary if isinstance(node, nodes.MapEntry))
    partition_nodes = boundary | OrderedSet(node for node in top_level if helpers.is_callback_tasklet(node, sdfg))

    for partition in subgraphs_after_removing(state, partition_nodes):
        if not any(
            isinstance(node, nodes.AccessNode) and not isinstance(sdfg.arrays[node.data], data.Scalar)
            for node in partition
        ):
            continue
        remove_outer_access_nodes(state, partition)
        # A partition is a dataflow component, and a map scope spans one, so it can hold a lone MapEntry.
        closed = scope_closed_partition(state, partition, partition_nodes)
        if closed:
            map_entry, map_exit = wrap_region_in_size1_map(state, closed)
            insert_access_between_adjacent_maps(state, map_exit)
            forward_input_only_map_data(state, map_entry, map_exit)


def scope_closed_partition(state: SDFGState, region: OrderedSet, boundary: OrderedSet) -> OrderedSet | None:
    """``region`` grown until every map scope it touches lies wholly inside it, or None.

    A size-1 map around half a scope is not a scope. None when closing would swallow a node of
    ``boundary`` (a kernel or a device-wide call): such a group is left alone.
    """
    closed: OrderedSet = OrderedSet(region)
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
            if extra in boundary:
                return None
            closed.add(extra)
            queue.append(extra)
    return closed


def rewire_boundary(state: SDFGState, edges: list, scope_node: nodes.Node, prefix: str) -> None:
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


def wrap_region_in_size1_map(state: SDFGState, region_nodes: OrderedSet) -> tuple[nodes.MapEntry, nodes.MapExit]:
    map_label, map_param = helpers.get_new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
    map_entry, map_exit = state.add_map(
        name=map_label, ndrange={map_param: "0:1"}, schedule=dtypes.ScheduleType.GPU_Device
    )

    # Lists, not sets: the connector numbering follows this order.
    boundary_in = [e for node in region_nodes for e in state.in_edges(node) if e.src not in region_nodes]
    boundary_out = [e for node in region_nodes for e in state.out_edges(node) if e.dst not in region_nodes]
    rewire_boundary(state, boundary_in, map_entry, "REGION_IN")
    rewire_boundary(state, boundary_out, map_exit, "REGION_OUT")

    # A root or leaf no rewired edge reaches would sit outside the scope while its neighbors are inside.
    for node in region_nodes:
        if state.in_degree(node) == 0:
            state.add_nedge(map_entry, node, Memlet())
        if state.out_degree(node) == 0:
            state.add_nedge(node, map_exit, Memlet())
    return map_entry, map_exit


def subgraphs_after_removing(state: SDFGState, partition_nodes: OrderedSet) -> list[OrderedSet]:
    """Weakly connected components of the state's top level once ``partition_nodes`` are removed."""
    visited: OrderedSet = OrderedSet()
    components = []
    for start in state.scope_children()[None]:
        if start in partition_nodes or start in visited:
            continue
        component: OrderedSet = OrderedSet()
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in [*state.successors(node), *state.predecessors(node)]:
                if neighbor not in partition_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def remove_outer_access_nodes(state: SDFGState, group: OrderedSet) -> None:
    """Peel access nodes off the group's boundary until none is left there: they stay outside the wrapper."""
    while True:
        outer = OrderedSet(
            node
            for node in group
            if isinstance(node, nodes.AccessNode)
            and (
                all(e.src not in group for e in state.in_edges(node))
                or all(e.dst not in group for e in state.out_edges(node))
            )
        )
        if not outer:
            return
        group -= outer


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

    for input_memlet in input_memlets:
        if input_memlet.data not in last_accesses:
            continue
        connector = map_exit.next_connector(input_memlet.data)
        in_conn, out_conn = f"IN_{connector}", f"OUT_{connector}"
        map_exit.add_in_connector(in_conn)
        map_exit.add_out_connector(out_conn)
        state.add_edge(last_accesses[input_memlet.data], None, map_exit, in_conn, deepcopy(input_memlet))
        state.add_edge(map_exit, out_conn, state.add_access(input_memlet.data), None, deepcopy(input_memlet))


def last_access_nodes_in_map(
    state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, data_names: OrderedSet
) -> dict[str, nodes.AccessNode]:
    """The access node of each name in ``data_names`` that a breadth-first walk from the entry reaches last."""
    last_access: dict[str, nodes.AccessNode] = {}
    queue = deque([map_entry])
    visited = OrderedSet([map_entry])
    while data_names and queue:
        node = queue.popleft()
        if isinstance(node, nodes.AccessNode) and node.data in data_names:
            last_access[node.data] = node
        if node is map_exit:
            continue
        for child in state.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return last_access
