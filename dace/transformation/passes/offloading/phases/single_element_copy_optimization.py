# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move a single-element copy into the map that alone reads it: ``A -> s -> Map`` becomes ``A -> Map -> s``."""
from copy import deepcopy
from typing import List

from ordered_set import OrderedSet

from dace import data, Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers


def drop_stale_other_subset(memlet: Memlet, src_node: nodes.Node, dst_node: nodes.Node) -> None:
    """Clear ``other_subset`` once the edge no longer runs between two containers: it would describe
    a node that is not on the edge any more."""
    if not isinstance(src_node, nodes.AccessNode) or not isinstance(dst_node, nodes.AccessNode):
        memlet.other_subset = None


def single_element_copies_into_map(sdfg: SDFG) -> None:
    changes = []
    for state in sdfg.states():
        for map_entry in state.nodes():
            if not isinstance(map_entry, nodes.MapEntry):
                continue
            for access in OrderedSet(state.predecessors(map_entry)):
                # Only a copy the map alone reads moves: another reader would be left reading a node
                # inside this map's scope.
                if (isinstance(access, nodes.AccessNode) and state.out_degree(access) == 1 and
                    (isinstance(sdfg.arrays[access.data], data.Scalar) or helpers.is_length1_array(access.data, sdfg))
                        and state.in_degree(access) == 1
                        and isinstance(state.in_edges(access)[0].src, nodes.AccessNode)):
                    changes.append((state, access, map_entry))

    for state, access, map_entry in changes:
        rewire_access_into_map(state, access, map_entry)


def rewire_access_into_map(state: SDFGState, access: nodes.AccessNode, map_entry: nodes.MapEntry) -> None:
    """Rewire ``B -> access -> map -> C`` into ``B -> map -> access -> C``."""
    source_edge = state.in_edges(access)[0]
    access_to_map = state.out_edges(access)[0]

    connector = map_entry.next_connector(access.data)
    in_conn, out_conn = f"IN_{connector}", f"OUT_{connector}"
    map_entry.add_in_connector(in_conn)
    map_entry.add_out_connector(out_conn)
    state.remove_edge(source_edge)
    outer, inner = deepcopy(source_edge.data), deepcopy(source_edge.data)
    drop_stale_other_subset(outer, source_edge.src, map_entry)
    drop_stale_other_subset(inner, map_entry, access)
    state.add_edge(source_edge.src, source_edge.src_conn, map_entry, in_conn, outer)
    state.add_edge(map_entry, out_conn, access, None, inner)

    # The access now feeds the map's readers directly.
    passthrough = passthrough_out_connectors(access_to_map.dst_conn)
    readers = [e for e in state.out_edges(map_entry) if e.src_conn in passthrough]
    if access_to_map.dst_conn is not None:
        map_entry.remove_in_connector(access_to_map.dst_conn)
    for connector in passthrough:
        map_entry.remove_out_connector(connector)
    state.remove_edge(access_to_map)
    for edge in readers:
        memlet = deepcopy(edge.data)
        drop_stale_other_subset(memlet, access, edge.dst)
        state.add_edge(access, None, edge.dst, edge.dst_conn, memlet)
        state.remove_edge(edge)


def passthrough_out_connectors(in_connector: str) -> List[str]:
    """``OUT_x`` for a map entry's ``IN_x``; none for a connector that is not a pass-through."""
    if in_connector is None or not in_connector.startswith("IN_"):
        return []
    return ["OUT_" + in_connector[3:]]
