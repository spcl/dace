import dace.transformation.helpers as xfh
import dace.sdfg.analysis.cutout as cutter

from dace import SDFG, SDFGState, nodes


def cutout_map(state: SDFGState, map_entry: nodes.MapEntry, make_copy: bool):
    map_exit = state.exit_node(map_entry)
    subgraph_nodes = state.all_nodes_between(map_entry, map_exit)
    subgraph_nodes.add(map_entry)
    subgraph_nodes.add(map_exit)

    for edge in state.in_edges(map_entry):
        subgraph_nodes.add(edge.src)

    for edge in state.out_edges(map_exit):
        subgraph_nodes.add(edge.dst)

    subgraph_nodes = list(set(subgraph_nodes))
    cutout = cutter.cutout_state(state, *subgraph_nodes, make_copy=make_copy)
    return cutout


def map_levels(map: SDFG):
    levels = {}
    for node in map.start_state.nodes():
        if not isinstance(node, nodes.MapEntry):
            continue

        parent_map = xfh.get_parent_map(map.start_state, node)
        if not parent_map is None:
            parent_map, _ = parent_map

        # CONSTRAINT: Only chains
        assert parent_map not in levels

        levels[parent_map] = node

    return levels


def map_params(map: SDFG):
    levels = map_levels(map)

    desc = []
    map_entry = None
    while map_entry in levels:
        map_entry = levels[map_entry]
        desc.append(map_entry.map.params)

    return desc
