from typing import Dict
import dace
import copy


def copy_state_contents(old_state: dace.SDFGState, new_state: dace.SDFGState) -> Dict[dace.nodes.Node, dace.nodes.Node]:
    node_map = dict()

    for n in old_state.nodes():
        c_n = copy.deepcopy(n)
        node_map[n] = c_n
        new_state.add_node(c_n)

    for e in old_state.edges():
        c_src = node_map[e.src]
        c_dst = node_map[e.dst]

        new_state.add_edge(c_src, e.src_conn, c_dst, e.dst_conn, copy.deepcopy(e.data))

    return node_map
