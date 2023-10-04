from argparse import ArgumentParser
from typing import Union
import logging

import dace
from dace.data import Array
from dace.sdfg import SDFGState
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG

logger = logging.getLogger(__name__)


def print_memlet(memlet: Memlet):
    print(f"{memlet.data:25}: {memlet.subset}")


def print_memlets_of_map_node(node: Union[nodes.MapEntry, nodes.MapExit], state: SDFGState):
    print("   * In edges *")
    for inedge in state.in_edges(node):
        if inedge.data.data in state.parent.arrays and isinstance(state.parent.data(inedge.data.data), Array):
            print_memlet(inedge.data)
    print("   * Out edges *")
    for inedge in state.out_edges(node):
        if inedge.data.data in state.parent.arrays and isinstance(state.parent.data(inedge.data.data), Array):
            print_memlet(inedge.data)


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to a map")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('nsdfg', type=str, help='Name of the map')
    args = parser.parse_args()
    sdfg = SDFG.from_file(args.sdfg_file)

    map_entry, map_state = None, None
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG) and node.label == args.nsdfg:
            map_entry = node
            map_state = state

    print("*** Map Entry ***")
    print_memlets_of_map_node(map_entry, map_state)


if __name__ == '__main__':
    main()
