from argparse import ArgumentParser
from tabulate import tabulate

import dace
from dace.data import Array
from dace.sdfg import nodes, SDFG

def print_all_shapes(sdfg: SDFG):
    shapes = {}

    for node, state in sdfg.all_nodes_recursive():
        if (isinstance(node, nodes.AccessNode)
           and node.data[0] == 'Z'
           # and state.parent.arrays[node.data].transient
           and not node.data[0:3] == 'gpu'
           and isinstance(state.parent.arrays[node.data], Array)):

            if node.data not in shapes:
                shapes[node.data] = [set(), set()]
            shapes[node.data][0].add(state.parent.arrays[node.data].shape)
            shapes[node.data][1].add(state.parent.arrays[node.data].strides)

    shapes_array = []
    for array in sorted(shapes):
        shapes_array.append([array, *shapes[array]])
    print(tabulate(shapes_array, headers=["array", "shape", "strides"]))


def print_shapes_detail(sdfg: SDFG, data_name: str):
    shapes = {}
    for node, state in sdfg.all_nodes_recursive():
        if (isinstance(node, nodes.AccessNode)
           and node.data == data_name):

            shape = state.parent.arrays[node.data].shape
            if shape not in shapes:
                shapes[shape] = []
            shapes[shape].append(state)

    shapes_array = []
    for shape in shapes:
        shapes_array.append([shape, *shapes[shape]])
    print(tabulate(shapes_array, headers=["shape", "states"]))


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to the AccessNodes of an array")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--array', type=str, help='Name of the array', default=None)
    args = parser.parse_args()
    sdfg = SDFG.from_file(args.sdfg_file)

    if args.array is None:
        print_all_shapes(sdfg)
    else:
        print_shapes_detail(sdfg, args.array)

if __name__ == '__main__':
    main()
