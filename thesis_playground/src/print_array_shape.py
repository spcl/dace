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
    data = []
    for node, state in sdfg.all_nodes_recursive():
        if (isinstance(node, nodes.AccessNode)
           and node.data == data_name
           and isinstance(state.parent.arrays[node.data], Array)):

            shape = state.parent.arrays[node.data].shape
            stride = state.parent.arrays[node.data].strides
            data.append([state, state.parent.label, shape, stride])

    print(tabulate(data, headers=["state", "sdfg", "shape", "stride"]))


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to the AccessNodes of an array")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--array', type=str, help='Name of the array', default=None)
    parser.add_argument('--temporary-arrays', action='store_true', default=False)
    args = parser.parse_args()
    sdfg = SDFG.from_file(args.sdfg_file)

    temp_arrays = ['ZLIQFRAC', 'ZPFPLSX', 'ZFOEALFA', 'ZAORIG', 'ZQSLIQ', 'ZLNEG', 'ZFOEEW', 'ZFOEEWMT', 'ZQX0', 'ZA',
                   'ZQX', 'ZQSICE', 'ZICEFRAC', 'ZQXN2D']

    if args.temporary_arrays:
        for array in temp_arrays:
            print(f"*** {array} ***")
            print_shapes_detail(sdfg, array)
    elif args.array is None:
        print_all_shapes(sdfg)
    else:
        print_shapes_detail(sdfg, args.array)

if __name__ == '__main__':
    main()
