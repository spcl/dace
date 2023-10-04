from argparse import ArgumentParser
from typing import Optional
import logging
from tabulate import tabulate

import dace
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG

logger = logging.getLogger(__name__)


def print_interstate_of_arrays(sdfg: SDFG, array: str):
    iedges = []
    for edge, parent in sdfg.all_edges_recursive():


    print("In edges")
    print(tabulate(memlets_in, headers=["subset", "src", "dst", "state", "parent sdfg"]))
    print()
    print("out edges")
    print(tabulate(memlets_out, headers=["subset", "src", "dst", "state", "parent sdfg"]))


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to the AccessNodes of an array")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--array', type=str, help='Name of the array', default=None)
    parser.add_argument('--temporary-arrays', action='store_true', default=False)
    parser.add_argument('--nsdfg', help="print memlets inside the given nested SDFG", default=None)
    args = parser.parse_args()
    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    temp_arrays = ['ZLIQFRAC', 'ZPFPLSX', 'ZFOEALFA', 'ZAORIG', 'ZQSLIQ', 'ZLNEG', 'ZFOEEW', 'ZFOEEWMT', 'ZQX0', 'ZA',
                   'ZQX', 'ZQSICE', 'ZICEFRAC', 'ZQXN2D']

    if args.nsdfg:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.NestedSDFG) and node.label == args.nsdfg:
                sdfg = node.sdfg
                break

    if args.temporary_arrays:
        for array in temp_arrays:
            print(f"*** {array} ***")
            print_memlet_of_arrays(sdfg, array)
    elif args.array is not None:
        print_memlet_of_arrays(sdfg, args.array)


if __name__ == '__main__':
    main()
