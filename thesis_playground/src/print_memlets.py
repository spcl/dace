from argparse import ArgumentParser
from typing import Optional
import logging

import dace
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG

logger = logging.getLogger(__name__)


def check_memlet(memlet: Memlet) -> bool:
    """
    Checks if memlet has the stange access pattern 0:_for_it_xxx

    :param memlet: [TODO:description]
    :type memlet: Memlet
    :return: [TODO:description]
    :rtype: bool
    """
    if len(memlet.subset) > 1:
        # if memlet.subset[1][0] == 0 and dace.symbolic.issymbolic(memlet.subset[1][1]):
        if (memlet.subset[1][0] == 0
                and dace.symbolic.issymbolic(memlet.subset[1][1])
                and str(memlet.subset[1][1]).startswith('_')):
            return True
    return False


def fix_memlet(memlet: Memlet, print: bool = False):
    to_fix = ['ZQX', 'ZA', 'ZTP1', 'ZPFPLSX']
    if memlet.data in to_fix:
        if print:
            print(f"Change memlet {memlet} to subset ", end="")
        memlet.subset[1] = (memlet.subset[1][1]-1, memlet.subset[1][1], memlet.subset[1][2])
        if print:
            print(memlet.subset)


def find_all_strange_memlets(sdfg: SDFG, new_name: Optional[str] = None) -> SDFG:
    """
    Find all strange memlets in the temporary arrays and fix them

    :param sdfg: [TODO:description]
    :type sdfg: SDFG
    :param new_name: [TODO:description]
    :type new_name: Optional[str]
    """
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            for iedge in state.in_edges(node):
                if check_memlet(iedge.data):
                    fix_memlet(iedge.data)
                for edge in state.memlet_path(iedge):
                    if check_memlet(edge.data):
                        fix_memlet(edge.data)
            for oedge in state.out_edges(node):
                if check_memlet(oedge.data):
                    fix_memlet(oedge.data)
                for edge in state.memlet_path(oedge):
                    if check_memlet(edge.data):
                        fix_memlet(edge.data)

    if new_name is not None:
        print(f"Save new graph into {new_name}")
        sdfg.save(new_name)
    return sdfg


def print_memlet_of_arrays(sdfg: SDFG, array: str):
    memlets_in = []
    memlets_out = []
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and node.data == array:
            for iedge in state.in_edges(node):
                memlets_in.append((iedge.data.subset, iedge.src, iedge.dst, state))
                for edge in state.memlet_path(iedge):
                    memlets_in.append((edge.data.subset, edge.src, edge.dst, state))
            for oedge in state.out_edges(node):
                memlets_out.append((oedge.data.subset, oedge.src, oedge.dst, state))
                for edge in state.memlet_path(oedge):
                    memlets_out.append((edge.data.subset, edge.src, edge.dst, state))

    print("In edges")
    for memlet_in in memlets_in:
        print(f"  {memlet_in}")
    print("out edges")
    for memlet_out in memlets_out:
        print(f"  {memlet_out}")


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to the AccessNodes of an array")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--array', type=str, help='Name of the array', default=None)
    parser.add_argument('--temporary-arrays', action='store_true', default=False)
    args = parser.parse_args()
    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    temp_arrays = ['ZLIQFRAC', 'ZPFPLSX', 'ZFOEALFA', 'ZAORIG', 'ZQSLIQ', 'ZLNEG', 'ZFOEEW', 'ZFOEEWMT', 'ZQX0', 'ZA',
                   'ZQX', 'ZQSICE', 'ZICEFRAC', 'ZQXN2D']

    if args.temporary_arrays:
        for array in temp_arrays:
            print(f"*** {array} ***")
            print_memlet_of_arrays(sdfg, array)
    elif args.array is not None:
        print_memlet_of_arrays(sdfg, args.array)
    else:
        find_all_strange_memlets(sdfg, 'fixed_memlets.sdfg')


if __name__ == '__main__':
    main()
