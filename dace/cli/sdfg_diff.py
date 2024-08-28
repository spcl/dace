# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG diff tool. """

import argparse
from hashlib import sha256
import json
import os
from typing import Dict, Union
import dace
from dace import memlet as mlt
from dace.sdfg import nodes as nd
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowBlock
import dace.serialize


DiffableT = Union[ControlFlowBlock, nd.Node, MultiConnectorEdge[mlt.Memlet], Edge[InterstateEdge]]


def _sdfg_diff(sdfg_A: dace.SDFG, sdfg_B: dace.SDFG):
    all_id_elements_A: Dict[str, DiffableT] = dict()
    all_id_elements_B: Dict[str, DiffableT] = dict()

    all_id_elements_A[sdfg_A.id] = sdfg_A
    for n, _ in sdfg_A.all_nodes_recursive():
        all_id_elements_A[n.id] = n
    for e, _ in sdfg_A.all_edges_recursive():
        all_id_elements_A[e.data.id] = e

    all_id_elements_B[sdfg_B.id] = sdfg_B
    for n, _ in sdfg_B.all_nodes_recursive():
        all_id_elements_B[n.id] = n
    for e, _ in sdfg_B.all_edges_recursive():
        all_id_elements_B[e.data.id] = e

    a_keys = set(all_id_elements_A.keys())
    b_keys = set(all_id_elements_B.keys())

    added_keys = b_keys - a_keys
    removed_keys = a_keys - b_keys
    changed_keys = set()

    remaining_keys = a_keys - removed_keys
    if remaining_keys != b_keys - added_keys:
        raise RuntimeError(
            'The sets of remaining keys between graphs A and B after accounting for added and removed keys do not match'
        )
    for k in remaining_keys:
        el_a = all_id_elements_A[k]
        el_b = all_id_elements_B[k]

        try:
            if isinstance(el_a, Edge):
                attr_a = dace.serialize.all_properties_to_json(el_a.data)
            else:
                attr_a = dace.serialize.all_properties_to_json(el_a)
            hash_a = sha256(json.dumps(attr_a).encode('utf-8')).hexdigest()
        except KeyError:
            hash_a = None
        try:
            if isinstance(el_b, Edge):
                attr_b = dace.serialize.all_properties_to_json(el_b.data)
            else:
                attr_b = dace.serialize.all_properties_to_json(el_b)
            hash_b = sha256(json.dumps(attr_b).encode('utf-8')).hexdigest()
        except KeyError:
            hash_b = None

        if hash_a != hash_b:
            changed_keys.add(k)

    print('Removed Elements:')
    for k in removed_keys:
        print(all_id_elements_A[k])

    print('')

    print('Added Elements:')
    for k in added_keys:
        print(all_id_elements_B[k])

    print('')

    print('Changed Elements:')
    for k in changed_keys:
        print(all_id_elements_B[k])


def main():
    # Command line options parser
    parser = argparse.ArgumentParser(description='SDFG diff tool.')

    # Required argument for SDFG file path
    parser.add_argument('sdfg_A_path', help='<PATH TO FIRST SDFG FILE>', type=str)
    parser.add_argument('sdfg_B_path', help='<PATH TO SECOND SDFG FILE>', type=str)

    parser.add_argument('-g',
                        '--graphical',
                        dest='graphical',
                        action='store_true',
                        help="If set, visualize the difference graphically",
                        default=False)

    args = parser.parse_args()

    if not os.path.isfile(args.sdfg_A_path):
        print('SDFG file', args.sdfg_A_path, 'not found')
        exit(1)

    if not os.path.isfile(args.sdfg_B_path):
        print('SDFG file', args.sdfg_B_path, 'not found')
        exit(1)

    sdfg_A = dace.SDFG.from_file(args.sdfg_A_path)
    sdfg_B = dace.SDFG.from_file(args.sdfg_B_path)

    _sdfg_diff(sdfg_A, sdfg_B)


if __name__ == '__main__':
    main()
