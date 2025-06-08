# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG diff tool. """

import argparse
from hashlib import sha256
import json
import os
import platform
import tempfile
from typing import Dict, Set, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import dace
from dace import memlet as mlt
from dace.sdfg import nodes as nd
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowBlock
import dace.serialize

DiffableT = Union[ControlFlowBlock, nd.Node, MultiConnectorEdge[mlt.Memlet], Edge[InterstateEdge]]
DiffSetsT = Tuple[Set[str], Set[str], Set[str]]


def _print_diff(sdfg_A: dace.SDFG, sdfg_B: dace.SDFG, diff_sets: DiffSetsT) -> None:
    all_id_elements_A: Dict[str, DiffableT] = dict()
    all_id_elements_B: Dict[str, DiffableT] = dict()

    all_id_elements_A[sdfg_A.guid] = sdfg_A
    for n, _ in sdfg_A.all_nodes_recursive():
        all_id_elements_A[n.guid] = n
    for e, _ in sdfg_A.all_edges_recursive():
        all_id_elements_A[e.data.guid] = e

    all_id_elements_B[sdfg_B.guid] = sdfg_B
    for n, _ in sdfg_B.all_nodes_recursive():
        all_id_elements_B[n.guid] = n
    for e, _ in sdfg_B.all_edges_recursive():
        all_id_elements_B[e.data.guid] = e

    no_removed = True
    no_added = True
    no_changed = True
    if len(diff_sets[0]) > 0:
        print('Removed elements:')
        for k in diff_sets[0]:
            print(all_id_elements_A[k])
        no_removed = False
    if len(diff_sets[1]) > 0:
        if not no_removed:
            print('')
        print('Added elements:')
        for k in diff_sets[1]:
            print(all_id_elements_B[k])
        no_added = False
    if len(diff_sets[2]) > 0:
        if not no_removed or not no_added:
            print('')
        print('Changed elements:')
        for k in diff_sets[2]:
            print(all_id_elements_B[k])
        no_changed = False

    if no_removed and no_added and no_changed:
        print('SDFGs are identical')


def _sdfg_diff(sdfg_A: dace.SDFG, sdfg_B: dace.SDFG, eq_strategy=Union[Literal['hash', '==']]) -> DiffSetsT:
    all_id_elements_A: Dict[str, DiffableT] = dict()
    all_id_elements_B: Dict[str, DiffableT] = dict()

    all_id_elements_A[sdfg_A.guid] = sdfg_A
    for n, _ in sdfg_A.all_nodes_recursive():
        all_id_elements_A[n.guid] = n
    for e, _ in sdfg_A.all_edges_recursive():
        all_id_elements_A[e.data.guid] = e

    all_id_elements_B[sdfg_B.guid] = sdfg_B
    for n, _ in sdfg_B.all_nodes_recursive():
        all_id_elements_B[n.guid] = n
    for e, _ in sdfg_B.all_edges_recursive():
        all_id_elements_B[e.data.guid] = e

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

        if eq_strategy == 'hash':
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
        else:
            if isinstance(el_a, Edge):
                attr_a = dace.serialize.all_properties_to_json(el_a.data)
            else:
                attr_a = dace.serialize.all_properties_to_json(el_a)
            if isinstance(el_b, Edge):
                attr_b = dace.serialize.all_properties_to_json(el_b.data)
            else:
                attr_b = dace.serialize.all_properties_to_json(el_b)

            if attr_a != attr_b:
                changed_keys.add(k)

    return removed_keys, added_keys, changed_keys


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
    parser.add_argument('-o', '--output', dest='output', help="The output filename to generate", type=str)
    parser.add_argument('-H',
                        '--hash',
                        dest='hash',
                        action='store_true',
                        help="If set, use the hash of JSON serialized properties for change checks instead of " +
                        "Python's dictionary equivalence checks. This makes changes order sensitive.",
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

    eq_strategy = 'hash' if args.hash else '=='

    diff_sets = _sdfg_diff(sdfg_A, sdfg_B, eq_strategy)

    if args.graphical:
        try:
            import jinja2
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Graphical SDFG diff requires jinja2, please install by running `pip install jinja2`')

        basepath = os.path.join(os.path.dirname(os.path.realpath(dace.__file__)), 'viewer')
        template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(basepath, 'templates'))
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template('sdfv_diff_view.html')

        # if we are serving, the base path should just be root
        html = template.render(sdfgA=json.dumps(dace.serialize.dumps(sdfg_A.to_json())),
                               sdfgB=json.dumps(dace.serialize.dumps(sdfg_B.to_json())),
                               removedKeysList=json.dumps(list(diff_sets[0])),
                               addedKeysList=json.dumps(list(diff_sets[1])),
                               changedKeysList=json.dumps(list(diff_sets[2])),
                               dir=basepath + '/')

        if args.output:
            fd = None
            html_filename = args.output
        else:
            fd, html_filename = tempfile.mkstemp(suffix=".sdfg.html")

        with open(html_filename, 'w') as f:
            f.write(html)

        if fd is not None:
            os.close(fd)

        system = platform.system()

        if system == 'Windows':
            os.system(html_filename)
        elif system == 'Darwin':
            os.system('open %s' % html_filename)
        else:
            os.system('xdg-open %s' % html_filename)
    else:
        _print_diff(sdfg_A, sdfg_B, diff_sets)


if __name__ == '__main__':
    main()
