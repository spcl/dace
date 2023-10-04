from argparse import ArgumentParser
from typing import Optional
import logging

from dace.sdfg import SDFGState, SDFG
from dace.sdfg.nodes import NestedSDFG


def print_scopetree(state: SDFGState, indent=0, max_indent=20):
    snodes = state.scope_children()[None]
    print(' '*indent, state)
    if indent < max_indent:
        for node in snodes:
            if isinstance(node, NestedSDFG):
                for nstate in node.sdfg.states():
                    print_scopetree(nstate, indent+2)


def search_state_recursive(sdfg: SDFG, state_name: str):
    for state in sdfg.states():
        if str(state.name) == state_name:
            print_scopetree(state)
        for node in state.nodes():
            if isinstance(node, NestedSDFG):
                search_state_recursive(node.sdfg, state_name)


def main():
    parser = ArgumentParser(description="List all in and outging memlets from/to the AccessNodes of an array")
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('state', type=str, help='State to print scopetree of')
    args = parser.parse_args()
    sdfg = SDFG.from_file(args.sdfg_file)

    search_state_recursive(sdfg, args.state)

if __name__ == '__main__':
    main()
