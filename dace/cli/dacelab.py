#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
import numpy
import pickle
import json

import dace
from dace.frontend.octave import parse
from dace.sdfg.nodes import AccessNode


def compile(inputfile):
    buf = open(inputfile).read()
    statements = parse.parse(buf, debug=False)
    print("After Parsing")
    print(str(statements))
    print("===============")
    statements.provide_parents()
    statements.specialize()
    print("After Specialization:")
    print(str(statements))
    print("===============")
    sdfg = statements.generate_code()
    sdfg.fill_scope_connectors()
    sdfg.set_sourcecode(buf, "matlab")

    # Clean isolated nodes
    for state in sdfg.nodes():
        for node in state.nodes():
            if (isinstance(node, AccessNode) and (state.in_degree(node) + state.out_degree(node) == 0)):
                state.remove_node(node)

    return sdfg


def main():
    argparser = argparse.ArgumentParser(description="dacelab: An Octave to SDFG compiler")
    argparser.add_argument("infile", metavar='infile', type=argparse.FileType('r'), help="Input file (Octave code)")
    argparser.add_argument("-o",
                           "--outfile",
                           metavar='outfile',
                           type=argparse.FileType('w'),
                           default="out.sdfg",
                           help="Output file, defaults to out.sdfg")
    args = argparser.parse_args()
    sdfg = compile(args.infile.name)
    sdfg.save(args.outfile.name)
    print("SDFG Generation finished")


if __name__ == "__main__":
    main()
