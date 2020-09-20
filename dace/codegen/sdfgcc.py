# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple SDFG command-line compiler. """

import dace
import os
import sys
import argparse


def main():
    # Command line options parser
    parser = argparse.ArgumentParser(description='Simple SDFG command-line compiler.')

    # Required argument for SDGF file path
    parser.add_argument('SDFGfile', help='<PATH TO SDFG FILE>', type=str)

    # Optional argument for output location
    parser.add_argument('-o','--out', type=str, help='If provided, save output to path or filename')

    args = parser.parse_args()

    filename = args.SDFGfile
    if not os.path.isfile(filename):
        print('SDFG file', filename, 'not found')
        exit(1)

    # Load SDFG
    sdfg = dace.SDFG.from_file(filename)

    # Compile SDFG
    sdfg.compile(args.out)

if __name__ == '__main__':
    main()