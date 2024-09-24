# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple Fortran SDFG command-line compiler. """

import dace
import os
import sys
import argparse
import shutil
from dace.frontend.fortran import fortran_parser

def main():
    # Command line options parser
    parser = argparse.ArgumentParser(description='Fortran to SDFG command-line transpiler.')

    # Required argument for Fortran file path
    parser.add_argument('filepath', help='<PATH TO FORTRAN FILE>', type=str)

    # Optional argument for output location
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        help='If provided, saves library as the given file or in the specified path, '
                        'together with a header file.')

    parser.add_argument('-O',
                        '--optimize',
                        dest='optimize',
                        action='store_true',
                        help="If set, invokes the command-line optimization"
                        " interface",
                        default=False)

    args = parser.parse_args()

    filepath = args.filepath
    if not os.path.isfile(filepath):
        print('Fortran file', filepath, 'not found')
        exit(1)

    outpath = args.out

    # Load SDFG
    sdfg = fortran_parser.create_sdfg_from_fortran_file(filepath)

    if args.optimize:
        sdfg.optimize()

    # Compile SDFG
    sdfg.compile(outpath)

    # Copying header file to optional path
    if outpath is not None:
        source = os.path.join(sdfg.build_folder, 'include', sdfg.name + '.h')
        if os.path.isdir(outpath):
            outpath = os.path.join(outpath, sdfg.name + '.h')
        else:
            outpath = os.path.join(os.path.dirname(outpath), sdfg.name + '.h')
        shutil.copyfile(source, outpath)


if __name__ == '__main__':
    main()
