# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple SDFG command-line compiler. """

import dace
import os
import sys
import argparse
import shutil


def main():
    # Command line options parser
    parser = argparse.ArgumentParser(description='Simple SDFG command-line compiler.')

    # Required argument for SDGF file path
    parser.add_argument('SDFGfilepath', help='<PATH TO SDFG FILE>', type=str)

    # Optional argument for output location
    parser.add_argument('-o','--out', type=str, help='If provided, saves lib and header file to path. Directories in path need to exist beforehand.')

    args = parser.parse_args()

    filepath = args.SDFGfilepath
    if not os.path.isfile(filepath):
        print('SDFG file', filepath, 'not found')
        exit(1)

    # Load SDFG
    sdfg = dace.SDFG.from_file(filepath)

    # Compile SDFG
    sdfg.compile(args.out)

    # Copying header file to optional path
    if args.out:
        source = os.path.join(sdfg.build_folder, 'src/cpu', sdfg.name+'.h')
        destination = os.path.join(args.out,sdfg.name+'.h')
        shutil.copyfile(source, destination)


if __name__ == '__main__':
    main()