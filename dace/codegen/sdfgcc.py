# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple SDFG command-line compiler. """

import dace
import os
import sys


def main():
    if len(sys.argv) != 2:
        print('USAGE: sdfgcc <PATH TO SDFG FILE>')
        exit(1)

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print('SDFG file', filename, 'not found')
        exit(2)

    # Load SDFG
    sdfg = dace.SDFG.from_file(filename)

    # Compile SDFG
    sdfg.compile()


if __name__ == '__main__':
    main()
