# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple SDFG command-line compiler. """

import dace
import os
import sys
import argparse
import shutil
from dace.transformation import auto_optimize as aopt

def main():
    # Command line options parser
    parser = argparse.ArgumentParser(
        description='Simple SDFG command-line compiler.')

    # Required argument for SDFG file path
    parser.add_argument('filepath', help='<PATH TO SDFG FILE>', type=str)

    # Optional argument for output location
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        help=
        'If provided, saves library as the given file or in the specified path, '
        'together with a header file.')

    parser.add_argument('-O',
                        '--optimize',
                        dest='optimize',
                        type=str,
                        choices=['0', '1', '1.5', '2', 'manual'],
                        help='''Chooses optimization mode:
  -O0: Perform no changes to the program;
  -O1 (default): Perform dataflow coarsening (strict transformations);
  -O2: Invoke automatic optimization heuristics (see also --device flag);
  -Omanual: Use the command-line optimization tool for transformations.
''',
                        default='1')
    parser.add_argument('--device', '-D', dest='device', type=str,
                        choices=['cpu', 'gpu', 'fpga'],
                        help='Chooses device to transform code to (used '
                        'in -O1 and -O2 modes only).', default='cpu')
    parser.add_argument('--sequential',
                        default=False,
                        action='store_true',
                        dest='sequential',
                        help='Generate code without parallelism.')
    
    args = parser.parse_args()

    filepath = args.filepath
    if not os.path.isfile(filepath):
        print('SDFG file', filepath, 'not found')
        exit(1)
        
    outpath = args.out

    # Load SDFG
    sdfg = dace.SDFG.from_file(filepath)

    # Choose optimization mode
    if args.optimize == '0':
        pass
    elif args.optimize == '1':
        sdfg.apply_strict_transformations()
        if args.device == 'gpu':
            sdfg.apply_gpu_transformations()
        elif args.device == 'fpga':
            sdfg.apply_fpga_transformations()
            
    elif args.optimize == '1.5':
        dev = dace.DeviceType[args.device.upper()]
        aopt.auto_optimize(sdfg, device=dev, subgraph_fuse=False)
        
    elif args.optimize == '2':
        dev = dace.DeviceType[args.device.upper()]
        aopt.auto_optimize(sdfg, device=dev)
        
    elif args.optimize == 'manual':
        sdfg.optimize()

    if args.sequential:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.Sequential
        
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
