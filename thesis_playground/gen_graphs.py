from argparse import ArgumentParser

import dace

from test import read_source
from utils import get_programs_data, save_graph, get_sdfg, reset_graph_files
from my_auto_opt import auto_optimize


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'program',
            type=str,
            help='Name of the program to generate the SDFGs of')
    parser.add_argument(
            '--normalize-memlets',
            action='store_true',
            default=False)
    parser.add_argument(
            '--only-graph',
            action='store_true',
            help='Does not compile the SDFGs into C++ code, only creates the SDFGs and runs the transformations')

    device = dace.DeviceType.GPU
    args = parser.parse_args()

    reset_graph_files(args.program)

    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    program_name = programs[args.program]
    sdfg = get_sdfg(fsource, program_name, args.normalize_memlets)
    save_graph(sdfg, args.program, "before_auto_opt")
    auto_optimize(sdfg, device, program=args.program)
    save_graph(sdfg, args.program, "after_auto_opt")
    sdfg.instrument = dace.InstrumentationType.Timer
    if not args.only_graph:
        sdfg.compile()
    return sdfg


if __name__ == '__main__':
    main()
