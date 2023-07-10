from argparse import ArgumentParser

import dace

from execute.parameters import ParametersProvider
from utils.general import get_programs_data, get_sdfg, reset_graph_files, read_source, enable_debug_flags, optimize_sdfg


def main():
    parser = ArgumentParser()
    parser.add_argument('program', type=str, help='Name of the program to generate the SDFGs of')
    parser.add_argument(
        '--only-graph',
        action='store_true',
        help='Does not compile the SDFGs into C++ code, only creates the SDFGs and runs the transformations')
    parser.add_argument('--debug', action='store_true', default=False, help="Configure for debug build")
    parser.add_argument('--not-specialise', action='store_true', help='Do not specialise symbols')
    parser.add_argument('--KLON', type=int, default=None)
    parser.add_argument('--KLEV', type=int, default=None)
    parser.add_argument('--NBLOCKS', type=int, default=None)
    parser.add_argument('--KIDIA', type=int, default=None)
    parser.add_argument('--KFDIA', type=int, default=None)

    device = dace.DeviceType.GPU
    args = parser.parse_args()

    if args.debug:
        enable_debug_flags()

    reset_graph_files(args.program)

    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    if args.program in programs:
        program_name = programs[args.program]
    else:
        program_name = args.program
    sdfg = get_sdfg(fsource, program_name)

    add_args = {}
    if not args.not_specialise:
        params = ParametersProvider(args.program)
        params.update_from_args(args)
        print(f"Use {params} for specialisation")
        add_args['symbols'] = params.get_dict()

    optimize_sdfg(sdfg, device, verbose_name=args.program, **add_args)
    sdfg.instrument = dace.InstrumentationType.Timer
    if not args.only_graph:
        sdfg.compile()
    return sdfg


if __name__ == '__main__':
    main()
