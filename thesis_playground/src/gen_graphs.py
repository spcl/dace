from argparse import ArgumentParser
import logging

import dace

from execute.parameters import ParametersProvider
from utils.run_config import RunConfig
from utils.generate_sdfg import get_optimised_sdfg, remove_basic_sdfg
from utils.general import get_programs_data, get_sdfg, reset_graph_files, read_source, enable_debug_flags, \
                          optimize_sdfg, save_graph
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.log import setup_logging


def main():
    parser = ArgumentParser()
    parser.add_argument('program', type=str, help='Name of the program to generate the SDFGs of')
    parser.add_argument(
        '--only-graph',
        action='store_true',
        help='Does not compile the SDFGs into C++ code, only creates the SDFGs and runs the transformations')
    parser.add_argument('--debug', action='store_true', default=False, help="Configure for debug build")
    parser.add_argument('--not-specialise', action='store_true', help='Do not specialise symbols')
    parser.add_argument('--k-caching', action='store_true', default=False, help="use k-caching")
    parser.add_argument('--change-stride', action='store_true', default=False, help="change stride")
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--verbose-name',
                        type=str,
                        default=None,
                        help='Foldername under which intermediate SDFGs should be stored, uses the program name by '
                        'default')
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')
    parser.add_argument('--no-outer-loop-first', action='store_true', default=False, help='Disable outer loops first')
    parser.add_argument('--log-file', default=None, help='Path to logfile with level DEBUG')
    parser.add_argument('--log-level', default='info')
    parser.add_argument('--transfer-to-gpu', default=False, action='store_true',
                        help='Disable setting storage location for input arrays to GPU global')
    parser.add_argument('--clear-basic-graph', action='store_true', default=False, help="Remove stored basic SDFG")
    add_cloudsc_size_arguments(parser)

    args = parser.parse_args()
    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    if args.log_file is not None:
        setup_logging(full_logfile=args.log_file, level=args.log_level.upper())
    else:
        setup_logging(level=args.log_level.upper())

    if args.debug:
        enable_debug_flags()

    verbose_name = args.program
    if args.verbose_name is not None:
        verbose_name = args.verbose_name
    reset_graph_files(verbose_name)


    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    if args.program in programs:
        program_name = programs[args.program]
    else:
        program_name = args.program

    add_args = {}
    if not args.not_specialise:
        params = ParametersProvider(args.program)
        params.update_from_args(args)
        print(f"Use {params} for specialisation")
        add_args['symbols'] = params.get_dict()

    add_args['k_caching'] = args.k_caching
    add_args['change_stride'] = args.change_stride
    if args.use_dace_auto_opt:
        add_args['use_my_auto_opt'] = False
    if args.no_outer_loop_first:
        add_args['outside_first'] = False
    # sdfg = get_sdfg(fsource, program_name)
    # sdfg = optimize_sdfg(sdfg, device, verbose_name=verbose_name, **add_args)
    run_config = RunConfig()
    run_config.set_from_args(args)
    run_config.device = device

    if args.clear_basic_graph:
        remove_basic_sdfg(args.program, run_config, ['NBLOCKS'])

    sdfg = get_optimised_sdfg(args.program, run_config, params, ['NBLOCKS'], verbose_name=verbose_name,
                              storage_on_gpu=not args.transfer_to_gpu)
    save_graph(sdfg, verbose_name, "after_auto_opt")
    # if not args.only_graph:
    #     sdfg.compile()


if __name__ == '__main__':
    main()
