from argparse import ArgumentParser
import dace
import numpy as np
import copy

from execute.parameters import ParametersProvider
from utils.log import setup_logging
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import get_programs_data, get_sdfg, read_source, optimize_sdfg, generate_arguments_fortran, \
                          get_fortran, compare_output_all, use_cache, reset_graph_files


def main():
    parser = ArgumentParser()
    parser.add_argument('program')
    parser.add_argument('--debug', action='store_true', default=False, help="Configure for debug build")
    parser.add_argument('--not-specialise', action='store_true', help='Do not specialise symbols')
    parser.add_argument('--k-caching', action='store_true', default=False, help="use k-caching")
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--compare-to-fortran', default=False, action='store_true')
    parser.add_argument('--cache', default=False, action='store_true')
    parser.add_argument('--sdfg-file', type=str, default=None, help='File to read sdfg from')
    parser.add_argument('--verbose-name',
                        type=str,
                        default=None,
                        help='Foldername under which intermediate SDFGs should be stored')
    parser.add_argument('--change-stride', action='store_true', default=False, help="change stride")
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')
    parser.add_argument('--no-outer-loop-first', action='store_true', default=False, help='Disable outer loops first')
    add_cloudsc_size_arguments(parser)

    args = parser.parse_args()
    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    setup_logging()

    add_args = {}
    params = ParametersProvider(args.program)
    params.update_from_args(args)
    if not args.not_specialise:
        add_args['symbols'] = params.get_dict()

    add_args['k_caching'] = args.k_caching
    add_args['change_stride'] = args.change_stride
    if args.no_outer_loop_first:
        add_args['outside_first'] = False

    if args.cache:
        use_cache(args.program)

    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    if args.program in programs:
        program_name = programs[args.program]
    else:
        program_name = args.program

    if args.sdfg_file is not None:
        sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    else:
        if args.verbose_name:
            reset_graph_files(args.verbose_name)
            add_args['verbose_name'] = args.verbose_name
        sdfg = get_sdfg(fsource, program_name)
        if args.use_dace_auto_opt:
            add_args['use_my_auto_opt'] = False
        sdfg = optimize_sdfg(sdfg, device_map[args.device], **add_args)

    sdfg.instrument = dace.InstrumentationType.Timer
    arguments_dace = generate_arguments_fortran(args.program, f"{program_name}_routine", np.random.default_rng(42), params)
    arguments_original = copy.deepcopy(arguments_dace)
    print(sdfg.build_folder)

    if args.compare_to_fortran:
        print("Compare to fortran")
        arguments_fortran = copy.deepcopy(arguments_dace)
        routine_name = f'{program_name}_routine'
        ffunc = get_fortran(fsource, program_name, routine_name)
        ffunc(**{k.lower(): v for k, v in arguments_fortran.items()})
        if compare_output_all(arguments_fortran, arguments_original, print_if_differ=False):
            print("WARNING: Fortran arguments did not change at all")

    if args.device == 'GPU':
        from utils.gpu_general import copy_to_device
        arguments_dace = copy_to_device(arguments_dace)

    sdfg.save('/tmp/graph.sdfg')
    sdfg = dace.sdfg.sdfg.SDFG.from_file('/tmp/graph.sdfg')
    csdfg = sdfg.compile()
    print(f"Arguments: {list(arguments_dace.keys())}")
    csdfg(**{k.upper(): v for k, v in arguments_dace.items()})
    if compare_output_all(arguments_dace, arguments_original, print_if_differ=False):
        print("WARNING: DaCe arguments did not change at all")

    if args.compare_to_fortran:
        result = compare_output_all(arguments_fortran, arguments_dace, name_a='Fortran', name_b='DaCe')
        if result:
            print("SUCCESS")


if __name__ == '__main__':
    main()
