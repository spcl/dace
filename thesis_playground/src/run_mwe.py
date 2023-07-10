from argparse import ArgumentParser
import dace
import numpy as np
import copy

from execute.parameters import ParametersProvider
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import get_programs_data, get_sdfg, read_source, optimize_sdfg, generate_arguments_fortran, \
                          get_fortran, compare_output_all


def main():
    parser = ArgumentParser()
    parser.add_argument('program')
    parser.add_argument('--debug', action='store_true', default=False, help="Configure for debug build")
    parser.add_argument('--not-specialise', action='store_true', help='Do not specialise symbols')
    parser.add_argument('--k-caching', action='store_true', default=False, help="use k-caching")
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--compare-to-fortran', default=False, action='store_true')
    add_cloudsc_size_arguments(parser)

    args = parser.parse_args()
    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}

    add_args = {}
    params = ParametersProvider(args.program)
    params.update_from_args(args)
    if not args.not_specialise:
        print(f"Use {params} for specialisation")
        add_args['symbols'] = params.get_dict()

    add_args['k_caching'] = args.k_caching

    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    if args.program in programs:
        program_name = programs[args.program]
    else:
        program_name = args.program
    sdfg = get_sdfg(fsource, program_name)

    optimize_sdfg(sdfg, device_map[args.device], **add_args)
    sdfg.instrument = dace.InstrumentationType.Timer
    arguments_dace = generate_arguments_fortran(program_name, np.random.default_rng(42), params)
    arguments_original = copy.deepcopy(arguments_dace)

    if args.compare_to_fortran:
        arguments_fortran = copy.deepcopy(arguments_dace)
        routine_name = f'{program_name}_routine'
        ffunc = get_fortran(fsource, program_name, routine_name)
        ffunc(**{k.lower(): v for k, v in arguments_fortran.items()})
        if compare_output_all(arguments_fortran, arguments_original, print_if_differ=False):
            print("WARNING: Fortran arguments did not change at all")

    if args.device == 'GPU':
        from utils.gpu_general import copy_to_device
        arguments_dace = copy_to_device(arguments_dace)

    csdfg = sdfg.compile()
    csdfg(**{k.upper(): v for k, v in arguments_dace.items()})
    if compare_output_all(arguments_dace, arguments_original, print_if_differ=False):
        print("WARNING: DaCe arguments did not change at all")

    if args.compare_to_fortran:
        result = compare_output_all(arguments_fortran, arguments_dace)
        if result:
            print("SUCCESS")


if __name__ == '__main__':
    main()
