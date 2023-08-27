from argparse import ArgumentParser

import dace
from utils.log import setup_logging
from execute.parameters import ParametersProvider
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import reset_graph_files, optimize_sdfg


def main():
    parser = ArgumentParser()
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('program', type=str, help='Name of the program of the SDFG')
    parser.add_argument('--k-caching', action='store_true', default=False, help="use k-caching")
    parser.add_argument('--change-stride', action='store_true', default=False, help="change stride")
    parser.add_argument('--verbose-name',
                        type=str,
                        default=None,
                        help='Foldername under which intermediate SDFGs should be stored, uses the program name by '
                        'default')
    parser.add_argument('--no-outer-loop-first', action='store_true', default=False, help='Disable outer loops first')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--log-file', default=None, help='Path to logfile with level DEBUG')
    add_cloudsc_size_arguments(parser)
    args = parser.parse_args()

    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    if args.log_file is not None:
        setup_logging(full_logfile=args.log_file)
    else:
        setup_logging()

    verbose_name = args.program
    if args.verbose_name is not None:
        verbose_name = args.verbose_name
    reset_graph_files(verbose_name)

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    add_args = {}
    params = ParametersProvider(args.program)
    params.update_from_args(args)
    print(f"Use {params} for specialisation")
    add_args['symbols'] = params.get_dict()
    add_args['k_caching'] = args.k_caching
    add_args['change_stride'] = args.change_stride
    if args.no_outer_loop_first:
        add_args['outside_first'] = False
    sdfg = optimize_sdfg(sdfg, device, verbose_name=verbose_name, **add_args)


if __name__ == '__main__':
    main()
