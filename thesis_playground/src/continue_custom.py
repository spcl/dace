from argparse import ArgumentParser
import dace
from dace.transformation.auto.auto_optimize import greedy_fuse

from execute.parameters import ParametersProvider
from utils.general import save_graph
from utils.log import setup_logging
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import reset_graph_files


def main():
    parser = ArgumentParser()
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--verbose-name',
                        type=str,
                        default=None,
                        help='Foldername under which intermediate SDFGs should be stored, uses the program name by '
                        'default')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--log-level', default='DEBUG', help='Log level for console, defaults to DEBUG')
    parser.add_argument('--log-file', default=None)
    add_cloudsc_size_arguments(parser)
    args = parser.parse_args()

    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    add_args = {}
    if args.log_file is not None:
        add_args['full_logfile'] = args.log_file
    setup_logging(level=args.log_level.upper(), **add_args)

    if args.verbose_name is not None:
        verbose_name = args.verbose_name
        reset_graph_files(verbose_name)

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    params = ParametersProvider('cloudscexp4')
    symbols = params.get_dict()
    validate_all = True
    greedy_fuse(sdfg, device=device, validate_all=validate_all, k_caching_args={
        'max_difference_start': symbols['NCLDTOP']+1,
        'max_difference_end': 1,
        'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
                                          symbols['KLEV'])
    })

    save_graph(sdfg, verbose_name, "after_greedy_fuse")


if __name__ == '__main__':
    main()
