from argparse import ArgumentParser
import logging
import dace
from dace.transformation.auto.auto_optimize import greedy_fuse
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation import helpers as xfh
from dace.sdfg.nodes import NestedSDFG
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import MoveLoopIntoMap, LoopUnroll, StateFusion
from dace.transformation.interstate.loop_detection import find_for_loop

from execute.parameters import ParametersProvider
from execute.my_auto_opt import auto_optimize_phase_2, make_outermost_map
from utils.general import save_graph
from utils.log import setup_logging
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import reset_graph_files

logger = logging.getLogger(__name__)


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

    verbose_name = None
    if args.verbose_name is not None:
        verbose_name = args.verbose_name
        reset_graph_files(verbose_name)

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    params = ParametersProvider('cloudscexp4')
    symbols = params.get_dict()
    validate_all = True

    cloudsc_state = sdfg.find_state('stateCLOUDSC')
    cloudsc_nsdfg = [n for n in cloudsc_state.nodes() if isinstance(n, NestedSDFG)][0]

    # make_outermost_map(sdfg, symbols, 'KLEV', allowed_difference=1)
    # save_graph(sdfg, verbose_name, "after_make_klev_outermost")

    # sdfg.apply_transformations_repeated([RefineNestedAccess])
    # if verbose_name:
    #     save_graph(sdfg, verbose_name, "after_refine_nested_access")
    # return 0

    # single_state_body_5 = cloudsc_nsdfg.sdfg.find_state('single_state_body_5')

    # loop_body_nsdfg = [n for n in single_state_body_5.nodes() if isinstance(n, NestedSDFG)][0]
    # print()

    # loop_body_nsdfg.sdfg.apply_transformations_repeated([RefineNestedAccess], states=[
    #     loop_body_nsdfg.sdfg.find_state('single_state_body_20')
    #     ])

    # cloudsc_nsdfg.sdfg.apply_transformations_repeated([RefineNestedAccess], states=[
    #     single_state_body_5
    #     ])

    # This fails, but why is the loop it fails for not converted to a map in the first place?
    # SimplifyPass(validate=True, validate_all=True, skip=['RemoveUnusedSymbols']).apply_pass(sdfg, {})
    # if program is not None:
    #     save_graph(sdfg, program, "after_simplify")

    # for s in sdfg.sdfg_list:
    #     xfh.split_interstate_edges(s)
    # save_graph(sdfg, verbose_name, "after_splitting_interstate_edges")
    # count = sdfg.apply_transformations_repeated([LoopToMap])
    # logger.info("Applied LoopToMap transformation %s times", count)
    # # can_be_applied is never called? Pattern never matches???!?
    # save_graph(sdfg, verbose_name, "after_loop_to_map")

    sdfg.apply_transformations_repeated([StateFusion])
    save_graph(sdfg, verbose_name, "after_state_fusion")

    greedy_fuse(sdfg, device=device, validate_all=validate_all, k_caching_args={
        'max_difference_end': 1, 'max_difference_start': 0, 'disjoint_subsets': False,
                'is_map_sequential': lambda map: False})
    save_graph(sdfg, verbose_name, "after_greedy_fuse")

    for i in range(1):
        greedy_fuse(sdfg, device=device, validate_all=validate_all, k_caching_args={
            'max_difference_start': symbols['NCLDTOP']+1,
            'max_difference_end': 1,
            'disjoint_subsets': False,
            'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
                                              symbols['KLEV'])
        })

        save_graph(sdfg, verbose_name, "after_greedy_fuse")


if __name__ == '__main__':
    main()
