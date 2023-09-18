from argparse import ArgumentParser
import os
import logging
import dace
from dace.sdfg import SDFG
from dace.dtypes import ScheduleType
from dace.transformation.auto.auto_optimize import greedy_fuse
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation import helpers as xfh
from dace.sdfg.nodes import NestedSDFG, MapEntry
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import MoveLoopIntoMap, LoopUnroll, StateFusion
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate.loop_detection import find_for_loop

from execute.parameters import ParametersProvider
from execute.my_auto_opt import auto_optimize_phase_2, make_outermost_map, apply_subgraph_fusion, is_map_over_symbol, \
                                k_caching_prototype_v1_fuse, change_strides
from utils.general import save_graph, replace_symbols_by_values
from utils.log import setup_logging
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import reset_graph_files
from utils.paths import get_full_cloudsc_log_dir

logger = logging.getLogger(__name__)

def continue_full_cloudsc_fuse(sdfg: SDFG, device: dace.DeviceType, verbose_name: str):
    validate = True
    validate_all = True
    params = ParametersProvider('cloudscexp4')
    symbols = params.get_dict()
    full_cloudsc_fixes = True
    program = verbose_name
    k_caching_prototype_v1_fuse(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)
    save_graph(sdfg, verbose_name, "after_fuse")
    auto_optimize_phase_2(sdfg, device, program, validate, validate_all, symbols, k_caching=False,
                          move_assignments_outside=True, storage_on_gpu=False, full_cloudsc_fixes=True, skip_fusing=True)

    schedule = ScheduleType.GPU_Device if device == dace.DeviceType.GPU else ScheduleType.Default
    schedule = ScheduleType.Default
    logger.info("Change strides using schedule %s", schedule)
    sdfg = change_strides(sdfg, ('NBLOCKS', ), schedule)
    logger.info("Set gpu block size to (32, 1, 1)")
    for state in sdfg.states():
        for node, state in state.all_nodes_recursive():
            if isinstance(node, MapEntry):
                logger.debug(f"Set block size for {node}")
                node.map.gpu_block_size = (32, 1, 1)
    sdfg.save(os.path.join(get_full_cloudsc_log_dir(), "cloudscexp4_all_opt_custom.sdfg"))


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
    sdfg.validate()
    params = ParametersProvider('cloudscexp4')
    symbols = params.get_dict()
    validate_all = True

    continue_full_cloudsc_fuse(sdfg, device, verbose_name)

    # sdfg.apply_transformations_repeated([StateFusion])
    # sdfg.apply_gpu_transformations()
    # save_graph(sdfg, verbose_name, "after_gpu_transformations")
    # apply_subgraph_fusion(sdfg, {
    #     'max_difference_start': symbols['NCLDTOP']+1,
    #     'max_difference_end': 1,
    #     'disjoint_subsets': False,
    #     'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
    #                                       symbols['KLEV']),
    #     'fixed_new_shapes': {'ZQXN2D': [symbols['KLON'], 1, symbols['NCLV']]}
    #     },
    #         symbols, verbose_name)

    # continue_search = True
    # while continue_search:
    #     xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapToForLoop], permissive=True)]
    #     logger.debug("Found %i many possible transformations to transform map back to for-loop", len(xforms))
    #     continue_search = False
    #     for xf in xforms:
    #         # expect that maps only have one dimension, as we did the MapExpansion transformation before
    #         xf_sdfg = sdfg.sdfg_list[xf.sdfg_id]
    #         xf_state = xf_sdfg.find_state(xf.state_id)
    #         if is_map_over_symbol(xf.map_entry.map.range, symbols, 'KLEV', 1):
    #             continue_search = True
    #             logger.debug("Found the correct map. Apply it to state %s and sdfg %s", xf_state.name, xf_sdfg.label)
    #             xf.apply(xf_state, xf_sdfg)
    #             if verbose_name is not None:
    #                 save_graph(sdfg, verbose_name, "after_map_to_for_loop")
    #             break
    # sdfg.validate()
    # save_graph(sdfg, verbose_name, "after_map_to_for_loop")


if __name__ == '__main__':
    main()
