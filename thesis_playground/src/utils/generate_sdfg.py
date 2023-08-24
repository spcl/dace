import os
from typing import List
import logging
import dace
from dace.sdfg import nodes, SDFG
from dace.dtypes import ScheduleType

from utils.general import replace_symbols_by_values
from utils.paths import get_basic_sdfg_dir
from utils.general import get_programs_data, read_source, get_sdfg
from utils.execute_dace import RunConfig
from execute.parameters import ParametersProvider
from execute.my_auto_opt import auto_optimize_phase_1, auto_optimize_phase_2, change_strides

logger = logging.getLogger(__name__)


def get_name_string_of_config(run_config: RunConfig, params_to_ignore: List[str] = []) -> str:
    """
    Returns string incooperating the necessary run_config and params_to_ignore information which is used in creating a
    basic SDFG.

    :param run_config: The run config
    :type run_config: RunConfig
    :param params_to_ignore: List of parameters ignored, defaults to []
    :type params_to_ignore: List[str], optional
    :return: Name/description string
    :rtype: str
    """
    name = ""
    if run_config.specialise_symbols:
        name += "specialised_"
    else:
        name += "unspecialised_"
    if run_config.outside_loop_first:
        name += "outsidefirst_"
    else:
        name += "nooutsidefirst_"
    if len(params_to_ignore) > 0:
        name += '_'.join(params_to_ignore)
    return name


def generate_basic_sdfg(
        program: str,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = []) -> SDFG:
    """
    Generates basic SDFG of the given program.

    :param program: The name of the program
    :type program: str
    :param run_config: The run config used. Does not need all information from it
    :type run_config: RunConfig
    :param params: The parameters used
    :type params: ParametersProvider
    :param params_to_ignore: Any parameters in params which should not be specialised, defaults to []
    :type params_to_ignore: List[str], optional
    :return: The basic SDFG
    :rtype: SDFG
    """
    programs = get_programs_data()['programs']
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name)
    add_args = {}
    params_dict = params.get_dict()
    for name in params_to_ignore:
        del params_dict[name]
    if run_config.specialise_symbols:
        add_args['symbols'] = params_dict
    add_args['outside_first'] = run_config.outside_loop_first
    logger.debug(f"Optimise SDFG for phase 1 (no device) ignoring {params_to_ignore}")
    replace_symbols_by_values(sdfg, {
        'NCLDTOP': '15',
        'NCLV': '10',
        'NCLDQI': '3',
        'NCLDQL': '4',
        'NCLDQS': '6',
        'NCLDQV': '7'})
    auto_optimize_phase_1(sdfg, **add_args)
    return sdfg


def get_path_of_basic_sdfg(program: str, run_config: RunConfig, params_to_ignore: List[str] = []) -> str:
    """
    Get path to where the basic sdfg of the program and run_config is saved

    :param program: The name of the program
    :type program: str
    :param run_config: The run_config
    :type run_config: RunConfig
    :param params_to_ignore: Parameters not specialised, defaults to []
    :type params_to_ignore: List[str], optional
    :return: Path to the basic SDFG
    :rtype: str
    """
    filename = f"{program}_" + get_name_string_of_config(run_config, params_to_ignore) + ".sdfg"
    return os.path.join(get_basic_sdfg_dir(), filename)


def get_basic_sdfg(program: str, run_config: RunConfig, params: ParametersProvider, params_to_ignore: List[str] = []) -> SDFG:
    """
    Get the basic sdfg. Will load it if available or generated it if not

    :param program: The name of the program
    :type program: str
    :param run_config: The run_config
    :type run_config: RunConfig
    :param params: The parameters used
    :type params: ParametersProvider
    :param params_to_ignore: Parameters not specialised, defaults to []
    :type params_to_ignore: List[str], optional
    :return: The basic sdfg
    :rtype: SDFG
    """
    path = get_path_of_basic_sdfg(program, run_config, params_to_ignore)
    logger.debug(f"Check fo basic SDFG in {path}")
    if os.path.exists(path):
        logger.info(f"Load basic SDFG from {path}")
        return SDFG.from_file(path)
    else:
        logger.info("Generate basic SDFG")
        sdfg = generate_basic_sdfg(program, run_config, params, params_to_ignore)
        sdfg.save(path)
        return sdfg


def get_optimised_sdfg(
        program: str,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = []) -> SDFG:
    logger.debug(f"SDFG for {program} using {run_config} and ignore {params_to_ignore}")
    basic_sdfg = get_basic_sdfg(program, run_config, params, params_to_ignore)

    add_args = {}
    if run_config.specialise_symbols:
        add_args['symbols'] = params.get_dict()
    add_args['k_caching'] = run_config.k_caching
    add_args['outside_first'] = run_config.outside_loop_first
    add_args['move_assignments_outside'] = run_config.move_assignment_outside
    logger.debug("Continue optimisation after getting basic SDFG")
    sdfg = auto_optimize_phase_2(
        basic_sdfg,
        run_config.device,
        **add_args)

    if run_config.change_stride:
        schedule = ScheduleType.GPU_Device if run_config.device == dace.DeviceType.GPU else ScheduleType.Default
        logger.info("Change strides")
        sdfg = change_strides(sdfg, ('NBLOCKS', ), params.get_dict(), schedule)

    if run_config.device == dace.DeviceType.GPU:
        logger.info("Set gpu block size to (32, 1, 1)")
        for state in sdfg.states():
            for node, state in state.all_nodes_recursive():
                if isinstance(node, nodes.MapEntry):
                    logger.debug(f"Set block size for {node}")
                    node.map.gpu_block_size = (32, 1, 1)

    logger.debug("Instrument SDFG")
    sdfg.instrument = dace.InstrumentationType.Timer
    return sdfg
