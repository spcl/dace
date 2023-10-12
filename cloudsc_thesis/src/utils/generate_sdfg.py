import os
from typing import List, Optional
import logging
import dace
from dace.sdfg import nodes, SDFG
from dace.dtypes import ScheduleType

from utils.general import replace_symbols_by_values, reset_graph_files
from utils.paths import get_basic_sdfg_dir
from utils.general import get_programs_data, read_source, get_sdfg, save_graph
from utils.run_config import RunConfig
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
    if program in programs:
        program_name = programs[program]
    else:
        program_name = program

    verbose_name = f"basic_{program}"
    reset_graph_files(verbose_name)

    logger.debug(f"program name: {program_name}")
    fsource = read_source(program)
    sdfg = get_sdfg(fsource, program_name)
    add_args = {}
    params_dict = params.get_dict()
    for name in params_to_ignore:
        del params_dict[name]
    if run_config.specialise_symbols:
        add_args['symbols'] = params_dict
    add_args['outside_first'] = run_config.outside_loop_first
    logger.debug(f"Optimise SDFG for phase 1 (no device) ignoring {params_to_ignore}")
    save_graph(sdfg, verbose_name, "before_replace_symbols_by_values")
    replace_symbols_by_values(sdfg, {
        'NPROMA': str(params['NPROMA']),
        'NCLV': str(params['NCLV']),
        'NCLDQI': str(params['NCLDQI']),
        'NCLDQR': str(params['NCLDQR']),
        'NCLDQL': str(params['NCLDQL']),
        'NCLDQS': str(params['NCLDQS']),
        'NCLDQV': str(params['NCLDQV'])})
    save_graph(sdfg, verbose_name, "after_replace_symbols_by_values")
    auto_optimize_phase_1(sdfg, program=verbose_name, **add_args)
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


def check_sdfg_symbols(sdfg: SDFG, params: ParametersProvider) -> bool:
    """
    Checks that all constants set inside the given SDFG have the same value as the params given

    :param sdfg: The SDFG to check
    :type sdfg: SDFG
    :param params: The parameters to check agains
    :type params: ParametersProvider
    :return: True if symbol/constants value match, False otherwise
    :rtype: bool
    """
    for symbol, value in sdfg.constants.items():
        if params[symbol] != value:
            logger.debug("Symobl %s has different value: %s (SDFG) vs %s (parameters)", symbol, value, params[symbol])
            return False
    return True


def get_basic_sdfg(
        program: str,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = []) -> SDFG:
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
        sdfg = SDFG.from_file(path)
        if check_sdfg_symbols(sdfg, params):
            return sdfg
        else:
            logger.info("Symbols of stored basic SDFG don't match. Generate new basic SDFG")
    logger.info("Generate basic SDFG")
    sdfg = generate_basic_sdfg(program, run_config, params, params_to_ignore)
    sdfg.save(path)
    return sdfg


def remove_basic_sdfg(program: str, run_config: RunConfig, params_to_ignore: List[str] = []):
    """
    Removes the basic sdfg if already computed and stored. Needs to be done if the source code changes.

    :param program: The name of the program
    :type program: str
    :param run_config: The run_config
    :type run_config: RunConfig
    :param params_to_ignore: Parameters not specialised, defaults to []
    :type params_to_ignore: List[str], optional
    """
    path = get_path_of_basic_sdfg(program, run_config, params_to_ignore)
    if os.path.exists(path):
        os.remove(path)


def optimise_basic_sdfg(
        sdfg: SDFG,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = [],
        instrument: bool = True,
        verbose_name: Optional[str] = None,
        storage_on_gpu: bool = True
        ):
    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "basic_sdfg")

    add_args = {}
    symbols = params.get_dict()
    for p in params_to_ignore:
        del symbols[p]
    if run_config.specialise_symbols:
        add_args['symbols'] = symbols
    add_args['k_caching'] = run_config.k_caching
    add_args['move_assignments_outside'] = run_config.move_assignment_outside
    add_args['program'] = verbose_name
    add_args['storage_on_gpu'] = storage_on_gpu
    add_args['full_cloudsc_fixes'] = run_config.full_cloudsc_fixes
    logger.debug("Continue optimisation after getting basic SDFG")
    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "before_phase_2")

    auto_optimize_phase_2(
        sdfg,
        run_config.device,
        **add_args)

    if run_config.change_stride:
        schedule = ScheduleType.GPU_Device if run_config.device == dace.DeviceType.GPU else ScheduleType.Default
        if not storage_on_gpu:
            schedule = ScheduleType.Default
        logger.info("Change strides using schedule %s", schedule)
        sdfg = change_strides(sdfg, ('NBLOCKS', ), schedule)
        if verbose_name is not None:
            save_graph(sdfg, verbose_name, "after_change_stride")

    if run_config.device == dace.DeviceType.GPU:
        logger.info("Set gpu block size to (32, 1, 1)")
        for state in sdfg.states():
            for node, state in state.all_nodes_recursive():
                if isinstance(node, nodes.MapEntry):
                    logger.debug(f"Set block size for {node}")
                    node.map.gpu_block_size = (32, 1, 1)
    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "after_set_gpu_block_size")

    if instrument:
        logger.debug("Instrument SDFG")
        sdfg.instrument = dace.InstrumentationType.Timer
    return sdfg


def get_optimised_sdfg(
        program: str,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = [],
        instrument: bool = True,
        verbose_name: Optional[str] = None,
        storage_on_gpu: bool = True
        ) -> SDFG:
    logger.debug(f"SDFG for {program} using {run_config} and ignore {params_to_ignore}")
    sdfg = get_basic_sdfg(program, run_config, params, params_to_ignore)
    return optimise_basic_sdfg(sdfg, run_config, params, params_to_ignore, instrument, verbose_name, storage_on_gpu)
