import os
from typing import List
import logging
import dace
from dace.sdfg.sdfg import SDFG

from utils.paths import get_basic_sdfg_dir
from utils.general import get_programs_data, read_source, get_sdfg
from utils.execute_date import RunConfig
from execute.parameters import ParametersProvider
from execute.my_auto_opt import auto_optimize_phase_1, auto_optimize_phase_2

logger = logging.getLogger("run2")


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
        name += "specialised"
    else:
        name += "unspecialised"
    if run_config.outside_loop_first:
        name += "outsidefirst"
    else:
        name =+ "nooutsidefirst"
    if len(params_to_ignore) > 0:
        name += '_'.join(params_to_ignore)
    return


def generate_basic_sdfg(
        program: str,
        params: ParametersProvider,
        run_config: RunConfig,
        params_to_ignore: List[str] = []) -> SDFG:
    """
    Generates basic SDFG of the given program.

    :param program: The name of the program
    :type program: str
    :param params: The parameters used
    :type params: ParametersProvider
    :param run_config: The run config used. Does not need all information from it
    :type run_config: RunConfig
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
    params_dict = params
    params_dict = params.get_dict()
    for name in params_to_ignore:
        del params_dict[name]
    if run_config.specialise_symbols:
        add_args['symbols'] = params_dict
    add_args['outside_first'] = run_config.outside_loop_first
    sdfg = auto_optimize_phase_1(sdfg, run_config.device, use_my_auto_opt=not run_config.use_dace_auto_opt, **add_args)
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
    filename = "program_" + get_name_string_of_config(run_config, params_to_ignore) + ".sdfg"
    return os.path.join(get_basic_sdfg_dir(), filename)


def get_basic_sdfg(program: str, run_config: RunConfig, params_to_ignore: List[str] = []) -> SDFG:
    """
    Get the basic sdfg. Will load it if available or generated it if not

    :param program: The name of the program
    :type program: str
    :param run_config: The run_config
    :type run_config: RunConfig
    :param params_to_ignore: Parameters not specialised, defaults to []
    :type params_to_ignore: List[str], optional
    :return: The basic sdfg
    :rtype: SDFG
    """
    path = get_path_of_basic_sdfg(program, run_config, params_to_ignore)
    if os.path.exists(path):
        logger.info(f"Load basic SDFG from {path}")
        return SDFG.from_file(path)
    else:
        logger.info("Generate basic SDFG")
        sdfg = generate_basic_sdfg(program, run_config, params_to_ignore)
        sdfg.save(path)
        return sdfg


def get_optimised_sdfg(
        program: str,
        run_config: RunConfig,
        params: ParametersProvider,
        params_to_ignore: List[str] = []) -> SDFG:
    basic_sdfg = get_basic_sdfg(program, run_config, params_to_ignore)

    add_args = {}
    if run_config.specialise_symbols:
        add_args['symbols'] = params.get_dict()
    add_args['k_caching'] = run_config.k_caching
    add_args['change_stride'] = run_config.change_stride
    add_args['outside_first'] = run_config.outside_loop_first
    add_args['move_assignments_outside'] = run_config.move_assignment_outside
    sdfg = auto_optimize_phase_2(
        basic_sdfg,
        run_config.device,
        use_my_auto_opt=not run_config.use_dace_auto_opt,
        **add_args)
    sdfg.instrument = dace.InstrumentationType.Timer
    return sdfg
