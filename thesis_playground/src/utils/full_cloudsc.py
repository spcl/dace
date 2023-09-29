from subprocess import run
import os
import logging
from typing import Optional, List, Dict, Union
import pandas as pd
from numbers import Number
import dace
from dace.sdfg import SDFG
from dace import nodes

from utils.paths import get_dacecache, get_full_cloudsc_log_dir, get_full_cloudsc_results_dir
from utils.general import enable_debug_flags, remove_build_folder
from utils.run_config import RunConfig


logger = logging.getLogger(__name__)


opt_levels = {
    "baseline": {
        "run_config": RunConfig(k_caching=False, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "baseline"
        },
    "k-caching": {
        "run_config": RunConfig(k_caching=True, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "k_caching"
        },
    "change-strides": {
        "run_config": RunConfig(k_caching=False, change_stride=True, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "change_strides"
        },
    "all": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt"
        },
    "all-custom": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt_custom"
        }
}


def get_program_name(version: int) -> str:
    if version == 3:
        program = 'cloudscexp3'
    elif version == 4:
        program = 'cloudscexp4'
    else:
        program = 'cloudscexp2'
    return program


def get_build_folder_name(version: int) -> str:
    if version == 3:
        program = 'CLOUDSCOUTER3'
    elif version == 4:
        program = 'CLOUDSCOUTER4'
    else:
        program = 'CLOUDSCOUTER2'
    return program


def add_synchronize(dacecache_folder: str):
    """
    Adds a cudaDeviceSynchronize and cudaStreamSynchronize before each time measurement start or end. A time measurement
    is a std::chrono::high_resoltion_clock::now() call. Only does this inside the cpu file.

    :param dacecache_folder: The folde in .dacecache where the 'src' folder with the source files lies.
    :type dacecache_folder: str
    """
    src_dir = os.path.join(get_dacecache(), dacecache_folder, 'src', 'cpu')
    src_file = os.path.join(src_dir, os.listdir(src_dir)[0])
    if len(os.listdir(src_dir)) > 1:
        logger.warning(f"More than one files in {src_dir}")
    lines = run(['grep', '-rn', 'std::chrono::high_resolution_clock::now', src_file], capture_output=True).stdout.decode('UTF-8')
    logger.debug("Add synchronizes to %s", src_file)
    line_numbers_to_insert = []

    for line in lines.split('\n'):
        if len(line) > 0:
            if line.split()[2].startswith('__dace_t'):
                line_numbers_to_insert.append(int(line.split(':')[0])-1)

    logger.debug("Insert synchronize into line numbers: %s", line_numbers_to_insert)
    with open(src_file, 'r') as f:
        contents = f.readlines()
        for offset, line_number in enumerate(line_numbers_to_insert):
            contents.insert(line_number+offset,
                    "DACE_GPU_CHECK(cudaDeviceSynchronize());DACE_GPU_CHECK(cudaStreamSynchronize(__state->gpu_context->streams[0]));\n")

    with open(src_file, 'w') as f:
        contents = "".join(contents)
        f.write(contents)


def instrument_sdfg(sdfg: SDFG, opt_level: str, device: str):
    """
    Instruments the given SDFG. Uses the opt level to decide on what to instrument.

    :param sdfg: The SDFG to instrument. Does it in-place
    :type sdfg: SDFG
    :param opt_level: The opt-level, used to decide what to instrument
    :type opt_level: str
    :param device: The device for which the SDFG was generated.
    :type device: str
    """
    if opt_level in ['k-caching', 'baseline']:
        if device == 'CPU':
            cloudsc_state = sdfg.find_state('stateCLOUDSC')
        elif device == 'GPU':
            cloudsc_state = sdfg.find_state('CLOUDSCOUTER4_copyin')
        nblocks_maps = [n for n in cloudsc_state.nodes() if isinstance(n, nodes.MapEntry) and n.label ==
                        'stateCLOUDSC_map']
    if opt_level in ['all', 'change-strides']:
        changed_strides_state = sdfg.find_state('with_changed_strides')
        nsdfg_changed_strides = [n for n in changed_strides_state.nodes() if isinstance(n, nodes.NestedSDFG)][0]
        nblocks_maps = [n for n in nsdfg_changed_strides.sdfg.find_state('CLOUDSCOUTER4_copyin').nodes()
                        if isinstance(n, nodes.MapEntry) and n.label == 'stateCLOUDSC_map']
        sdfg.find_state('transform_data').instrument = dace.InstrumentationType.Timer
        sdfg.find_state('transform_data_back').instrument = dace.InstrumentationType.Timer

    if len(nblocks_maps) > 1:
        logger.warning("There is more than one map to instrument: %s", nblocks_maps)
    nblocks_maps[0].instrument = dace.InstrumentationType.Timer


def get_sdfg(opt_level: str, device: str, version: int) -> SDFG:
    """
    Loads the generated SDFG from memory

    :param opt_level: The optimisation level
    :type opt_level: str
    :param device: The device for which the SDFG was generated
    :type device: str
    :param version: The version used
    :type version: int
    :return: The loaded SDFG
    :rtype: SDFG
    """
    program = get_program_name(version)
    verbose_name = f"{program}_{opt_levels[opt_level]['name']}"
    sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}_{device.lower()}.sdfg")
    return dace.sdfg.sdfg.SDFG.from_file(sdfg_file)


def compile_sdfg(sdfg: SDFG,
                 nblocks: int,
                 version: int,
                 opt_level: str,
                 device: str,
                 instrument: bool = True,
                 debug: bool = False,
                 build_dir: Optional[str] = None):
    """
    Compile the given SDFG

    :param sdfg: The SDFG to compile
    :type sdfg: SDFG
    :param nblocks: The number of blocks to compile it for
    :type nblocks: int
    :param version: The version used
    :type version: int
    :param opt_level: The optimisation level used
    :type opt_level: str
    :param device: The device to SDFG is generated for
    :type device: str
    :param instrument: True if to instrument the SDFG, defaults to True
    :type instrument: bool, optional
    :param debug: True if to build with debug flags, defaults to False
    :type debug: bool, optional
    :param build_dir: The directory to generate the build folder in. If none will take the default in .dacecache,
                      defaults to None
    :type build_dir: Optional[str], optional
    """
    remove_build_folder(dacecache_folder=get_build_folder_name(version))
    program = get_program_name(version)
    for nsdfg in sdfg.sdfg_list:
        nsdfg.add_constant('NBLOCKS', nblocks)

    if debug:
        logger.info("Enable Debug Flags")
        enable_debug_flags()
    if build_dir is not None:
        sdfg.build_folder = build_dir
    else:
        sdfg.build_folder = os.path.abspath(sdfg.build_folder)
    logger.info("Build into %s", sdfg.build_folder)

    if instrument:
        instrument_sdfg(sdfg, opt_level, device)
    sdfg.validate()
    sdfg.compile()
    if instrument and device == 'GPU':
        add_synchronize(get_build_folder_name(version))
    signature_file = os.path.join(get_full_cloudsc_log_dir(), f"signature_dace_{program}.txt")
    logger.info("Write signature file into %s", signature_file)
    with open(signature_file, 'w') as file:
        file.write(sdfg.signature())


def get_experiment_list_df() -> pd.DataFrame:
    experiments_file = os.path.join(get_full_cloudsc_results_dir(), 'experiments.csv')
    if not os.path.exists(experiments_file):
        return pd.DataFrame()
    else:
        return pd.read_csv(experiments_file, index_col=['experiment id'])


def save_experiment_list_df(df: pd.DataFrame):
    experiments_file = os.path.join(get_full_cloudsc_results_dir(), 'experiments.csv')
    df.to_csv(experiments_file)


def read_reports(sdfg: SDFG) -> List[Dict[str, Number]]:
    reports = sdfg.get_instrumentation_reports()
    data = []
    for index, report in enumerate(reports):
        for entry in report.durations.values():
            for key in entry:
                data.append({'scope': key, 'runtime': list(entry[key].values())[0][0], 'run': index})
    return data


def run_cloudsc_cuda(
        executable_name: str,
        data_name: str,
        size: int,
        repetitions: int) -> List[Dict[str, Union[str, Number]]]:
    """
    Runs the cloudsc CUDA version and returns runtime without data movement

    :param executable_name: Name of the executable
    :type executable_name: str
    :param data_name: Name to give in the 'opt level' entry in the returned data
    :type data_name: str
    :param size: Size to pass to the executable
    :type size: int
    :param repetitions: Number of repetitions to run
    :type repetitions: int
    :return: List with one dictionary per run listing runtime and other metadata
    :rtype: List[Dict[str, Union[str, Number]]]
    """
    data = []
    logger.debug('Run %s using %s for size %s repeating it %i time', data_name, executable_name, size, repetitions)
    for i in range(repetitions):
        cloudsc_output = run([f"./bin/{executable_name}", '1', str(size), '128'],
                             cwd='/users/msamuel/dwarf-p-cloudsc-original/build_cuda',
                             capture_output=True)
        if cloudsc_output.returncode == 0:
            for line in cloudsc_output.stdout.decode('UTF-8').split('\n'):
                if 'core' in line:
                    data.append({
                        'scope': 'Map stateCLOUDSC_map',
                        'opt level': data_name,
                        'nblocks': size,
                        'runtime': line.split()[7],
                        'run': i
                        })
                    logger.debug('Results line: %s', line)
        else:
            logger.warning('Running cuda cloudsc failed')
            logger.warning('stdout: %s', cloudsc_output.stdout.decode('UTF-8'))
            logger.warning('stderr: %s', cloudsc_output.stderr.decode('UTF-8'))
    return data


