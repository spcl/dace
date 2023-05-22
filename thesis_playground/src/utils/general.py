import os
import numpy as np
from numpy import f2py
from numbers import Number
from typing import Dict, Union, List, Optional
from importlib import import_module
import sys
import tempfile
import json
from glob import glob
from subprocess import run, PIPE, Popen, check_output
import cupy as cp

import dace
from dace.config import Config
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils, SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion
from dace.transformation.auto.auto_optimize import auto_optimize as dace_auto_optimize

from execute.data import ParametersProvider, get_iteration_ranges, get_data
from utils.paths import get_dacecache, get_verbose_graphs_dir


# Copied from tests/fortran/cloudsc.py as well as the functions/dicts below
def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], 'fortran_programs',
                           f'{filename}.{extension}'),
              'r') as file:
        source = file.read()
    assert source
    return source


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        # Can set verbose to true to get more output when compiling
        f2py.compile(source, modulename=program_name, verbose=False, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name)
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = True) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)
    if not normalize_offsets:
        print("WARNING: Not normalizing offsets")

    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    if normalize_offsets:
        my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    else:
        my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    if normalize_offsets:
        utils.normalize_offsets(sdfg)

    return sdfg


# Copied from tests/fortran/cloudsc.py
def get_inputs(program: str, rng: np.random.Generator, params: ParametersProvider) \
        -> Dict[str, Union[Number, np.ndarray]]:
    """
    Returns dict with the input data for the requested program

    :param program: The program for which to get the input data
    :type program: str
    :param rng: The random number generator to use
    :type rng: np.random.Generator
    :param parameters: The parameters to use.
    :type parameters: ParametersProvider
    :return: Dictionary with the input data
    :rtype: Dict[str, Union[Number, np.ndarray]]
    """
    data = get_data(params)
    programs_data = get_programs_data()
    inp_data = dict()
    for p in programs_data['program_parameters'][program]:
        inp_data[p] = params[p]
    for inp in programs_data['program_inputs'][program]:
        shape = data[inp]
        if shape == (0, ):  # Scalar
            inp_data[inp] = rng.random()
        else:
            if inp in ['LDCUM', 'LDCUM_NF', 'LDCUM_NFS']:
                # Random ints in range [0,1]
                inp_data[inp] = np.asfortranarray(rng.integers(0, 2, shape, dtype=np.int32))
            else:
                inp_data[inp] = np.asfortranarray(rng.random(shape))
    return inp_data


def copy_to_device(host_data: Dict[str, Union[Number, np.ndarray]]) -> Dict[str, cp.ndarray]:
    """
    Take a dictionary with data and converts all numpy arrays to cupy arrays

    :param host_data: The numpy (host) data
    :type host_data: Dict[str, Union[Number, np.ndarray]]
    :return: The converted cupy (device) data
    :rtype: Dict[str, cp.ndarray]
    """
    device_data = {}
    for name, array in host_data.items():
        if isinstance(array, np.ndarray):
            device_data[name] = cp.asarray(array)
        else:
            device_data[name] = array
    return device_data


def copy_to_host(device_data: Dict[str, Union[Number, cp.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Take a dictionary with data and converts all cupy arrays to numpy arrays

    :param host_data: The cupy (device) data
    :type host_data: Dict[str, Union[Number, cp.ndarray]]
    :return: The converted cupy (host) data
    :rtype: Dict[str, np.ndarray]
    """
    host_data = {}
    for name, array in device_data.items():
        if isinstance(array, cp.ndarray):
            host_data[name] = cp.asnumpy(array)
        else:
            host_data[name] = array
    return host_data


# Copied from tests/fortran/cloudsc.py
def get_outputs(program: str, rng: np.random.Generator, params: ParametersProvider) \
        -> Dict[str, Union[Number, np.ndarray]]:
    """
    Returns dict with the output data for the requested program

    :param program: The program for which to get the output data
    :type program: str
    :param rng: The random number generator to use
    :type rng: np.random.Generator
    :param parameters: The parameters to use.
    :type parameters: ParametersProvider
    :return: Dictionary with the output data
    :rtype: Dict[str, Union[Number, np.ndarray]]
    """
    data = get_data(params)
    programs_data = get_programs_data()
    out_data = dict()
    for out in programs_data['program_outputs'][program]:
        shape = data[out]
        if shape == (0, ):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(rng.random(shape))
            # out_data[out] = np.asfortranarray(np.zeros(shape))
    return out_data


def get_programs_data(not_working: List[str] = ['cloudsc_class2_1001', 'mwe_test']) -> Dict[str, Dict]:
    """
    Returns a dictionary which contains several other dicts with information about the programs. These are the
    dictionaries defined in programs.json, removes all program names  which are in not_working


    :param not_working: List of programs to exclude, defaults to ['cloudsc_class2_1001', 'mwe_test']
    :type not_working: List[str], optional
    :return: Dictionary with information about programs
    :rtype: Dict[str, Dict]
    """
    programs_file = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], 'programs.json')
    with open(programs_file) as file:
        programs_data = json.load(file)

    for program in not_working:
        for dict_key in programs_data:
            del programs_data[dict_key][program]
    return programs_data


counter = 0


def save_graph(sdfg: SDFG, program: str, name: str, prefix=""):
    global counter
    if prefix != "":
        prefix = f"{prefix}_"
    if not os.path.exists(get_verbose_graphs_dir()):
        os.mkdir(get_verbose_graphs_dir())
    if not os.path.exists(os.path.join(get_verbose_graphs_dir(), program)):
        os.mkdir(os.path.join(get_verbose_graphs_dir(), program))
    filename = os.path.join(get_verbose_graphs_dir(), program, f"{prefix}{counter}_{name}.sdfg")
    sdfg.save(filename)
    print(f"Saved graph to {filename}")
    counter = counter + 1


def reset_graph_files(program: str):
    for file in glob(os.path.join(get_verbose_graphs_dir(), program, "*.sdfg")):
        os.remove(file)


def use_cache(program: Optional[str] = None, dacecache_folder: Optional[str] = None) -> bool:
    """
    Puts the environment variables to tell dace to use the cached generated coded. Also builts the generated code

    :param program: The name of the program, used for building the code. Can be None if dacecache_folder is given
    instead.
    :type program: Optional[str]
    :param dacecache_folder: Name of the folder of the program in the .dacecache folder. If None will infer from program
    name
    :type dacecache_folder: Optional[str]
    :return: True if building was successful, false otherwise
    :rtype: bool
    """
    programs = get_programs_data()['programs']
    os.environ['DACE_compiler_use_cache'] = '1'
    os.putenv('DACE_compiler_use_cache', '1')
    print("Build it without regenerating the code")
    dacecache_folder = f"{programs[program]}_routine" if dacecache_folder is None else dacecache_folder
    build = run(['make'],
                cwd=os.path.join(get_dacecache(), dacecache_folder, 'build'),
                capture_output=True)
    if build.returncode != 0:
        print("ERROR: Error encountered while building")
        print(build.stderr.decode('UTF-8'))
        return False
    return True


def disable_cache():
    """
    Disables using the cache in the settings and env variables
    """
    os.environ['DACE_compiler_use_cache'] = '0'
    os.putenv('DACE_compiler_use_cache', '0')


def compare_output(output_a: Dict, output_b: Dict, program: str, params: ParametersProvider) -> bool:
    """
    Compares two outputs. Outputs are dictionaries where key is the variable name and value a n-dimensional matrix. They
    are compared based on the ranges given by get_iteration_ranges for the given program

    :param output_a: First output
    :type output_a: Dict
    :param output_b: Second output
    :type output_b: Dict
    :param program: The program name
    :type program: str
    :param testing_dataset: Set to true if the dataset used for testing should be used, defaults to False
    :type testing_dataset: bool, optional
    :return: True if the outputs are equal, false otherwise
    :rtype: bool
    """
    ranges = get_iteration_ranges(params, program)
    same = True
    range_keys = []
    for range in ranges:
        range_keys.extend(range['variables'])
        for key in range['variables']:
            if key not in output_a or key not in output_b:
                print(f"WARNING: {key} given in range for {program} but not found in its outputs: "
                      f"{list(output_a.keys())} and {list(output_b.keys())}")
            selection = []
            for start, stop in zip(range['start'], range['end']):
                if stop is not None:
                    # We have a slice
                    selection.append(slice(start, stop))
                else:
                    # We have a list
                    selection.append(start)
            selection = tuple(selection)
            this_same = np.allclose(
                    output_a[key][selection],
                    output_b[key][selection],
                    atol=10e-5)
            if not this_same:
                print(f"{key} is not the same for range {selection}")
                print_compare_matrix(output_a[key][selection], output_b[key][selection], selection)
            same = same and this_same
    set_range_keys = set(range_keys)
    set_a_keys = set(output_a.keys())
    set_b_keys = set(output_b.keys())
    if set_range_keys != set_a_keys:
        print(f"WARNING: Keys don't match. Range: {set_range_keys}, output_a: {set_a_keys}")
    if set_range_keys != set_b_keys:
        print(f"WARNING: Keys don't match. Range: {set_range_keys}, output_b: {set_b_keys}")

    return same


def compare_output_all(output_a: Dict, output_b: Dict, print_if_differ: bool = True) -> bool:
    same = True
    for key in output_a.keys():
        local_same = np.allclose(output_a[key], output_b[key])
        same = same and local_same
        if not local_same and print_if_differ:
            print(f"Variable {key} differs")
    return same


def print_compare_matrix(output_a: np.ndarray, output_b: np.ndarray, selection):
    if len(selection) == 0:
        diff = np.isclose(output_a, output_b)
        if diff:
            print(f"       {output_a:.3f}", end="   ")
        else:
            print(f"{output_a:.3f}!={output_b:.3f}", end="   ")
            # print(output_a-output_b)
    else:
        for elem_a, elem_b in zip(output_a, output_b):
            print_compare_matrix(elem_a, elem_b, selection[1:])
        print()


def enable_debug_flags():
    print("Configure for debugging")
    Config.set('compiler', 'build_type', value='Debug')
    Config.set('compiler', 'cuda', 'syncdebug', value=True)
    nvcc_args = Config.get('compiler', 'cuda', 'args')
    Config.set('compiler', 'cuda', 'args', value=nvcc_args + ' -g -G')


def optimize_sdfg(sdfg: SDFG, device: dace.DeviceType, use_my_auto_opt: bool = True,
                  verbose_name: Optional[str] = None, symbols: Optional[Dict[str, Number]] = None):
    """
    Optimizes the given SDFG for the given device using auto_optimize. Will use DaCe or my version based on the given
    flag

    :param sdfg: The SDFG to optimize
    :type sdfg: SDFG
    :param device: The device to optimize it on
    :type device: dace.DeviceType
    :param use_my_auto_opt: Flag to control if my custon auto_opt should be used, defaults to True
    :type use_my_auto_opt: bool, optional
    :param verbose_name: Name of the folder to store any intermediate sdfg. Will only do this if is not None, default
    None
    :type verbose_name: Optional[str]
    :param symbols: Dictionary of key, value pairs defining symbols which can be set to const, defaults to None
    :type symbols: Optional[Dict[str, Number]]
    """

    for k, v in sdfg.arrays.items():
        if not v.transient and type(v) == dace.data.Array:
            v.storage = dace.dtypes.StorageType.GPU_Global

    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "before_auto_opt")
    # avoid cyclic dependency
    from execute.my_auto_opt import auto_optimize as my_auto_optimize
    if device == dace.DeviceType.GPU:
        additional_args = {}
        if symbols:
            additional_args['symbols'] = symbols
        if use_my_auto_opt:
            if verbose_name is not None:
                additional_args['program'] = verbose_name
            my_auto_optimize(sdfg, device, **additional_args)
        else:
            dace_auto_optimize(sdfg, device, **additional_args)

    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "after_auto_opt")
    return sdfg


def print_non_zero_percentage(data: Dict[str, Union[Number, Union[np.ndarray, cp.ndarray]]], variable: str):
    """
    Prints the number of total and nonzero values and the percentage of a given variable. Data can have numpy or cupy
    arrays

    :param data: The data where the variable is inside
    :type data: Dict[str, Union[Number, Union[np.ndarray, cp.ndarray]]]
    :param variable: The variable name
    :type variable: str
    """
    variable_data = data[variable]
    if isinstance(variable_data, cp.ndarray):
        variable_data = cp.asnumpy(variable_data)
    num_nnz = np.count_nonzero(variable_data)
    num_tot = np.prod(variable_data.shape)
    print(f"{variable}: {num_nnz}/{num_tot} = {num_nnz/num_tot*100}%")


def convert_to_seconds(value: Number, unit: str) -> float:
    """
    Converts a given value into seconds

    :param value: The value given
    :type value: Number
    :param unit: The unit in which the value is given as a string
    :type unit: str
    :return: The given value in seconds
    :rtype: float
    """
    factors = {
            'second': 1,
            'msecond': 1000,
            'usecond': 1000000
            }
    if unit in factors:
        return float(value) / float(factors[unit])
    else:
        print(f"ERROR: No factor recorded for {unit}")
        return None


def convert_to_bytes(value: Number, unit: str) -> float:
    """
    Converts a given value into bytes

    :param value: The value given
    :type value: Number
    :param unit: The unit in which the value is given as a string
    :type unit: str
    :return: The given value in seconds
    :rtype: float
    """
    factors = {
            'byte': 1,
            'Kbyte': 1e3,
            'Mbyte': 1e6,
            'Gbyte': 1e9,
            'KB': 1e3,
            'MB': 1e6,
            'GB': 1e9,
            }
    if unit in factors:
        return float(value) * float(factors[unit])
    else:
        return None
        print(f"ERROR: No factor recorded for {unit}")


def insert_heap_size_limit(dacecache_folder_name: str, limit: str):
    src_dir = os.path.join(get_dacecache(), dacecache_folder_name, 'src', 'cuda')
    src_file = os.path.join(src_dir, os.listdir(src_dir)[0])
    if len(os.listdir(src_dir)) > 1:
        print(f"WARNING: More than one files in {src_dir}")
    print(f"Adding heap limit of {limit} to {src_file}")

    lines = run(['grep', '-rn', 'cudaLaunchKernel', src_file], capture_output=True).stdout.decode('UTF-8')
    last_line = lines.split('\n')[-2]
    line_number = int(last_line.split(':')[0]) - 1

    set_heap_limit_str = f"size_t required_heap_size = ({limit}) * NBLOCKS * 8 * 2;\n" + \
                         "cudaError_t limit_error = cudaDeviceSetLimit(cudaLimitMallocHeapSize, required_heap_size);\n"

    with open(src_file, 'r') as f:
        contents = f.readlines()
        contents.insert(line_number, set_heap_limit_str)
        # print(contents)

    with open(src_file, 'w') as f:
        contents = "".join(contents)
        f.write(contents)
