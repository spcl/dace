import os
import numpy as np
from numpy import f2py
from numbers import Number
from typing import Dict, Union, List
from importlib import import_module
import sys
import tempfile
import json
from tabulate import tabulate
from glob import glob
from datetime import datetime

import dace
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils, SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion
from data import get_program_parameters_data, get_testing_parameters_data

from measurement_data import ProgramMeasurement, MeasurementRun


# Copied from tests/fortran/cloudsc.py as well as the functions/dicts below
def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
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


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)

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
def get_inputs(program: str, rng: np.random.Generator, testing_dataset: bool = False) \
        -> Dict[str, Union[Number, np.ndarray]]:
    """
    Returns dict with the input data for the requested program

    :param program: The program for which to get the input data
    :type program: str
    :param rng: The random number generator to use
    :type rng: np.random.Generator
    :param testing_dataset: Set to true if the dataset used for testing should be used, defaults to False
    :type testing_dataset: bool, optional
    :return: Dictionary with the input data
    :rtype: Dict[str, Union[Number, np.ndarray]]
    """
    params_data = get_testing_parameters_data() if testing_dataset else get_program_parameters_data(program)
    programs_data = get_programs_data()
    inp_data = dict()
    for p in programs_data['program_parameters'][program]:
        inp_data[p] = params_data['parameters'][p]
    for inp in programs_data['program_inputs'][program]:
        shape = params_data['data'][inp]
        if shape == (0, ):  # Scalar
            inp_data[inp] = rng.random()
        else:
            inp_data[inp] = np.asfortranarray(rng.random(shape))
    return inp_data


# Copied from tests/fortran/cloudsc.py
# TODO: Init to 0?
def get_outputs(program: str, rng: np.random.Generator, testing_dataset: bool = False) \
        -> Dict[str, Union[Number, np.ndarray]]:
    """
    Returns dict with the output data for the requested program

    :param program: The program for which to get the output data
    :type program: str
    :param rng: The random number generator to use
    :type rng: np.random.Generator
    :param testing_dataset: Set to true if the dataset used for testing should be used, defaults to False
    :type testing_dataset: bool, optional
    :return: Dictionary with the output data
    :rtype: Dict[str, Union[Number, np.ndarray]]
    """
    params_data = get_testing_parameters_data() if testing_dataset else get_program_parameters_data(program)
    programs_data = get_programs_data()
    out_data = dict()
    for out in programs_data['program_outputs'][program]:
        shape = params_data['data'][out]
        if shape == (0, ):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(rng.random(shape))
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
    programs_file = os.path.join(os.path.dirname(__file__), 'programs.json')
    with open(programs_file) as file:
        programs_data = json.load(file)

    for program in not_working:
        for dict_key in programs_data:
            del programs_data[dict_key][program]
    return programs_data


def print_results_v2(run_data: MeasurementRun):
    headers = ["program", "measurement", "min", "max", "avg", "median"]
    flat_data = []
    for program_measurement in run_data.data:
        for measurement in program_measurement.measurements.values():
            name = measurement.name
            if measurement.kernel_name is not None:
                name += f" of {measurement.kernel_name}"
            name += f" [{measurement.unit}] (#={measurement.amount()})"
            flat_data.append([program_measurement.program, name, measurement.min(), measurement.max(),
                              measurement.average(), measurement.median()])

    print(tabulate(flat_data, headers=headers))


def get_results_dir() -> str:
    """
    Returns path to the directory where the results are stored

    :return: Path to the directory
    :rtype: str
    """
    return os.path.join(os.path.dirname(__file__), 'results')


counter = 0
graphs_dir = os.path.join(os.path.dirname(__file__), 'sdfg_graphs')


def save_graph(sdfg: SDFG, program: str, name: str, prefix=""):
    global counter
    filename = os.path.join(graphs_dir, f"{prefix}{program}_{counter}_{name}.sdfg")
    sdfg.save(filename)
    print(f"Saved graph to {filename}")
    counter = counter + 1


def reset_graph_files(program: str):
    for file in glob(os.path.join(graphs_dir, f"*{program}_*.sdfg")):
        os.remove(file)


def print_with_time(text: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
