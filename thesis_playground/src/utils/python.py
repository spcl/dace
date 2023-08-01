"""
General helper functions to deal with python programs
"""
import inspect
import os
import re
from typing import Dict, Tuple, Union, List
from numbers import Number
import dace
import sympy
import copy
import numpy as np
import pandas as pd

from utils.general import reset_graph_files, optimize_sdfg
from utils.paths import get_playground_results_dir

df_index_cols = ['program', 'NBLOCKS', 'specialised', 'run_number', 'experiment_id']
description_df_path = os.path.join(get_playground_results_dir(), 'python', 'descriptions.csv')

# Symbols to decrement by one for python version due to 1 indexing for fortran
decrement_symbols = ['NCLDQI', 'NCLDQL', 'NCLDQR', 'NCLDQS']


def eval_argument_shape(parameter: inspect.Parameter, symbols: Dict[str, int]) -> Tuple[int]:
    """
    Evaluate the shape of the given parameter/argument for an array

    :param parameter: The parameter/argument
    :type parameter: inspect.Parameter
    :param symbols: The parameters/symbols used to evaulate
    :type symbols: Dict[str, int]
    :return: The shape
    :rtype: Tuple[int]
    """
    shape = list(parameter.annotation.shape)
    for index, dim in enumerate(shape):
        if isinstance(dim, sympy.core.expr.Expr):
            shape[index] = int(dim.evalf(subs=symbols))
    return shape


def gen_arguments(f: dace.frontend.python.parser.DaceProgram,
                  symbols: Dict[str, int]) -> Dict[str, Union[np.ndarray, Number]]:
    """
    Generates the neccessary arguments to call the given function

    :param f: The DaceProgram
    :type f: dace.frontend.python.parser.DaceProgram
    :param symbols: Values for symbols
    :type symbols: Dict[str, int]
    :return: Dict, keys are argument names, values are argument values
    :rtype: Dict[str, Union[nd.array, Number]]
    """
    rng = np.random.default_rng(42)
    arguments = {}
    for parameter in inspect.signature(f.f).parameters.values():
        if isinstance(parameter.annotation, dace.dtypes.typeclass):
            arguments[parameter.name] = rng.random(dtype=parameter.annotation.dtype.as_numpy_dtype())
        elif isinstance(parameter.annotation, dace.data.Array):
            shape = eval_argument_shape(parameter, symbols)
            dtype = parameter.annotation.dtype.as_numpy_dtype()
            if np.issubdtype(dtype, np.integer):
                arguments[parameter.name] = rng.integers(0, 2, shape, dtype=parameter.annotation.dtype.as_numpy_dtype())
            else:
                arguments[parameter.name] = rng.random(shape, dtype=parameter.annotation.dtype.as_numpy_dtype())
    return arguments


def get_size_of_parameters(dace_f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int]) -> int:
    """
    Returns size of all the parameters of the given function in number of bytes.

    :param dace_f: The functions whose parameters we want to look at
    :type dace_f: dace.frontend.python.parser.DaceProgram
    :param symbols: The symbols and their values used in the sizes of the function parameters
    :type symbols: Dict[str, int]
    :return: The size in number of bytes
    :rtype: int
    """
    size = 0
    for name, parameter in inspect.signature(dace_f.f).parameters.items():
        if isinstance(parameter.annotation, dace.dtypes.typeclass):
            size += 1
        elif isinstance(parameter.annotation, dace.data.Array):
            shape = eval_argument_shape(parameter, symbols)
            size += np.prod(shape)
            # print(f"{name:15} ({shape}) adds {np.prod(shape):12,} bytes. New size {size:12,}")
    return int(size * 8)


def get_dacecache_folder(program_name: str) -> str:
    """
    Get path to the dacecache folder given name of program/kernel. Assumes that the program is located inside the
    python_programs module

    :param program_name: The program/kernel name
    :type program_name: str
    :return: The name of the dacecache folder
    :rtype: str
    """
    return f"python_programs_{program_name}_{program_name}"


def get_joined_df(paths: List[str]) -> pd.DataFrame:
    """
    Reads all data from different DataFrames stored in different frames and combines them into one dataframe

    :param paths: List of filenames in the python subdirectory in the playground results directory
    :type paths: List[str]
    :return: The combined DataFrame
    :rtype: pd.DataFrame
    """
    joined_df = pd.DataFrame()
    for file in paths:
        if not file.endswith('.csv'):
            file = f"{file}.csv"
        path = os.path.join(get_playground_results_dir(), 'python', file)
        df = pd.read_csv(path, index_col=df_index_cols)
        return pd.concat([joined_df, df])


def convert_to_plain_python(dace_f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, Number]) -> str:
    """
    Converts a dace python function into code for a plain/vanilla python function

    :param dace_f: The dace function
    :type dace_f: dace.frontend.python.parser.DaceProgram
    :param symbols: Dictionary with symbols used in the dace function
    :type symbols: Dict[str, Number]
    :return: Source code for the function with symbols definition before it
    :rtype: str
    """
    assert inspect.isfunction(dace_f.f)
    py_code = inspect.getsource(dace_f.f)
    py_code = "import numpy as np\n" + py_code
    for name, value in symbols.items():
        py_code = f"{name} = {value}\n" + py_code

    matches = re.findall(r"dace\.map\[([A-z0-9-+*]*):([A-z0-9-+*]*):?([A-z0-9-+*]?)\]:", py_code)
    # replace dace.map calls
    for match in matches:
        if match[2] == '':
            match = (match[0], match[1])
        dace_range = f"dace.map[{':'.join(match)}]"
        py_range = f"range({','.join(match)})"
        py_code = py_code.replace(dace_range, py_range)

    # remove @dace.program decorator
    py_code = py_code.replace("@dace.program", "")

    # replace dace.float64 or similar with corresponding numpy versions
    matches = re.findall(r"dace\.(float64|float32|int32|int64)\[([A-z0-9, +*\-]*)\]", py_code)
    for match in matches:
        py_code = py_code.replace(f"dace.{match[0]}[{match[1]}]", f"np.ndarray(({match[1]}))")
    return py_code


def generate_sdfg(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                  define_symbols: bool = False, use_dace_auto_opt: bool = False) -> dace.SDFG:
    """
    Generates the SDFG for the given dace program

    :param f: The dace program for which to generate the SDFG
    :type f: dace.frontend.python.parser.DaceProgram
    :param symbols: The symbols used to specialise the SDFG
    :type symbols: Dict[str, int]
    :param save_graphs: If intermediate SDFGs should be stored, defaults to False
    :type save_graphs: bool, optional
    :param define_symbols: If symbols should be specialised, defaults to False
    :type define_symbols: bool, optional
    :param use_dace_auto_opt: Set to true if DaCe default auto opt should be used, defaults to False
    :type use_dace_auto_opt: bool
    :return: The generated SDFG
    :rtype: dace.SDFG
    """
    print(symbols)
    sdfg = f.to_sdfg(validate=True, simplify=True)
    additional_args = {}
    if save_graphs:
        program_name = f"py_{f.name}"
        reset_graph_files(program_name)
        additional_args['verbose_name'] = program_name
    if define_symbols:
        print(f"[run_function_dace] symbols: {symbols}")
        additional_args['symbols'] = copy.deepcopy(symbols)
        for symbol in decrement_symbols:
            additional_args['symbols'][symbol] -= 1
    if use_dace_auto_opt:
        additional_args['use_my_auto_opt'] = False

    sdfg = optimize_sdfg(sdfg, device=dace.DeviceType.GPU, **additional_args)
    return sdfg
