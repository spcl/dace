import inspect
import os
from typing import Dict, Tuple, Union, Callable, List
from numbers import Number
import dace
import sympy
import numpy as np
import pandas as pd

from utils.paths import get_playground_results_dir

df_index_cols = ['program', 'NBLOCKS', 'specialised', 'run_number', 'experiment_id']
description_df_path = os.path.join(get_playground_results_dir(), 'python', 'descriptions.csv')


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


def vert_loop_symbol_wrapper(NBLOCKS: int, KLEV: int, NCLV: int, KLON: int, NCLDTOP: int, KFDIA: int, KIDIA: int,
                             NCLDQL: int, NCLDQI: int, NCLDQS: int, func: Callable,
                             func_args: Dict[str, Union[Number, np.ndarray]]):

    # NBLOCKS = dace.symbol('NBLOCKS')
    # KLEV = dace.symbol('KLEV')
    # NCLV = dace.symbol('NCLV')
    # KLON = dace.symbol('KLON')
    # NCLDTOP = dace.symbol('NCLDTOP')
    # KFDIA = dace.symbol('KFDIA')
    # KIDIA = dace.symbol('KIDIA')
    # NCLDQL = dace.symbol('NCLDQL')
    # NCLDQI = dace.symbol('NCLDQI')
    # NCLDQS = dace.symbol('NCLDQS')
    func(**func_args)


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
