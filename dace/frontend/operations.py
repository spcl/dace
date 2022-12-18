# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function
from functools import partial

from contextlib import contextmanager
from timeit import default_timer as timer
import time
import ast
import numpy as np
import sympy
import os
import sys

from dace import dtypes
from dace.config import Config
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from dace.codegen.compiled_sdfg import CompiledSDFG

class CompiledSDFGProfiler:
    """
    A context manager that prints the time it takes to execute the generated SDFG code
    (excluding init and shutdown).
    """

    times: List[Tuple[str, List[float]]]  #: The list of names and times for each SDFG called within the context.

    def __init__(self, repetitions: int = 0) -> None:
        self.times = []
        self.repetitions = repetitions or int(Config.get('treps'))
    
    @contextmanager
    def time_compiled_sdfg(self, compiled_sdfg: 'CompiledSDFG', *args, **kwargs):
        start = timer()

        times = [start] * (self.repetitions + 1)
        ret = None
        print('\nProfiling...')

        iterator = range(self.repetitions)
        if Config.get_bool('profiling_status'):
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Profiling", file=sys.stdout)
            except ImportError:
                print('WARNING: Cannot show profiling progress, missing optional '
                    'dependency tqdm...\n\tTo see a live progress bar please install '
                    'tqdm (`pip install tqdm`)\n\tTo disable this feature (and '
                    'this warning) set `profiling_status` to false in the dace '
                    'config (~/.dace.conf).')

        for i in iterator:
            # Call function
            ret = compiled_sdfg(*args, **kwargs)
            times[i + 1] = timer()

        diffs = np.array([(times[i] - times[i - 1]) for i in range(1, self.repetitions + 1)])

        # Create profiling directory if it doesn't exist
        profiling_dir = os.path.join(compiled_sdfg.sdfg.build_folder, 'profiling')
        os.makedirs(profiling_dir, exist_ok=True)
        timestamp_string = str(int(time.time() * 1000))
        outfile_path = os.path.join(profiling_dir, 'results-' + timestamp_string + '.csv')

        # Write profiling results to file
        with open(outfile_path, 'w') as f:
            f.write('Program,Runtime_sec\n')
            for d in diffs:
                f.write('%s,%.8f\n' % (compiled_sdfg.sdfg.name, d))

        # Print profiling results
        time_secs = np.median(diffs)
        print(compiled_sdfg.sdfg.name, time_secs * 1000, 'ms')

        self.times.append((compiled_sdfg.sdfg.name, diffs))
        return ret


def detect_reduction_type(wcr_str, openmp=False):
    """ Inspects a lambda function and tries to determine if it's one of the 
        built-in reductions that frameworks such as MPI can provide.

        :param wcr_str: A Python string representation of the lambda function.
        :param openmp: Detect additional OpenMP reduction types.
        :return: dtypes.ReductionType if detected, dtypes.ReductionType.Custom
                 if not detected, or None if no reduction is found.
    """
    if wcr_str == '' or wcr_str is None:
        return None

    # Get lambda function from string
    wcr = eval(wcr_str)
    wcr_ast = ast.parse(wcr_str).body[0].value.body

    # Run function through symbolic math engine
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    try:
        result = wcr(a, b)
    except (TypeError, AttributeError, NameError):  # e.g., "Cannot determine truth value of relational"
        result = None

    # Check resulting value
    if result == sympy.Max(a, b) or (isinstance(wcr_ast, ast.Call) and isinstance(wcr_ast.func, ast.Name)
                                     and wcr_ast.func.id == 'max'):
        return dtypes.ReductionType.Max
    elif result == sympy.Min(a, b) or (isinstance(wcr_ast, ast.Call) and isinstance(wcr_ast.func, ast.Name)
                                       and wcr_ast.func.id == 'min'):
        return dtypes.ReductionType.Min
    elif result == a + b:
        return dtypes.ReductionType.Sum
    elif result == a * b:
        return dtypes.ReductionType.Product
    elif result == a & b:
        return dtypes.ReductionType.Bitwise_And
    elif result == a | b:
        return dtypes.ReductionType.Bitwise_Or
    elif result == a ^ b:
        return dtypes.ReductionType.Bitwise_Xor
    elif isinstance(wcr_ast, ast.BoolOp) and isinstance(wcr_ast.op, ast.And):
        return dtypes.ReductionType.Logical_And
    elif isinstance(wcr_ast, ast.BoolOp) and isinstance(wcr_ast.op, ast.Or):
        return dtypes.ReductionType.Logical_Or
    elif (isinstance(wcr_ast, ast.Compare) and isinstance(wcr_ast.ops[0], ast.NotEq)):
        return dtypes.ReductionType.Logical_Xor
    elif result == b:
        return dtypes.ReductionType.Exchange
    # OpenMP extensions
    elif openmp and result == a - b:
        return dtypes.ReductionType.Sub
    elif openmp and result == a / b:
        return dtypes.ReductionType.Div

    return dtypes.ReductionType.Custom


def is_op_commutative(wcr_str):
    """ Inspects a custom lambda function and tries to determine whether
        it is symbolically commutative (disregarding data type).

        :param wcr_str: A string in Python representing a lambda function.
        :return: True if commutative, False if not, None if cannot be
                 determined.
    """
    if wcr_str == '' or wcr_str is None:
        return None

    # Get lambda function from string
    wcr = eval(wcr_str)

    # Run function through symbolic math engine
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    try:
        aRb = wcr(a, b)
        bRa = wcr(b, a)
    except (TypeError, AttributeError):  # e.g., "Cannot determine truth value of relational"
        return None

    return aRb == bRa


def is_op_associative(wcr_str):
    """ Inspects a custom lambda function and tries to determine whether
        it is symbolically associative (disregarding data type).

        :param wcr_str: A string in Python representing a lambda function.
        :return: True if associative, False if not, None if cannot be
                 determined.
    """
    if wcr_str == '' or wcr_str is None:
        return None

    # Get lambda function from string
    wcr = eval(wcr_str)

    # Run function through symbolic math engine
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    try:
        aRbc = wcr(a, wcr(b, c))
        abRc = wcr(wcr(a, b), c)
    except (TypeError, AttributeError):  # e.g., "Cannot determine truth value of relational"
        return None

    return aRbc == abRc


def reduce(op, in_array, out_array=None, axis=None, identity=None):
    """ Reduces an array according to a binary operation `op`, starting with initial value
        `identity`, over the given axis (or axes if axis is a list), to `out_array`.

        Requires `out_array` with `len(axis)` dimensions less than `in_array`, or a scalar if `axis` is None.

        :param op: binary operation to use for reduction.
        :param in_array: array to reduce.
        :param out_array: output array to write the result to. If `None`, a new array will be returned.
        :param axis: the axis or axes to reduce over. If `None`, all axes will be reduced.
        :param identity: intial value for the reduction. If `None`, uses value stored in output.
        :return: `None` if out_array is given, or the newly created `out_array` if `out_array` is `None`.
    """
    # The function is empty because it is parsed in the Python frontend
    return None


def elementwise(func, in_array, out_array=None):
    """ Applies a function to each element of the array
    
        :param in_array: array to apply to.
        :param out_array: output array to write the result to. If `None`, a new array will be returned
        :param func: lambda function to apply to each element.
        :return: new array with the lambda applied to each element
    """
    # The function is empty because it is parsed in the Python frontend
    return None
