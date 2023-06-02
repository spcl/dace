import inspect
import numpy as np
import dace
import cupy as cp
from typing import Dict, Union, Tuple, Optional
from numbers import Number
import sympy
from dace.sdfg import SDFG
from dace.transformation.auto.auto_optimize import auto_optimize as dace_auto_optimize

NBLOCKS = dace.symbol('NBLOCKS')
KLEV = dace.symbol('KLEV')
KLON = dace.symbol('KLON')
NCLDTOP = dace.symbol('NCLDTOP')


@dace.program
def kernel(PLUDE_NF: dace.float64[KLEV, NBLOCKS]):
    for JN in dace.map[0:NBLOCKS:KLON]:
        # Those two lines I'd expect to lead to the same output, but they are not
        # for JK in range(2, 4):
        for JK in range(NCLDTOP, KLEV):
            PLUDE_NF[JK, JN] = KLEV
            # PLUDE_NF[JK, JN] = NCLDTOP


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


def optimize_sdfg(sdfg: SDFG, device: dace.DeviceType, use_my_auto_opt: bool = True,
                  symbols: Optional[Dict[str, Number]] = None):
    """
    Optimizes the given SDFG for the given device using auto_optimize. Will use DaCe or my version based on the given
    flag

    :param sdfg: The SDFG to optimize
    :type sdfg: SDFG
    :param device: The device to optimize it on
    :type device: dace.DeviceType
    :param use_my_auto_opt: Flag to control if my custon auto_opt should be used, defaults to True
    :type use_my_auto_opt: bool, optional
    :param symbols: Dictionary of key, value pairs defining symbols which can be set to const, defaults to None
    :type symbols: Optional[Dict[str, Number]]
    """

    for k, v in sdfg.arrays.items():
        if not v.transient and type(v) == dace.data.Array:
            v.storage = dace.dtypes.StorageType.GPU_Global

    # avoid cyclic dependency
    # from execute.my_auto_opt import auto_optimize as my_auto_optimize
    if device == dace.DeviceType.GPU:
        additional_args = {}
        if symbols:
            additional_args['symbols'] = symbols
        # if use_my_auto_opt:
        #     my_auto_optimize(sdfg, device, **additional_args)
        # else:
        dace_auto_optimize(sdfg, device, **additional_args)

    return sdfg


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


def run(program: dace.frontend.python.parser.DaceProgram):
    symbols = {'KLON': 1, 'KLEV': 4, 'NBLOCKS': 5, 'NCLDTOP': 2, 'NCLV': 10}
    sdfg = program.to_sdfg(validate=True, simplify=True)
    optimize_sdfg(sdfg, device=dace.DeviceType.GPU, symbols=symbols)
    print(sdfg.name)
    csdfg = sdfg.compile()
    arguments = gen_arguments(program, symbols)
    arguments_device = copy_to_device(arguments)
    print(symbols)
    # print(arguments_device['PLUDE_NF'].get())
    csdfg(**arguments_device, **symbols)
    print(arguments_device['PLUDE_NF'].get())


if __name__ == '__main__':
    run(kernel)
