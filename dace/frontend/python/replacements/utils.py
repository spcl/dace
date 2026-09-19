# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains utility functions and types for replacements in the DaCe Python frontend.
"""

from dace import data, dtypes, symbolic
from dace import Memlet, SDFG, SDFGState
from dace.frontend.python import astutils

import itertools
import math
from numbers import Number, Integral
from typing import List, Sequence, Tuple, TYPE_CHECKING, Union

import ml_dtypes
import numpy as np
import sympy as sp

########################################################################
# Type hint definitions
########################################################################

if TYPE_CHECKING:
    from dace.frontend.python.newast import ProgramVisitor
else:
    ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'

Size = Union[int, symbolic.symbol]
Shape = Sequence[Size]
UfuncInput = Union[str, Number, sp.Basic]
UfuncOutput = Union[str, None]

########################################################################
# Helper functions
########################################################################


def simple_call(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                inpname: str,
                func: str,
                restype: dtypes.typeclass = None):
    """ Implements a simple call of the form `out = func(inp)`. """
    create_input = True
    if isinstance(inpname, (list, tuple)):  # TODO investigate this
        inpname = inpname[0]
    if not isinstance(inpname, str) and not symbolic.issymbolic(inpname):
        # Constant parameter
        cst = inpname
        inparr = data.create_datadescriptor(cst)
        inpname = sdfg._find_new_name('constant')
        inparr.transient = True
        sdfg.add_constant(inpname, cst, inparr)
        sdfg.add_datadesc(inpname, inparr)
    elif symbolic.issymbolic(inpname):
        dtype = symbolic.symtype(inpname)
        inparr = data.Scalar(dtype)
        create_input = False
    else:
        inparr = sdfg.arrays[inpname]

    if restype is None:
        restype = inparr.dtype
    outname, outarr = sdfg.add_temp_transient_like(inparr, name=pv.get_target_name())
    outarr.dtype = restype
    num_elements = data._prod(inparr.shape)
    if num_elements == 1:
        if create_input:
            inp = state.add_read(inpname)
            inconn_name = '__inp'
        else:
            inconn_name = symbolic.symstr(inpname)

        out = state.add_write(outname)
        tasklet = state.add_tasklet(func, {'__inp'} if create_input else {}, {'__out'},
                                    f'__out = {func}({inconn_name})')
        if create_input:
            state.add_edge(inp, None, tasklet, '__inp', Memlet.from_array(inpname, inparr))
        state.add_edge(tasklet, '__out', out, None, Memlet.from_array(outname, outarr))
    else:
        state.add_mapped_tasklet(
            name=func,
            map_ranges={
                '__i%d' % i: '0:%s' % n
                for i, n in enumerate(inparr.shape)
            },
            inputs={'__inp': Memlet.simple(inpname, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            code='__out = {f}(__inp)'.format(f=func),
            outputs={'__out': Memlet.simple(outname, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            external_edges=True)

    return outname


def step_state(pv: 'ProgramVisitor', state: SDFGState) -> SDFGState:
    """The state that the next step of a lowering built from several maps has to be emitted into.

    Two maps dropped into one state carry no ordering between them, so a step reading what the
    previous step wrote gets a second, unconnected access node for that transient. Nothing then
    stops map fusion from joining the two maps and leaving the read pointed at a transient the
    fused map has not written yet. A state boundary is the dependency; simplification fuses the
    states back together once it has linked the access nodes.
    """
    if pv is None:
        return state
    return pv._add_state(f'{state.label}_step')


########################################################################
# Shape utilities
########################################################################


def normalize_axes(axes: Tuple[int], max_dim: int) -> List[int]:
    """ Normalize a list of axes by converting negative dimensions to positive.

        :param dims: the list of dimensions, possibly containing negative ints.
        :param max_dim: the total amount of dimensions.
        :return: a list of dimensions containing only positive ints.
    """

    return [ax if ax >= 0 else max_dim + ax for ax in axes]


def broadcast_to(target_shape, operand_shape):
    # the difference to normal broadcasting is that the broadcasted shape is the same as the target
    # I was unable to find documentation for this in numpy, so we follow the description from ONNX
    results = broadcast_together(target_shape, operand_shape, unidirectional=True)

    # the output_shape should be equal to the target_shape
    assert all(i == o for i, o in zip(target_shape, results[0]))

    return results


def broadcast_together(arr1_shape, arr2_shape, unidirectional=False):

    all_idx_dict, all_idx, a1_idx, a2_idx = {}, [], [], []

    max_i = max(len(arr1_shape), len(arr2_shape))

    def get_idx(i):
        return "__i" + str(max_i - i - 1)

    for i, (dim1, dim2) in enumerate(itertools.zip_longest(reversed(arr1_shape), reversed(arr2_shape))):
        all_idx.append(get_idx(i))

        if not symbolic.inequal_symbols(dim1, dim2):
            a1_idx.append(get_idx(i))
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim1

        # if unidirectional, dim2 must also be 1
        elif dim1 == 1 and dim2 is not None and not unidirectional:

            a1_idx.append("0")
            # dim2 != 1 must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 == 1 and dim1 is not None:
            # dim1 != 1 must hold here
            a1_idx.append(get_idx(i))
            a2_idx.append("0")

            all_idx_dict[get_idx(i)] = dim1

        # if unidirectional, this is not allowed
        elif dim1 is None and not unidirectional:
            # dim2 != None must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 is None:
            # dim1 != None must hold here
            a1_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim1
        else:
            if unidirectional:
                raise IndexError(f"could not broadcast input array from shape {arr2_shape} into shape {arr1_shape}")
            else:
                raise IndexError("operands could not be broadcast together with shapes {}, {}".format(
                    arr1_shape, arr2_shape))

    def to_string(idx):
        return ", ".join(reversed(idx))

    out_shape = tuple(reversed([all_idx_dict[idx] for idx in all_idx]))

    all_idx_tup = [(k, "0:" + str(all_idx_dict[k])) for k in reversed(all_idx)]

    return out_shape, all_idx_tup, to_string(all_idx), to_string(a1_idx), to_string(a2_idx)


########################################################################
# Type utilities
########################################################################


def complex_to_scalar(complex_type: dtypes.typeclass):
    if complex_type == dtypes.complex64:
        return dtypes.float32
    elif complex_type == dtypes.complex128:
        return dtypes.float64
    else:
        return complex_type


def representative_num(dtype: Union[dtypes.typeclass, Number]) -> Number:
    if isinstance(dtype, dtypes.typeclass):
        nptype = dtype.type
    else:
        nptype = dtype
    if isinstance(nptype, type):
        nptype_class = nptype
    else:
        nptype_class = type(nptype)
    if issubclass(nptype_class, bool):
        return True
    elif issubclass(nptype_class, np.bool_):
        return np.bool_(True)
    elif issubclass(nptype_class, Integral):
        # NOTE: Returning the max representable integer seems a better choice
        # than 1, however it was causing issues with some programs. This should
        # be revisited in the future.
        # return nptype(np.iinfo(nptype).max)
        return nptype(1)
    else:
        # ml_dtypes' finfo, not numpy's: bfloat16 and the two fp8 types are registered
        # outside numpy's float hierarchy, so np.finfo raises "not inexact" on them. It
        # answers for every numpy float and complex type identically.
        return nptype(ml_dtypes.finfo(nptype_class).resolution)


def np_result_type(nptypes):
    # Fix for np.result_type returning platform-dependent types,
    # e.g. np.longlong
    restype = np.result_type(*nptypes)
    if restype.type not in dtypes.dtype_to_typeclass().keys():
        for k in dtypes.dtype_to_typeclass().keys():
            if k == restype.type:
                return dtypes.dtype_to_typeclass(k)
    return dtypes.dtype_to_typeclass(restype.type)


#: The namespace :func:`sym_type` evaluates a representative value in. ``astutils.unparse`` renders
#: sympy's singletons through the math module -- ``zoo`` comes out as ``math.nan`` -- and the eval
#: below used to be given no globals at all, so those names resolved against whatever this module
#: happened to import. ``h = 1.0 / (N - 1)`` refused with "name 'math' is not defined" for exactly
#: that reason, in a frontend that has no business needing the caller to import math.
SYM_TYPE_NAMESPACE = {'math': math, 'np': np, 'numpy': np, 'nan': math.nan, 'inf': math.inf}


def representative_value(expr: sp.Basic):
    """``expr`` evaluated at one representative point, for the sole purpose of reading its TYPE.

    TWO points are tried, because the first is degenerate for an ordinary expression: the integer
    representative is 1, so every ``N - 1`` denominator is zero and sympy folds the whole expression
    to ``zoo``. A grid spacing ``1.0 / (N - 1)`` is not an error, and the type it infers should come
    from the arithmetic rather than from a division by zero, so the second point moves the integers
    off 1. The first point is kept where it evaluates, since it is the one the rest of the frontend
    has always agreed with. The degeneracy shows up two ways and both are handled: a float division
    folds to ``zoo``, while an integer one (``N // (N - 1)``) raises out of sympy's substitution
    before there is any value to inspect.
    """
    points = (0, 1)
    for offset in points:
        substitutions = [(s, representative_num(s.dtype) + offset) for s in expr.free_symbols]
        try:
            value = eval(astutils.unparse(expr.subs(substitutions)), dict(SYM_TYPE_NAMESPACE))
        except ZeroDivisionError:
            if offset == points[-1]:
                raise
            continue
        if not isinstance(value, float) or math.isfinite(value):
            return value
    return value


def sym_type(expr: Union[symbolic.symbol, sp.Basic]) -> dtypes.typeclass:
    if isinstance(expr, symbolic.symbol):
        return expr.dtype
    pyval = representative_value(expr)
    # Overflow check
    if isinstance(pyval, int) and (pyval > np.iinfo(np.int64).max or pyval < np.iinfo(np.int64).min):
        nptype = np.int64
    else:
        nptype = np.result_type(pyval)
    return np_result_type([nptype])


def cast_str(dtype: dtypes.typeclass) -> str:
    return dtypes.TYPECLASS_TO_STRING[dtype].replace('::', '.')
