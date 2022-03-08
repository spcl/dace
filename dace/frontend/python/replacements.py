# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import ast
import copy
from copy import deepcopy as dcpy
import itertools
import warnings
from functools import reduce
from numbers import Number, Integral
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import dace
from dace.codegen.tools import type_inference
from dace.config import Config
from dace import data, dtypes, subsets, symbolic, sdfg as sd
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
import dace.frontend.python.memlet_parser as mem_parser
from dace.frontend.python import astutils
from dace.frontend.python.nested_call import NestedCall
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG, SDFGState
from dace.symbolic import pystr_to_symbolic, issymbolic

import numpy as np
import sympy as sp

Size = Union[int, dace.symbolic.symbol]
Shape = Sequence[Size]


def normalize_axes(axes: Tuple[int], max_dim: int) -> List[int]:
    """ Normalize a list of axes by converting negative dimensions to positive.

        :param dims: the list of dimensions, possibly containing negative ints.
        :param max_dim: the total amount of dimensions.
        :return: a list of dimensions containing only positive ints.
    """

    return [ax if ax >= 0 else max_dim + ax for ax in axes]


##############################################################################
# Python function replacements ###############################################
##############################################################################


@oprepo.replaces('dace.define_local')
@oprepo.replaces('dace.ndarray')
def _define_local_ex(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     shape: Shape,
                     dtype: dace.typeclass,
                     storage: dtypes.StorageType = dtypes.StorageType.Default,
                     lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope):
    """ Defines a local array in a DaCe program. """
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    name, _ = sdfg.add_temp_transient(shape, dtype, storage=storage, lifetime=lifetime)
    return name


@oprepo.replaces('numpy.ndarray')
def _define_local(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dace.typeclass):
    """ Defines a local array in a DaCe program. """
    return _define_local_ex(pv, sdfg, state, shape, dtype)


@oprepo.replaces('dace.define_local_scalar')
def _define_local_scalar(pv: 'ProgramVisitor',
                         sdfg: SDFG,
                         state: SDFGState,
                         dtype: dace.typeclass,
                         storage: dtypes.StorageType = dtypes.StorageType.Default,
                         lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope):
    """ Defines a local scalar in a DaCe program. """
    name = sdfg.temp_data_name()
    _, desc = sdfg.add_scalar(name, dtype, transient=True, storage=storage, lifetime=lifetime)
    pv.variables[name] = name
    return name


@oprepo.replaces('dace.define_stream')
def _define_stream(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, dtype: dace.typeclass, buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_stream(name, dtype, buffer_size=buffer_size, transient=True)
    return name


@oprepo.replaces('dace.define_streamarray')
@oprepo.replaces('dace.stream')
def _define_streamarray(pv: 'ProgramVisitor',
                        sdfg: SDFG,
                        state: SDFGState,
                        shape: Shape,
                        dtype: dace.typeclass,
                        buffer_size: Size = 1):
    """ Defines a local stream array in a DaCe program. """
    name = sdfg.temp_data_name()
    sdfg.add_stream(name, dtype, shape=shape, buffer_size=buffer_size, transient=True)
    return name


@oprepo.replaces('numpy.array')
@oprepo.replaces('dace.array')
def _define_literal_ex(pv: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       obj: Any,
                       dtype: dace.typeclass = None,
                       copy: bool = True,
                       order: str = 'K',
                       subok: bool = False,
                       ndmin: int = 0,
                       like: Any = None,
                       storage: Optional[dtypes.StorageType] = None,
                       lifetime: Optional[dtypes.AllocationLifetime] = None):
    """ Defines a literal array in a DaCe program. """
    if like is not None:
        raise NotImplementedError('"like" argument unsupported for numpy.array')

    name = sdfg.temp_data_name()
    if dtype is not None and not isinstance(dtype, dtypes.typeclass):
        dtype = dtypes.typeclass(dtype)

    # From existing data descriptor
    if isinstance(obj, str):
        desc = dcpy(sdfg.arrays[obj])
        if dtype is not None:
            desc.dtype = dtype
    else:  # From literal / constant
        if dtype is None:
            arr = np.array(obj, copy=copy, order=order, subok=subok, ndmin=ndmin)
        else:
            npdtype = dtype.as_numpy_dtype()
            arr = np.array(obj, npdtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
        desc = data.create_datadescriptor(arr)

    # Set extra properties
    desc.transient = True
    if storage is not None:
        desc.storage = storage
    if lifetime is not None:
        desc.lifetime = lifetime

    sdfg.add_datadesc(name, desc)

    # If using existing array, make copy. Otherwise, make constant
    if isinstance(obj, str):
        # Make copy
        rnode = state.add_read(obj)
        wnode = state.add_write(name)
        state.add_nedge(rnode, wnode, dace.Memlet.from_array(name, desc))
    else:
        # Make constant
        sdfg.add_constant(name, arr, desc)

    return name


@oprepo.replaces('dace.reduce')
def _reduce(pv: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            redfunction: Callable[[Any, Any], Any],
            in_array: str,
            out_array=None,
            axis=None,
            identity=None):
    if out_array is None:
        inarr = in_array
        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_shape = None

        # check if we are reducing along all axes
        if axis is not None and len(axis) == len(input_subset.size()):
            reduce_all = all(x == y for x, y in zip(axis, range(len(input_subset.size()))))
        else:
            reduce_all = False

        if axis is None or reduce_all:
            output_shape = [1]
        else:
            output_subset = copy.deepcopy(input_subset)
            output_subset.pop(axis)
            output_shape = output_subset.size()
        if (len(output_shape) == 1 and output_shape[0] == 1):
            outarr = sdfg.temp_data_name()
            outarr, arr = sdfg.add_scalar(outarr, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage, transient=True)
        else:
            outarr, arr = sdfg.add_temp_transient(output_shape, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage)
        output_memlet = Memlet.from_array(outarr, arr)
    else:
        inarr = in_array
        outarr = out_array

        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        # Compute memlets
        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_subset = subsets.Range.from_array(sdfg.arrays[outarr])
        output_memlet = Memlet.simple(outarr, output_subset)

    # Create reduce subgraph
    inpnode = state.add_read(inarr)
    rednode = state.add_reduce(redfunction, axis, identity)
    outnode = state.add_write(outarr)
    state.add_nedge(inpnode, rednode, input_memlet)
    state.add_nedge(rednode, outnode, output_memlet)

    if out_array is None:
        return outarr
    else:
        return []


@oprepo.replaces('numpy.eye')
def eye(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, N, M=None, k=0, dtype=dace.float64):
    M = M or N
    name, _ = sdfg.add_temp_transient([N, M], dtype)

    state.add_mapped_tasklet('eye',
                             dict(i='0:%s' % N, j='0:%s' % M), {},
                             'val = 1 if i == (j - %s) else 0' % k,
                             dict(val=dace.Memlet.simple(name, 'i, j')),
                             external_edges=True)

    return name


@oprepo.replaces('numpy.empty')
def _numpy_empty(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dace.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_local(pv, sdfg, state, shape, dtype)


@oprepo.replaces('numpy.empty_like')
def _numpy_empty_like(pv: 'ProgramVisitor',
                      sdfg: SDFG,
                      state: SDFGState,
                      prototype: str,
                      dtype: dace.typeclass = None,
                      shape: Shape = None):
    """ Creates an unitialized array of the same shape and dtype as prototype.
        The optional dtype and shape inputs allow overriding the corresponding
        attributes of prototype.
    """
    if prototype not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=prototype))
    desc = sdfg.arrays[prototype]
    dtype = dtype or desc.dtype
    shape = shape or desc.shape
    return _define_local(pv, sdfg, state, shape, dtype)


@oprepo.replaces('numpy.identity')
def _numpy_identity(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, n, dtype=dace.float64):
    """ Generates the nxn identity matrix. """
    return eye(pv, sdfg, state, n, dtype=dtype)


@oprepo.replaces('numpy.full')
def _numpy_full(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                shape: Shape,
                fill_value: Union[sp.Expr, Number],
                dtype: dace.typeclass = None):
    """ Creates and array of the specified shape and initializes it with
        the fill value.
    """
    if isinstance(fill_value, (Number, np.bool_)):
        vtype = dtypes.DTYPE_TO_TYPECLASS[type(fill_value)]
    elif isinstance(fill_value, sp.Expr):
        vtype = _sym_type(fill_value)
    else:
        raise mem_parser.DaceSyntaxError(pv, None, "Fill value {f} must be a number!".format(f=fill_value))
    dtype = dtype or vtype
    name, _ = sdfg.add_temp_transient(shape, dtype)

    state.add_mapped_tasklet(
        '_numpy_full_', {"__i{}".format(i): "0: {}".format(s)
                         for i, s in enumerate(shape)}, {},
        "__out = {}".format(fill_value),
        dict(__out=dace.Memlet.simple(name, ",".join(["__i{}".format(i) for i in range(len(shape))]))),
        external_edges=True)

    return name


@oprepo.replaces('numpy.full_like')
def _numpy_full_like(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     a: str,
                     fill_value: Number,
                     dtype: dace.typeclass = None,
                     shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with the fill value.
    """
    if a not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=a))
    desc = sdfg.arrays[a]
    dtype = dtype or desc.dtype
    shape = shape or desc.shape
    return _numpy_full(pv, sdfg, state, shape, fill_value, dtype)


@oprepo.replaces('numpy.ones')
def _numpy_ones(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dace.typeclass = dace.float64):
    """ Creates and array of the specified shape and initializes it with ones.
    """
    return _numpy_full(pv, sdfg, state, shape, 1.0, dtype)


@oprepo.replaces('numpy.ones_like')
def _numpy_ones_like(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     a: str,
                     dtype: dace.typeclass = None,
                     shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with ones.
    """
    return _numpy_full_like(pv, sdfg, state, a, 1.0, dtype, shape)


@oprepo.replaces('numpy.zeros')
def _numpy_zeros(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 shape: Shape,
                 dtype: dace.typeclass = dace.float64):
    """ Creates and array of the specified shape and initializes it with zeros.
    """
    return _numpy_full(pv, sdfg, state, shape, 0.0, dtype)


@oprepo.replaces('numpy.zeros_like')
def _numpy_zeros_like(pv: 'ProgramVisitor',
                      sdfg: SDFG,
                      state: SDFGState,
                      a: str,
                      dtype: dace.typeclass = None,
                      shape: Shape = None):
    """ Creates and array of the same shape and dtype as a and initializes it
        with zeros.
    """
    return _numpy_full_like(pv, sdfg, state, a, 0.0, dtype, shape)


@oprepo.replaces('numpy.copy')
def _numpy_copy(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str):
    """ Creates a copy of array a.
    """
    if a not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=a))
    # TODO: The whole AddTransientMethod class should be move in replacements.py
    from dace.frontend.python.newast import _add_transient_data
    name, desc = _add_transient_data(sdfg, sdfg.arrays[a])
    rnode = state.add_read(a)
    wnode = state.add_write(name)
    state.add_nedge(rnode, wnode, dace.Memlet.from_array(name, desc))
    return name


@oprepo.replaces('numpy.flip')
def _numpy_flip(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, axis=None):
    """ Reverse the order of elements in an array along the given axis.
        The shape of the array is preserved, but the elements are reordered.
    """

    if arr not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=arr))
    desc = sdfg.arrays[arr]
    if isinstance(desc, data.Stream):
        raise mem_parser.DaceSyntaxError(pv, None, "Streams are not supported!")
    if isinstance(desc, data.Scalar):
        return arr

    ndim = len(desc.shape)
    if axis is None:
        axis = [True] * ndim
    else:
        if not isinstance(axis, (list, tuple)):
            axis = [axis]
        axis = [a if a >= 0 else a + ndim for a in axis]
        axis = [True if i in axis else False for i in range(ndim)]

    # TODO: The following code assumes that code generation resolves an inverted copy.
    # sset = ','.join([f'{s}-1:-1:-1' if a else f'0:{s}:1'
    #                  for a, s in zip(axis, desc.shape)])
    # dset = ','.join([f'0:{s}:1' for s in desc.shape])

    # view = _ndarray_reshape(pv, sdfg, state, arr, desc.shape)
    # acpy, _ = sdfg.add_temp_transient(desc.shape, desc.dtype, desc.storage)
    # vnode = state.add_read(view)
    # anode = state.add_read(acpy)
    # state.add_edge(vnode, None, anode, None, Memlet(f'{view}[{sset}] -> {dset}'))

    arr_copy, _ = sdfg.add_temp_transient_like(desc)
    inpidx = ','.join([f'__i{i}' for i in range(ndim)])
    outidx = ','.join([f'{s} - __i{i} - 1' if a else f'__i{i}' for i, (a, s) in enumerate(zip(axis, desc.shape))])
    state.add_mapped_tasklet(name="_numpy_flip_",
                             map_ranges={f'__i{i}': f'0:{s}:1'
                                         for i, s in enumerate(desc.shape)},
                             inputs={'__inp': Memlet(f'{arr}[{inpidx}]')},
                             code='__out = __inp',
                             outputs={'__out': Memlet(f'{arr_copy}[{outidx}]')},
                             external_edges=True)

    return arr_copy


@oprepo.replaces('numpy.rot90')
def _numpy_rot90(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, k=1, axes=(0, 1)):
    """ Rotate an array by 90 degrees in the plane specified by axes.
        Rotation direction is from the first towards the second axis.
    """

    if arr not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=arr))
    desc = sdfg.arrays[arr]
    if not isinstance(desc, (data.Array, data.View)):
        raise mem_parser.DaceSyntaxError(pv, None, "Only Arrays and Views supported!")

    ndim = len(desc.shape)
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == ndim:
        raise ValueError("Axes must be different.")

    if (axes[0] >= ndim or axes[0] < -ndim or axes[1] >= ndim or axes[1] < -ndim):
        raise ValueError("Axes={} out of range for array of ndim={}.".format(axes, ndim))

    k %= 4

    to_flip = []
    transpose = False

    axes_list = list(range(ndim))
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])
    inpidx = ','.join([f'__i{i}' for i in range(ndim)])

    if k == 0:
        return arr
    if k == 2:
        to_flip = [axes[0], axes[1]]
    elif k == 1:
        to_flip = [axes[1]]
        transpose = True
    else:  # k == 3
        to_flip = [axes[0]]
        transpose = True

    arr_copy, narr = sdfg.add_temp_transient_like(desc)

    shape_list = list(narr.shape)
    if transpose:
        shape_list[axes[0]], shape_list[axes[1]] = shape_list[axes[1]], shape_list[axes[0]]

        # Make C-contiguous array shape
        narr.shape = shape_list
        narr.strides = [data._prod(shape_list[i + 1:]) for i in range(len(shape_list))]
        narr.total_size = sum(((shp - 1) * s for shp, s in zip(narr.shape, narr.strides))) + 1
        narr.alignment_offset = 0

    out_indices = [f'{s} - __i{i} - 1' if i in to_flip else f'__i{i}' for i, s in enumerate(desc.shape)]
    if transpose:
        out_indices[axes[0]], out_indices[axes[1]] = out_indices[axes[1]], out_indices[axes[0]]

    outidx = ','.join(out_indices)
    state.add_mapped_tasklet(name="_rot90_",
                             map_ranges={f'__i{i}': f'0:{s}:1'
                                         for i, s in enumerate(desc.shape)},
                             inputs={'__inp': Memlet(f'{arr}[{inpidx}]')},
                             code='__out = __inp',
                             outputs={'__out': Memlet(f'{arr_copy}[{outidx}]')},
                             external_edges=True)

    return arr_copy


@oprepo.replaces('elementwise')
@oprepo.replaces('dace.elementwise')
def _elementwise(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, func: str, in_array: str, out_array=None):
    """Apply a lambda function to each element in the input"""

    inparr = sdfg.arrays[in_array]
    restype = sdfg.arrays[in_array].dtype

    if out_array is None:
        out_array, outarr = sdfg.add_temp_transient(inparr.shape, restype, inparr.storage)
    else:
        outarr = sdfg.arrays[out_array]

    func_ast = ast.parse(func)
    try:
        lambda_ast = func_ast.body[0].value
        if len(lambda_ast.args.args) != 1:
            raise SyntaxError("Expected lambda with one arg, but {} has {}".format(func, len(lambda_ast.args.arrgs)))
        arg = lambda_ast.args.args[0].arg
        replaced_ast = astutils.ASTFindReplace({arg: '__inp'}).visit(lambda_ast.body)
        body = astutils.unparse(replaced_ast)
    except AttributeError:
        raise SyntaxError("Could not parse func {}".format(func))

    code = "__out = {}".format(body)

    num_elements = reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(in_array)
        out = state.add_write(out_array)
        tasklet = state.add_tasklet("_elementwise_", {'__inp'}, {'__out'}, code)
        state.add_edge(inp, None, tasklet, '__inp', Memlet.from_array(in_array, inparr))
        state.add_edge(tasklet, '__out', out, None, Memlet.from_array(out_array, outarr))
    else:
        state.add_mapped_tasklet(
            name="_elementwise_",
            map_ranges={'__i%d' % i: '0:%s' % n
                        for i, n in enumerate(inparr.shape)},
            inputs={'__inp': Memlet.simple(in_array, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            code=code,
            outputs={'__out': Memlet.simple(out_array, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            external_edges=True)

    return out_array


def _simple_call(sdfg: SDFG, state: SDFGState, inpname: str, func: str, restype: dace.typeclass = None):
    """ Implements a simple call of the form `out = func(inp)`. """
    if isinstance(inpname, (list, tuple)):  # TODO investigate this
        inpname = inpname[0]
    if not isinstance(inpname, str):
        # Constant parameter
        cst = inpname
        inparr = data.create_datadescriptor(cst)
        inpname = sdfg.temp_data_name()
        inparr.transient = True
        sdfg.add_constant(inpname, cst, inparr)
        sdfg.add_datadesc(inpname, inparr)
    else:
        inparr = sdfg.arrays[inpname]

    if restype is None:
        restype = inparr.dtype
    outname, outarr = sdfg.add_temp_transient_like(inparr)
    outarr.dtype = restype
    num_elements = data._prod(inparr.shape)
    if num_elements == 1:
        inp = state.add_read(inpname)
        out = state.add_write(outname)
        tasklet = state.add_tasklet(func, {'__inp'}, {'__out'}, '__out = {f}(__inp)'.format(f=func))
        state.add_edge(inp, None, tasklet, '__inp', Memlet.from_array(inpname, inparr))
        state.add_edge(tasklet, '__out', out, None, Memlet.from_array(outname, outarr))
    else:
        state.add_mapped_tasklet(
            name=func,
            map_ranges={'__i%d' % i: '0:%s' % n
                        for i, n in enumerate(inparr.shape)},
            inputs={'__inp': Memlet.simple(inpname, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            code='__out = {f}(__inp)'.format(f=func),
            outputs={'__out': Memlet.simple(outname, ','.join(['__i%d' % i for i in range(len(inparr.shape))]))},
            external_edges=True)

    return outname


def _complex_to_scalar(complex_type: dace.typeclass):
    if complex_type is dace.complex64:
        return dace.float32
    elif complex_type is dace.complex128:
        return dace.float64
    else:
        return complex_type


@oprepo.replaces('exp')
@oprepo.replaces('dace.exp')
@oprepo.replaces('numpy.exp')
@oprepo.replaces('math.exp')
def _exp(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'exp')


@oprepo.replaces('sin')
@oprepo.replaces('dace.sin')
@oprepo.replaces('numpy.sin')
@oprepo.replaces('math.sin')
def _sin(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sin')


@oprepo.replaces('cos')
@oprepo.replaces('dace.cos')
@oprepo.replaces('numpy.cos')
@oprepo.replaces('math.cos')
def _cos(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'cos')


@oprepo.replaces('sqrt')
@oprepo.replaces('dace.sqrt')
@oprepo.replaces('numpy.sqrt')
@oprepo.replaces('math.sqrt')
def _sqrt(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'sqrt')


@oprepo.replaces('log')
@oprepo.replaces('dace.log')
@oprepo.replaces('numpy.log')
@oprepo.replaces('math.log')
def _log(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'log')


@oprepo.replaces('math.floor')
def _floor(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'floor', restype=dtypes.typeclass(int))


@oprepo.replaces('math.ceil')
def _ceil(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'ceil', restype=dtypes.typeclass(int))


@oprepo.replaces('conj')
@oprepo.replaces('dace.conj')
@oprepo.replaces('numpy.conj')
def _conj(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    return _simple_call(sdfg, state, input, 'conj')


@oprepo.replaces('real')
@oprepo.replaces('dace.real')
@oprepo.replaces('numpy.real')
def _real(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return _simple_call(sdfg, state, input, 'real', _complex_to_scalar(inptype))


@oprepo.replaces('imag')
@oprepo.replaces('dace.imag')
@oprepo.replaces('numpy.imag')
def _imag(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return _simple_call(sdfg, state, input, 'imag', _complex_to_scalar(inptype))


@oprepo.replaces('abs')
def _abs(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, input: Union[str, Number, symbolic.symbol]):
    return _simple_call(sdfg, state, input, 'abs')


@oprepo.replaces('transpose')
@oprepo.replaces('dace.transpose')
@oprepo.replaces('numpy.transpose')
def _transpose(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, inpname: str, axes=None):

    if axes is None:
        arr1 = sdfg.arrays[inpname]
        restype = arr1.dtype
        outname, arr2 = sdfg.add_temp_transient((arr1.shape[1], arr1.shape[0]), restype, arr1.storage)

        acc1 = state.add_read(inpname)
        acc2 = state.add_write(outname)
        import dace.libraries.blas  # Avoid import loop
        tasklet = dace.libraries.blas.Transpose('_Transpose_', restype)
        state.add_node(tasklet)
        state.add_edge(acc1, None, tasklet, '_inp', dace.Memlet.from_array(inpname, arr1))
        state.add_edge(tasklet, '_out', acc2, None, dace.Memlet.from_array(outname, arr2))
    else:
        arr1 = sdfg.arrays[inpname]
        if len(axes) != len(arr1.shape) or sorted(axes) != list(range(len(arr1.shape))):
            raise ValueError("axes don't match array")

        new_shape = [arr1.shape[i] for i in axes]
        outname, arr2 = sdfg.add_temp_transient(new_shape, arr1.dtype, arr1.storage)

        state.add_mapped_tasklet(
            "_transpose_", {"_i{}".format(i): "0:{}".format(s)
                            for i, s in enumerate(arr1.shape)},
            dict(_in=Memlet.simple(inpname, ", ".join("_i{}".format(i) for i, _ in enumerate(arr1.shape)))),
            "_out = _in",
            dict(_out=Memlet.simple(outname, ", ".join("_i{}".format(axes[i]) for i, _ in enumerate(arr1.shape)))),
            external_edges=True)

    return outname


@oprepo.replaces('numpy.sum')
def _sum(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(pv, sdfg, state, "lambda x, y: x + y", a, axis=axis, identity=0)


@oprepo.replaces('numpy.mean')
def _mean(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis=None):

    nest = NestedCall(pv, sdfg, state)

    sum = nest(_sum)(a, axis=axis)

    if axis is None:
        div_amount = reduce(lambda x, y: x * y, (d for d in sdfg.arrays[a].shape))
    elif isinstance(axis, (tuple, list)):
        axis = normalize_axes(axis, len(sdfg.arrays[a].shape))
        # each entry needs to be divided by the size of the reduction
        div_amount = reduce(lambda x, y: x * y, (d for i, d in enumerate(sdfg.arrays[a].shape) if i in axis))
    else:
        div_amount = sdfg.arrays[a].shape[axis]

    return nest, nest(_elementwise)("lambda x: x / ({})".format(div_amount), sum)


@oprepo.replaces('numpy.max')
@oprepo.replaces('numpy.amax')
def _max(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(pv,
                   sdfg,
                   state,
                   "lambda x, y: max(x, y)",
                   a,
                   axis=axis,
                   identity=dtypes.min_value(sdfg.arrays[a].dtype))


@oprepo.replaces('numpy.min')
@oprepo.replaces('numpy.amin')
def _min(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis=None):
    return _reduce(pv,
                   sdfg,
                   state,
                   "lambda x, y: min(x, y)",
                   a,
                   axis=axis,
                   identity=dtypes.max_value(sdfg.arrays[a].dtype))


def _minmax2(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, b: str, ismin=True):
    """ Implements the min or max function with 2 scalar arguments. """

    in_conn = set()
    out_conn = {'__out'}

    if isinstance(a, str) and a in sdfg.arrays.keys():
        desc_a = sdfg.arrays[a]
        read_a = state.add_read(a)
        conn_a = '__in_a'
        in_conn.add(conn_a)
    else:
        desc_a = a
        read_a = None
        conn_a = symbolic.symstr(a)

    if isinstance(b, str) and b in sdfg.arrays.keys():
        desc_b = sdfg.arrays[b]
        read_b = state.add_read(b)
        conn_b = '__in_b'
        in_conn.add(conn_b)
    else:
        desc_b = b
        read_b = None
        conn_b = symbolic.symstr(b)

    dtype_c, [cast_a, cast_b] = _result_type([desc_a, desc_b])
    arg_a, arg_b = "{in1}".format(in1=conn_a), "{in2}".format(in2=conn_b)
    if cast_a:
        arg_a = "{ca}({in1})".format(ca=str(cast_a).replace('::', '.'), in1=conn_a)
    if cast_b:
        arg_b = "{cb}({in2})".format(cb=str(cast_b).replace('::', '.'), in2=conn_b)

    func = 'min' if ismin else 'max'
    tasklet = nodes.Tasklet(f'__{func}2', in_conn, out_conn, f'__out = {func}({arg_a}, {arg_b})')

    c = _define_local_scalar(pv, sdfg, state, dtype_c)
    desc_c = sdfg.arrays[c]
    write_c = state.add_write(c)
    if read_a:
        state.add_edge(read_a, None, tasklet, '__in_a', Memlet.from_array(a, desc_a))
    if read_b:
        state.add_edge(read_b, None, tasklet, '__in_b', Memlet.from_array(b, desc_b))
    state.add_edge(tasklet, '__out', write_c, None, Memlet.from_array(c, desc_c))

    return c


# NOTE: We support only the version of Python max that takes scalar arguments.
# For iterable arguments one must use the equivalent NumPy methods.
@oprepo.replaces('max')
def _pymax(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: Union[str, Number, symbolic.symbol], *args):
    left_arg = a
    current_state = state
    for i, b in enumerate(args):
        if i > 0:
            pv._add_state('__min2_%d' % i)
            pv.last_state.set_default_lineinfo(pv.current_lineinfo)
            current_state = pv.last_state
        left_arg = _minmax2(pv, sdfg, current_state, left_arg, b, ismin=False)
    return left_arg


# NOTE: We support only the version of Python min that takes scalar arguments.
# For iterable arguments one must use the equivalent NumPy methods.
@oprepo.replaces('min')
def _pymin(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: Union[str, Number, symbolic.symbol], *args):
    left_arg = a
    current_state = state
    for i, b in enumerate(args):
        if i > 0:
            pv._add_state('__min2_%d' % i)
            pv.last_state.set_default_lineinfo(pv.current_lineinfo)
            current_state = pv.last_state
        left_arg = _minmax2(pv, sdfg, current_state, left_arg, b)
    return left_arg


@oprepo.replaces('slice')
def _slice(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, *args, **kwargs):
    return (slice(*args, **kwargs), )


@oprepo.replaces('numpy.argmax')
def _argmax(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis, result_type=dace.int32):
    return _argminmax(pv, sdfg, state, a, axis, func="max", result_type=result_type)


@oprepo.replaces('numpy.argmin')
def _argmin(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a: str, axis, result_type=dace.int32):
    return _argminmax(pv, sdfg, state, a, axis, func="min", result_type=result_type)


def _argminmax(pv: 'ProgramVisitor',
               sdfg: SDFG,
               state: SDFGState,
               a: str,
               axis,
               func,
               result_type=dace.int32,
               return_both=False):
    nest = NestedCall(pv, sdfg, state)

    assert func in ['min', 'max']

    if axis is None or not isinstance(axis, Integral):
        raise SyntaxError('Axis must be an int')

    a_arr = sdfg.arrays[a]

    if not 0 <= axis < len(a_arr.shape):
        raise SyntaxError("Expected 0 <= axis < len({}.shape), got {}".format(a, axis))

    reduced_shape = list(copy.deepcopy(a_arr.shape))
    reduced_shape.pop(axis)

    val_and_idx = dace.struct('_val_and_idx', val=a_arr.dtype, idx=result_type)

    # HACK: since identity cannot be specified for structs, we have to init the output array
    reduced_structs, reduced_struct_arr = sdfg.add_temp_transient(reduced_shape, val_and_idx)

    code = "__init = _val_and_idx(val={}, idx=-1)".format(
        dtypes.min_value(a_arr.dtype) if func == 'max' else dtypes.max_value(a_arr.dtype))

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_convert_".format(func),
        map_ranges={'__i%d' % i: '0:%s' % n
                    for i, n in enumerate(a_arr.shape) if i != axis},
        inputs={},
        code=code,
        outputs={
            '__init': Memlet.simple(reduced_structs,
                                    ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
        },
        external_edges=True)

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_reduce_".format(func),
        map_ranges={'__i%d' % i: '0:%s' % n
                    for i, n in enumerate(a_arr.shape)},
        inputs={'__in': Memlet.simple(a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))},
        code="__out = _val_and_idx(idx={}, val=__in)".format("__i%d" % axis),
        outputs={
            '__out':
            Memlet.simple(reduced_structs,
                          ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis),
                          wcr_str=("lambda x, y:"
                                   "_val_and_idx(val={}(x.val, y.val), "
                                   "idx=(y.idx if x.val {} y.val else x.idx))").format(
                                       func, '<' if func == 'max' else '>'))
        },
        external_edges=True)

    if return_both:
        outidx, outidxarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        outval, outvalarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, a_arr.dtype)

        nest.add_state().add_mapped_tasklet(
            name="_arg{}_extract_".format(func),
            map_ranges={'__i%d' % i: '0:%s' % n
                        for i, n in enumerate(a_arr.shape) if i != axis},
            inputs={
                '__in': Memlet.simple(reduced_structs,
                                      ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
            },
            code="__out_val = __in.val\n__out_idx = __in.idx",
            outputs={
                '__out_val': Memlet.simple(outval, ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis)),
                '__out_idx': Memlet.simple(outidx, ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
            },
            external_edges=True)

        return nest, (outval, outidx)

    else:
        # map to result_type
        out, outarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        nest(_elementwise)("lambda x: x.idx", reduced_structs, out_array=out)
        return nest, out


@oprepo.replaces('numpy.where')
def _array_array_where(visitor: 'ProgramVisitor',
                       sdfg: SDFG,
                       state: SDFGState,
                       cond_operand: str,
                       left_operand: str = None,
                       right_operand: str = None):
    if left_operand is None or right_operand is None:
        raise ValueError('numpy.where is only supported for the case where x and y are given')

    cond_arr = sdfg.arrays[cond_operand]
    left_arr = sdfg.arrays.get(left_operand, None)
    right_arr = sdfg.arrays.get(right_operand, None)

    left_type = left_arr.dtype if left_arr else dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
    right_type = right_arr.dtype if right_arr else dtypes.DTYPE_TO_TYPECLASS[type(right_operand)]

    # Implicit Python coversion implemented as casting
    arguments = [cond_arr, left_arr or left_type, right_arr or right_type]
    tasklet_args = ['__incond', '__in1' if left_arr else left_operand, '__in2' if right_arr else right_operand]
    result_type, casting = _result_type(arguments[1:])
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[1] = f"{str(left_cast).replace('::', '.')}({tasklet_args[1]})"
    if right_cast is not None:
        tasklet_args[2] = f"{str(right_cast).replace('::', '.')}({tasklet_args[2]})"

    left_shape = left_arr.shape if left_arr else [1]
    right_shape = right_arr.shape if right_arr else [1]
    cond_shape = cond_arr.shape if cond_arr else [1]

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    # Broadcast condition with broadcasted left+right
    _, _, _, cond_idx, _ = _broadcast_together(cond_shape, out_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])
    if isinstance(cond_arr, data.Scalar):
        cond_idx = subsets.Range([(0, 0, 1)])

    if left_arr is None and right_arr is None:
        raise ValueError('Both x and y cannot be scalars in numpy.where')
    storage = left_arr.storage if left_arr else right_arr.storage

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type, storage)

    if list(out_shape) == [1]:
        tasklet = state.add_tasklet('_where_', {'__incond', '__in1', '__in2'}, {'__out'},
                                    '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1], i2=tasklet_args[2]))
        n0 = state.add_read(cond_operand)
        n3 = state.add_write(out_operand)
        state.add_edge(n0, None, tasklet, '__incond', dace.Memlet.from_array(cond_operand, cond_arr))
        if left_arr:
            n1 = state.add_read(left_operand)
            state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_arr))
        if right_arr:
            n2 = state.add_read(right_operand)
            state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_arr))
    else:
        inputs = {}
        inputs['__incond'] = Memlet.simple(cond_operand, cond_idx)
        if left_arr:
            inputs['__in1'] = Memlet.simple(left_operand, left_idx)
        if right_arr:
            inputs['__in2'] = Memlet.simple(right_operand, right_idx)
        state.add_mapped_tasklet("_where_",
                                 all_idx_dict,
                                 inputs,
                                 '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1], i2=tasklet_args[2]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


##############################################################################
# Python operation replacements ##############################################
##############################################################################


def _unop(sdfg: SDFG, state: SDFGState, op1: str, opcode: str, opname: str):
    """ Implements a general element-wise array unary operator. """
    arr1 = sdfg.arrays[op1]

    restype, cast = _result_type([arr1], opname)
    tasklet_code = "__out = {} __in1".format(opcode)
    if cast:
        tasklet_code = tasklet_code.replace('__in1', "{}(__in1)".format(cast))

    # NOTE: This is a fix for np.bool_, which is a true boolean.
    # In this case, the invert operator must become a not operator.
    if opcode == '~' and arr1.dtype == dace.bool_:
        opcode = 'not'

    name, _ = sdfg.add_temp_transient(arr1.shape, restype, arr1.storage)
    state.add_mapped_tasklet("_%s_" % opname, {'__i%d' % i: '0:%s' % s
                                               for i, s in enumerate(arr1.shape)},
                             {'__in1': Memlet.simple(op1, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))},
                             '__out = %s __in1' % opcode,
                             {'__out': Memlet.simple(name, ','.join(['__i%d' % i for i in range(len(arr1.shape))]))},
                             external_edges=True)
    return name


def _broadcast_to(target_shape, operand_shape):
    # the difference to normal broadcasting is that the broadcasted shape is the same as the target
    # I was unable to find documentation for this in numpy, so we follow the description from ONNX
    results = _broadcast_together(target_shape, operand_shape, unidirectional=True)

    # the output_shape should be equal to the target_shape
    assert all(i == o for i, o in zip(target_shape, results[0]))

    return results


def _broadcast_together(arr1_shape, arr2_shape, unidirectional=False):

    all_idx_dict, all_idx, a1_idx, a2_idx = {}, [], [], []

    max_i = max(len(arr1_shape), len(arr2_shape))

    def get_idx(i):
        return "__i" + str(max_i - i - 1)

    for i, (dim1, dim2) in enumerate(itertools.zip_longest(reversed(arr1_shape), reversed(arr2_shape))):
        all_idx.append(get_idx(i))

        if dim1 == dim2:
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
        elif dim1 == None and not unidirectional:
            # dim2 != None must hold here
            a2_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim2

        elif dim2 == None:
            # dim1 != None must hold here
            a1_idx.append(get_idx(i))

            all_idx_dict[get_idx(i)] = dim1
        else:
            if unidirectional:
                raise SyntaxError(f"could not broadcast input array from shape {arr2_shape} into shape {arr1_shape}")
            else:
                raise SyntaxError("operands could not be broadcast together with shapes {}, {}".format(
                    arr1_shape, arr2_shape))

    def to_string(idx):
        return ", ".join(reversed(idx))

    out_shape = tuple(reversed([all_idx_dict[idx] for idx in all_idx]))

    all_idx_tup = [(k, "0:" + str(all_idx_dict[k])) for k in reversed(all_idx)]

    return out_shape, all_idx_tup, to_string(all_idx), to_string(a1_idx), to_string(a2_idx)


def _binop(sdfg: SDFG, state: SDFGState, op1: str, op2: str, opcode: str, opname: str, restype: dace.typeclass):
    """ Implements a general element-wise array binary operator. """
    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    out_shape, all_idx_tup, all_idx, arr1_idx, arr2_idx = _broadcast_together(arr1.shape, arr2.shape)

    name, _ = sdfg.add_temp_transient(out_shape, restype, arr1.storage)
    state.add_mapped_tasklet("_%s_" % opname,
                             all_idx_tup, {
                                 '__in1': Memlet.simple(op1, arr1_idx),
                                 '__in2': Memlet.simple(op2, arr2_idx)
                             },
                             '__out = __in1 %s __in2' % opcode, {'__out': Memlet.simple(name, all_idx)},
                             external_edges=True)
    return name


# Defined as a function in order to include the op and the opcode in the closure
def _makeunop(op, opcode):

    @oprepo.replaces_operator('Array', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        return _unop(sdfg, state, op1, opcode, op)

    @oprepo.replaces_operator('View', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        return _unop(sdfg, state, op1, opcode, op)

    @oprepo.replaces_operator('Scalar', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2=None):
        scalar1 = sdfg.arrays[op1]
        restype, _ = _result_type([scalar1], op)
        op2 = sdfg.temp_data_name()
        _, scalar2 = sdfg.add_scalar(op2, restype, transient=True)
        tasklet = state.add_tasklet("_%s_" % op, {'__in'}, {'__out'}, "__out = %s __in" % opcode)
        node1 = state.add_read(op1)
        node2 = state.add_write(op2)
        state.add_edge(node1, None, tasklet, '__in', dace.Memlet.from_array(op1, scalar1))
        state.add_edge(tasklet, '__out', node2, None, dace.Memlet.from_array(op2, scalar2))
        return op2

    @oprepo.replaces_operator('NumConstant', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: Number, op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)

    @oprepo.replaces_operator('BoolConstant', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: Number, op2=None):
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)

    @oprepo.replaces_operator('symbol', op)
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: 'symbol', op2=None):
        if opcode in _pyop2symtype.keys():
            try:
                return _pyop2symtype[opcode](op1)
            except TypeError:
                pass
        expr = '{o}(op1)'.format(o=opcode)
        vars = {'op1': op1}
        return eval(expr, vars)


def _is_op_arithmetic(op: str):
    if op in {'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Pow', 'Mod', 'FloatPow', 'Heaviside', 'Arctan2', 'Hypot'}:
        return True
    return False


def _is_op_bitwise(op: str):
    if op in {'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'Invert'}:
        return True
    return False


def _is_op_boolean(op: str):
    if op in {
            'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'NotIs', 'Xor', 'FpBoolean', 'SignBit'
    }:
        return True
    return False


def _representative_num(dtype: Union[dtypes.typeclass, Number]) -> Number:
    if isinstance(dtype, dtypes.typeclass):
        nptype = dtype.type
    else:
        nptype = dtype
    if issubclass(nptype, bool):
        return True
    elif issubclass(nptype, np.bool_):
        return np.bool_(True)
    elif issubclass(nptype, Integral):
        # NOTE: Returning the max representable integer seems a better choice
        # than 1, however it was causing issues with some programs. This should
        # be revisited in the future.
        # return nptype(np.iinfo(nptype).max)
        return nptype(1)
    else:
        return nptype(np.finfo(nptype).resolution)


def _np_result_type(nptypes):
    # Fix for np.result_type returning platform-dependent types,
    # e.g. np.longlong
    restype = np.result_type(*nptypes)
    if restype.type not in dtypes.DTYPE_TO_TYPECLASS.keys():
        for k in dtypes.DTYPE_TO_TYPECLASS.keys():
            if k == restype.type:
                return dtypes.DTYPE_TO_TYPECLASS[k]
    return dtypes.DTYPE_TO_TYPECLASS[restype.type]


def _sym_type(expr: Union[symbolic.symbol, sp.Basic]) -> dtypes.typeclass:
    if isinstance(expr, symbolic.symbol):
        return expr.dtype
    representative_value = expr.subs([(s, _representative_num(s.dtype)) for s in expr.free_symbols])
    pyval = eval(astutils.unparse(representative_value))
    # Overflow check
    if isinstance(pyval, int) and (pyval > np.iinfo(np.int64).max or pyval < np.iinfo(np.int64).min):
        nptype = np.int64
    else:
        nptype = np.result_type(pyval)
    return _np_result_type([nptype])


def _cast_str(dtype: dtypes.typeclass) -> str:
    return dtypes.TYPECLASS_TO_STRING[dtype].replace('::', '.')


def _result_type(arguments: Sequence[Union[str, Number, symbolic.symbol, sp.Basic]],
                 operator: str = None) -> Tuple[Union[List[dtypes.typeclass], dtypes.typeclass, str], ...]:

    datatypes = []
    dtypes_for_result = []
    for arg in arguments:
        if isinstance(arg, (data.Array, data.Stream)):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(arg.dtype.type)
        elif isinstance(arg, data.Scalar):
            datatypes.append(arg.dtype)
            dtypes_for_result.append(_representative_num(arg.dtype))
        elif isinstance(arg, (Number, np.bool_)):
            datatypes.append(dtypes.DTYPE_TO_TYPECLASS[type(arg)])
            dtypes_for_result.append(arg)
        elif symbolic.issymbolic(arg):
            datatypes.append(_sym_type(arg))
            dtypes_for_result.append(_representative_num(_sym_type(arg)))
        elif isinstance(arg, dtypes.typeclass):
            datatypes.append(arg)
            dtypes_for_result.append(_representative_num(arg))
        else:
            raise TypeError("Type {t} of argument {a} is not supported".format(t=type(arg), a=arg))

    complex_types = {dace.complex64, dace.complex128, np.complex64, np.complex128}
    float_types = {dace.float16, dace.float32, dace.float64, np.float16, np.float32, np.float64}
    signed_types = {dace.int8, dace.int16, dace.int32, dace.int64, np.int8, np.int16, np.int32, np.int64}
    # unsigned_types = {np.uint8, np.uint16, np.uint32, np.uint64}

    coarse_types = []
    for dtype in datatypes:
        if dtype in complex_types:
            coarse_types.append(3)  # complex
        elif dtype in float_types:
            coarse_types.append(2)  # float
        elif dtype in signed_types:
            coarse_types.append(1)  # signed integer, bool
        else:
            coarse_types.append(0)  # unsigned integer

    casting = [None] * len(arguments)

    if len(arguments) == 1:  # Unary operators

        if not operator:
            result_type = datatypes[0]
        elif operator == 'USub' and coarse_types[0] == 0:
            result_type = eval('dace.int{}'.format(8 * datatypes[0].bytes))
        elif operator == 'Abs' and coarse_types[0] == 3:
            result_type = eval('dace.float{}'.format(4 * datatypes[0].bytes))
        elif (operator in ('Fabs', 'Cbrt', 'Angles', 'SignBit', 'Spacing', 'Modf', 'Floor', 'Ceil', 'Trunc')
              and coarse_types[0] == 3):
            raise TypeError("ufunc '{}' not supported for complex input".format(operator))
        elif (operator in ('Fabs', 'Rint', 'Exp', 'Log', 'Sqrt', 'Cbrt', 'Trigonometric', 'Angles', 'FpBoolean',
                           'Spacing', 'Modf', 'Floor', 'Ceil', 'Trunc') and coarse_types[0] < 2):
            result_type = dace.float64
            casting[0] = _cast_str(result_type)
        elif operator in ('Frexp'):
            if coarse_types[0] == 3:
                raise TypeError("ufunc '{}' not supported for complex "
                                "input".format(operator))
            result_type = [None, dace.int32]
            if coarse_types[0] < 2:
                result_type[0] = dace.float64
                casting[0] = _cast_str(result_type[0])
            else:
                result_type[0] = datatypes[0]
        elif _is_op_bitwise(operator) and coarse_types[0] > 1:
            raise TypeError("unsupported operand type for {}: '{}'".format(operator, datatypes[0]))
        elif _is_op_boolean(operator):
            result_type = dace.bool_
            if operator == 'SignBit' and coarse_types[0] < 2:
                casting[0] = _cast_str(dace.float64)
        else:
            result_type = datatypes[0]

    elif len(arguments) == 2:  # Binary operators

        type1 = coarse_types[0]
        type2 = coarse_types[1]
        dtype1 = datatypes[0]
        dtype2 = datatypes[1]
        max_bytes = max(dtype1.bytes, dtype2.bytes)
        left_cast = None
        right_cast = None

        if _is_op_arithmetic(operator):

            # Float/True division between integers
            if operator == 'Div' and max(type1, type2) < 2:
                # NOTE: Leaving this here in case we implement a C/C++ flag
                # if type1 == type2 and type1 == 0:  # Unsigned integers
                #     result_type = eval('dace.uint{}'.format(8 * max_bytes))
                # else:
                #     result_type = eval('dace.int{}'.format(8 * max_bytes))
                result_type = dace.float64
            # Floor division with at least one complex argument
            # NOTE: NumPy allows this operation
            # elif operator == 'FloorDiv' and max(type1, type2) == 3:
            #     raise TypeError("can't take floor of complex number")
            # Floor division with at least one float argument
            elif operator == 'FloorDiv' and max(type1, type2) == 2:
                if type1 == type2:
                    result_type = eval('dace.float{}'.format(8 * max_bytes))
                else:
                    result_type = dace.float64
            # Floor division between integers
            elif operator == 'FloorDiv' and max(type1, type2) < 2:
                if type1 == type2 and type1 == 0:  # Unsigned integers
                    result_type = eval('dace.uint{}'.format(8 * max_bytes))
                else:
                    result_type = eval('dace.int{}'.format(8 * max_bytes))
            # Power with base integer and exponent signed integer
            elif (operator == 'Pow' and max(type1, type2) < 2 and dtype2 in signed_types):
                result_type = dace.float64
            elif operator == 'FloatPow':
                # Float power with integers or floats
                if max(type1, type2) < 3:
                    result_type = dace.float64
                # Float power with complex numbers
                else:
                    result_type = dace.complex128
            elif (operator in ('Heaviside', 'Arctan2', 'Hypot') and max(type1, type2) == 3):
                raise TypeError("ufunc '{}' not supported for complex input".format(operator))
            elif (operator in ('Heaviside', 'Arctan2', 'Hypot') and max(type1, type2) < 2):
                result_type = dace.float64
            # All other arithmetic operators and cases of the above operators
            else:
                result_type = _np_result_type(dtypes_for_result)

            if dtype1 != result_type:
                left_cast = _cast_str(result_type)
            if dtype2 != result_type:
                right_cast = _cast_str(result_type)

        elif _is_op_bitwise(operator):

            type1 = coarse_types[0]
            type2 = coarse_types[1]
            dtype1 = datatypes[0]
            dtype2 = datatypes[1]

            # Only integers may be arguments of bitwise and shifting operations
            if max(type1, type2) > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            result_type = _np_result_type(dtypes_for_result)
            if dtype1 != result_type:
                left_cast = _cast_str(result_type)
            if dtype2 != result_type:
                right_cast = _cast_str(result_type)

        elif _is_op_boolean(operator):
            result_type = dace.bool_

        elif operator in ('Gcd', 'Lcm'):
            if max(type1, type2) > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            result_type = _np_result_type(dtypes_for_result)
            if dtype1 != result_type:
                left_cast = _cast_str(result_type)
            if dtype2 != result_type:
                right_cast = _cast_str(result_type)

        elif operator and operator in ('CopySign', 'NextAfter'):
            if max(type1, type2) > 2:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            if max(type1, type2) < 2:
                result_type = dace.float64
            else:
                result_type = _np_result_type(dtypes_for_result)
            if dtype1 != result_type:
                left_cast = _cast_str(result_type)
            if dtype2 != result_type:
                right_cast = _cast_str(result_type)

        elif operator and operator in ('Ldexp'):
            if max(type1, type2) > 2 or type2 > 1:
                raise TypeError("unsupported operand type(s) for {}: "
                                "'{}' and '{}'".format(operator, dtype1, dtype2))
            if type1 < 2:
                result_type = dace.float64
                left_cast = _cast_str(result_type)
            else:
                result_type = dtype1
            if dtype2 != dace.int32:
                right_cast = _cast_str(dace.int32)
                if not np.can_cast(dtype2.type, np.int32):
                    warnings.warn("Second input to {} is of type {}, which "
                                  "cannot be safely cast to {}".format(operator, dtype2, dace.int32))

        else:  # Other binary operators
            result_type = _np_result_type(dtypes_for_result)
            if dtype1 != result_type:
                left_cast = _cast_str(result_type)
            if dtype2 != result_type:
                right_cast = _cast_str(result_type)

        casting = [left_cast, right_cast]

    else:  # Operators with 3 or more arguments
        result_type = _np_result_type(dtypes_for_result)
        for i, t in enumerate(coarse_types):
            if t != result_type:
                casting[i] = _cast_str(result_type)

    return result_type, casting


def _array_array_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    '''Both operands are Arrays (or Data in general)'''

    left_arr = sdfg.arrays[left_operand]
    right_arr = sdfg.arrays[right_operand]

    left_type = left_arr.dtype
    right_type = right_arr.dtype

    # Implicit Python coversion implemented as casting
    arguments = [left_arr, right_arr]
    tasklet_args = ['__in1', '__in2']
    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))

    left_shape = left_arr.shape
    right_shape = right_arr.shape

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type, left_arr.storage)

    if list(out_shape) == [1]:
        tasklet = state.add_tasklet('_%s_' % operator, {'__in1', '__in2'}, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n1 = state.add_read(left_operand)
        n2 = state.add_read(right_operand)
        n3 = state.add_write(out_operand)
        state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_arr))
        state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_arr))
    else:
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict, {
                                     '__in1': Memlet.simple(left_operand, left_idx),
                                     '__in2': Memlet.simple(right_operand, right_idx)
                                 },
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _array_const_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    '''Operands are an Array and a Constant'''

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = dtypes.DTYPE_TO_TYPECLASS[type(right_operand)]
        right_shape = [1]
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', f'({str(right_operand)})']
    else:
        left_arr = None
        left_type = dtypes.DTYPE_TO_TYPECLASS[type(left_operand)]
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [f'({str(left_operand)})', '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type, storage)

    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict,
                                 inp_memlets,
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _array_sym_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                     operator: str, opcode: str):
    '''Operands are an Array and a Symbol'''

    if left_operand in sdfg.arrays:
        left_arr = sdfg.arrays[left_operand]
        left_type = left_arr.dtype
        left_shape = left_arr.shape
        storage = left_arr.storage
        right_arr = None
        right_type = _sym_type(right_operand)
        right_shape = [1]
        arguments = [left_arr, right_operand]
        tasklet_args = ['__in1', f'({astutils.unparse(right_operand)})']
    else:
        left_arr = None
        left_type = _sym_type(left_operand)
        left_shape = [1]
        right_arr = sdfg.arrays[right_operand]
        right_type = right_arr.dtype
        right_shape = right_arr.shape
        storage = right_arr.storage
        arguments = [left_operand, right_arr]
        tasklet_args = [f'({astutils.unparse(left_operand)})', '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = _broadcast_together(left_shape, right_shape)

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type, storage)

    if list(out_shape) == [1]:
        if left_arr:
            inp_conn = {'__in1'}
            n1 = state.add_read(left_operand)
        else:
            inp_conn = {'__in2'}
            n2 = state.add_read(right_operand)
        tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                    '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
        n3 = state.add_write(out_operand)
        if left_arr:
            state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_arr))
        else:
            state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_arr))
    else:
        if left_arr:
            inp_memlets = {'__in1': Memlet.simple(left_operand, left_idx)}
        else:
            inp_memlets = {'__in2': Memlet.simple(right_operand, right_idx)}
        state.add_mapped_tasklet("_%s_" % operator,
                                 all_idx_dict,
                                 inp_memlets,
                                 '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand


def _scalar_scalar_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                         operator: str, opcode: str):
    '''Both operands are Scalars'''

    left_scal = sdfg.arrays[left_operand]
    right_scal = sdfg.arrays[right_operand]

    left_type = left_scal.dtype
    right_type = right_scal.dtype

    # Implicit Python coversion implemented as casting
    arguments = [left_scal, right_scal]
    tasklet_args = ['__in1', '__in2']
    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{}(__in1)".format(str(left_cast).replace('::', '.'))
    if right_cast is not None:
        tasklet_args[1] = "{}(__in2)".format(str(right_cast).replace('::', '.'))

    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True, storage=left_scal.storage)

    tasklet = state.add_tasklet('_%s_' % operator, {'__in1', '__in2'}, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n1 = state.add_read(left_operand)
    n2 = state.add_read(right_operand)
    n3 = state.add_write(out_operand)
    state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_scal))
    state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_scal))

    return out_operand


def _scalar_const_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                        operator: str, opcode: str):
    '''Operands are a Scalar and a Constant'''

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        storage = left_scal.storage
        right_scal = None
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', f'({str(right_operand)})']
    else:
        left_scal = None
        right_scal = sdfg.arrays[right_operand]
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [f'({str(left_operand)})', '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True, storage=storage)

    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_scal))

    return out_operand


def _scalar_sym_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                      operator: str, opcode: str):
    '''Operands are a Scalar and a Symbol'''

    if left_operand in sdfg.arrays:
        left_scal = sdfg.arrays[left_operand]
        left_type = left_scal.dtype
        storage = left_scal.storage
        right_scal = None
        right_type = _sym_type(right_operand)
        arguments = [left_scal, right_operand]
        tasklet_args = ['__in1', f'({astutils.unparse(right_operand)})']
    else:
        left_scal = None
        left_type = _sym_type(left_operand)
        right_scal = sdfg.arrays[right_operand]
        right_type = right_scal.dtype
        storage = right_scal.storage
        arguments = [left_operand, right_scal]
        tasklet_args = [f'({astutils.unparse(left_operand)})', '__in2']

    result_type, casting = _result_type(arguments, operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[0] = "{c}({o})".format(c=str(left_cast).replace('::', '.'), o=tasklet_args[0])
    if right_cast is not None:
        tasklet_args[1] = "{c}({o})".format(c=str(right_cast).replace('::', '.'), o=tasklet_args[1])

    out_operand = sdfg.temp_data_name()
    _, out_scal = sdfg.add_scalar(out_operand, result_type, transient=True, storage=storage)

    if left_scal:
        inp_conn = {'__in1'}
        n1 = state.add_read(left_operand)
    else:
        inp_conn = {'__in2'}
        n2 = state.add_read(right_operand)
    tasklet = state.add_tasklet('_%s_' % operator, inp_conn, {'__out'},
                                '__out = {i1} {op} {i2}'.format(i1=tasklet_args[0], op=opcode, i2=tasklet_args[1]))
    n3 = state.add_write(out_operand)
    if left_scal:
        state.add_edge(n1, None, tasklet, '__in1', dace.Memlet.from_array(left_operand, left_scal))
    else:
        state.add_edge(n2, None, tasklet, '__in2', dace.Memlet.from_array(right_operand, right_scal))
    state.add_edge(tasklet, '__out', n3, None, dace.Memlet.from_array(out_operand, out_scal))

    return out_operand


_pyop2symtype = {
    # Boolean ops
    "and": sp.And,
    "or": sp.Or,
    "not": sp.Not,
    # Comparsion ops
    "==": sp.Equality,
    "!=": sp.Unequality,
    ">=": sp.GreaterThan,
    "<=": sp.LessThan,
    ">": sp.StrictGreaterThan,
    "<": sp.StrictLessThan
}


def _const_const_binop(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, left_operand: str, right_operand: str,
                       operator: str, opcode: str):
    '''Both operands are Constants or Symbols'''

    _, casting = _result_type([left_operand, right_operand], operator)
    left_cast = casting[0]
    right_cast = casting[1]

    if isinstance(left_operand, (Number, np.bool_)) and left_cast is not None:
        left = eval(left_cast)(left_operand)
    else:
        left = left_operand
    if isinstance(right_operand, (Number, np.bool_)) and right_cast is not None:
        right = eval(right_cast)(right_operand)
    else:
        right = right_operand

    # Support for SymPy expressions
    if isinstance(left, sp.Basic) or isinstance(right, sp.Basic):
        if opcode in _pyop2symtype.keys():
            try:
                return _pyop2symtype[opcode](left, right)
            except TypeError:
                # This may happen in cases such as `False or (N + 1)`.
                # (N + 1) is a symbolic expressions, but because it is not
                # boolean, SymPy returns TypeError when trying to create
                # `sympy.Or(False, N + 1)`. In such a case, we try again with
                # the normal Python operator.
                pass

    expr = 'l {o} r'.format(o=opcode)
    vars = {'l': left, 'r': right}
    return eval(expr, vars)


def _makebinop(op, opcode):

    @oprepo.replaces_operator('Array', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Array', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('View', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_array_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_scalar_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('Scalar', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('NumConstant', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('BoolConstant', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='Array')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='View')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _array_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='Scalar')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _scalar_sym_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='NumConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='BoolConstant')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)

    @oprepo.replaces_operator('symbol', op, otherclass='symbol')
    def _op(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):
        return _const_const_binop(visitor, sdfg, state, op1, op2, op, opcode)


# Define all standard Python unary operators
for op, opcode in [('UAdd', '+'), ('USub', '-'), ('Not', 'not'), ('Invert', '~')]:
    _makeunop(op, opcode)

# Define all standard Python binary operators
# NOTE: ('MatMult', '@') is defined separately
for op, opcode in [('Add', '+'), ('Sub', '-'), ('Mult', '*'), ('Div', '/'), ('FloorDiv', '//'), ('Mod', '%'),
                   ('Pow', '**'), ('LShift', '<<'), ('RShift', '>>'), ('BitOr', '|'), ('BitXor', '^'), ('BitAnd', '&'),
                   ('And', 'and'), ('Or', 'or'), ('Eq', '=='), ('NotEq', '!='), ('Lt', '<'), ('LtE', '<='), ('Gt', '>'),
                   ('GtE', '>='), ('Is', 'is'), ('IsNot', 'is not')]:
    _makebinop(op, opcode)


@oprepo.replaces_operator('Array', 'MatMult')
@oprepo.replaces_operator('View', 'MatMult')
@oprepo.replaces_operator('Array', 'MatMult', 'View')
@oprepo.replaces_operator('View', 'MatMult', 'Array')
def _matmult(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op1: str, op2: str):

    from dace.libraries.blas.nodes.matmul import MatMul  # Avoid import loop

    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    if len(arr1.shape) > 1 and len(arr2.shape) > 1:  # matrix * matrix

        if len(arr1.shape) > 3 or len(arr2.shape) > 3:
            raise SyntaxError('Matrix multiplication of tensors of dimensions > 3 '
                              'not supported')

        if arr1.shape[-1] != arr2.shape[-2]:
            raise SyntaxError('Matrix dimension mismatch %s != %s' % (arr1.shape[-1], arr2.shape[-2]))

        from dace.libraries.blas.nodes.matmul import _get_batchmm_opts

        # Determine batched multiplication
        bopt = _get_batchmm_opts(arr1.shape, arr1.strides, arr2.shape, arr2.strides, None, None)
        if bopt:
            output_shape = (bopt['b'], arr1.shape[-2], arr2.shape[-1])
        else:
            output_shape = (arr1.shape[-2], arr2.shape[-1])

    elif len(arr1.shape) == 2 and len(arr2.shape) == 1:  # matrix * vector

        if arr1.shape[1] != arr2.shape[0]:
            raise SyntaxError("Number of matrix columns {} must match"
                              "size of vector {}.".format(arr1.shape[1], arr2.shape[0]))

        output_shape = (arr1.shape[0], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 2:  # vector * matrix

        if arr1.shape[0] != arr2.shape[0]:
            raise SyntaxError("Size of vector {} must match number of matrix "
                              "rows {} must match".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (arr2.shape[1], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 1:  # vector * vector

        if arr1.shape[0] != arr2.shape[0]:
            raise SyntaxError("Vectors in vector product must have same size: "
                              "{} vs. {}".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (1, )

    else:  # Dunno what this is, bail

        raise SyntaxError("Cannot multiply arrays with shapes: {} and {}".format(arr1.shape, arr2.shape))

    type1 = arr1.dtype.type
    type2 = arr2.dtype.type
    restype = dace.DTYPE_TO_TYPECLASS[np.result_type(type1, type2).type]

    op3, arr3 = sdfg.add_temp_transient(output_shape, restype, arr1.storage)

    acc1 = state.add_read(op1)
    acc2 = state.add_read(op2)
    acc3 = state.add_write(op3)

    tasklet = MatMul('_MatMult_')
    state.add_node(tasklet)
    state.add_edge(acc1, None, tasklet, '_a', dace.Memlet.from_array(op1, arr1))
    state.add_edge(acc2, None, tasklet, '_b', dace.Memlet.from_array(op2, arr2))
    state.add_edge(tasklet, '_c', acc3, None, dace.Memlet.from_array(op3, arr3))

    return op3


# NumPy ufunc support #########################################################

UfuncInput = Union[str, Number, sp.Basic]
UfuncOutput = Union[str, None]

# TODO: Add all ufuncs in subsequent PR's.
ufuncs = dict(
    add=dict(name="_numpy_add_",
             operator="Add",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = __in1 + __in2",
             reduce="lambda a, b: a + b",
             initial=np.add.identity),
    subtract=dict(name="_numpy_subtract_",
                  operator="Sub",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = __in1 - __in2",
                  reduce="lambda a, b: a - b",
                  initial=np.subtract.identity),
    multiply=dict(name="_numpy_multiply_",
                  operator="Mul",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = __in1 * __in2",
                  reduce="lambda a, b: a * b",
                  initial=np.multiply.identity),
    divide=dict(name="_numpy_divide_",
                operator="Div",
                inputs=["__in1", "__in2"],
                outputs=["__out"],
                code="__out = __in1 / __in2",
                reduce="lambda a, b: a / b",
                initial=np.divide.identity),
    logaddexp=dict(name="_numpy_logaddexp_",
                   operator=None,
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = log( exp(__in1) + exp(__in2) )",
                   reduce="lambda a, b: log( exp(a) + exp(b) )",
                   initial=np.logaddexp.identity),
    logaddexp2=dict(name="_numpy_logaddexp2_",
                    operator=None,
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = log2( exp2(__in1) + exp2(__in2) )",
                    reduce="lambda a, b: log( exp2(a) + exp2(b) )",
                    initial=np.logaddexp2.identity),
    true_divide=dict(name="_numpy_true_divide_",
                     operator="Div",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 / __in2",
                     reduce="lambda a, b: a / b",
                     initial=np.true_divide.identity),
    floor_divide=dict(name="_numpy_floor_divide_",
                      operator="FloorDiv",
                      inputs=["__in1", "__in2"],
                      outputs=["__out"],
                      code="__out = py_floor(__in1, __in2)",
                      reduce="lambda a, b: py_floor(a, b)",
                      initial=np.floor_divide.identity),
    negative=dict(name="_numpy_negative_",
                  operator="USub",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = - __in1",
                  reduce=None,
                  initial=np.negative.identity),
    positive=dict(name="_numpy_positive_",
                  operator="UAdd",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = + __in1",
                  reduce=None,
                  initial=np.positive.identity),
    power=dict(name="_numpy_power_",
               operator="Pow",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = __in1 ** __in2",
               reduce="lambda a, b: a ** b",
               initial=np.power.identity),
    float_power=dict(name="_numpy_float_power_",
                     operator="FloatPow",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = np_float_pow(__in1, __in2)",
                     reduce="lambda a, b: np_float_pow(a, b)",
                     initial=np.float_power.identity),
    remainder=dict(name="_numpy_remainder_",
                   operator="Mod",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = py_mod(__in1, __in2)",
                   reduce="lambda a, b: py_mod(a, b)",
                   initial=np.remainder.identity),
    mod=dict(name="_numpy_mod_",
             operator="Mod",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = py_mod(__in1, __in2)",
             reduce="lambda a, b: py_mod(a, b)",
             initial=np.mod.identity),
    fmod=dict(name="_numpy_fmod_",
              operator="Mod",
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = cpp_mod(__in1, __in2)",
              reduce="lambda a, b: cpp_mod(a, b)",
              initial=np.fmod.identity),
    divmod=dict(name="_numpy_divmod_",
                operator="Div",
                inputs=["__in1", "__in2"],
                outputs=["__out1", "__out2"],
                code="py_divmod(__in1, __in2, __out1, __out2)",
                reduce=None,
                initial=np.divmod.identity),
    absolute=dict(name="_numpy_absolute_",
                  operator="Abs",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = abs(__in1)",
                  reduce=None,
                  initial=np.absolute.identity),
    abs=dict(name="_numpy_abs_",
             operator="Abs",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = abs(__in1)",
             reduce=None,
             initial=np.abs.identity),
    fabs=dict(name="_numpy_fabs_",
              operator="Fabs",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = fabs(__in1)",
              reduce=None,
              initial=np.fabs.identity),
    rint=dict(name="_numpy_rint_",
              operator="Rint",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = round(__in1)",
              reduce=None,
              initial=np.rint.identity),
    sign=dict(name="_numpy_sign_",
              operator=None,
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sign(__in1)",
              reduce=None,
              initial=np.sign.identity),
    heaviside=dict(name="_numpy_heaviside_",
                   operator="Heaviside",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = heaviside(__in1, __in2)",
                   reduce="lambda a, b: heaviside(a, b)",
                   initial=np.heaviside.identity),
    conj=dict(name="_numpy_conj_",
              operator=None,
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = conj(__in1)",
              reduce=None,
              initial=np.conj.identity),
    conjugate=dict(name="_numpy_conjugate_",
                   operator=None,
                   inputs=["__in1"],
                   outputs=["__out"],
                   code="__out = conj(__in1)",
                   reduce=None,
                   initial=np.conjugate.identity),
    exp=dict(name="_numpy_exp_",
             operator="Exp",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = exp(__in1)",
             reduce=None,
             initial=np.exp.identity),
    exp2=dict(name="_numpy_exp2_",
              operator="Exp",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = exp2(__in1)",
              reduce=None,
              initial=np.exp2.identity),
    log=dict(name="_numpy_log_",
             operator="Log",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = log(__in1)",
             reduce=None,
             initial=np.log.identity),
    log2=dict(name="_numpy_log2_",
              operator="Log",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = log2(__in1)",
              reduce=None,
              initial=np.log2.identity),
    log10=dict(name="_numpy_log10_",
               operator="Log",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = log10(__in1)",
               reduce=None,
               initial=np.log10.identity),
    expm1=dict(name="_numpy_expm1_",
               operator="Exp",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = expm1(__in1)",
               reduce=None,
               initial=np.expm1.identity),
    log1p=dict(name="_numpy_log1p_",
               operator="Log",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = log1p(__in1)",
               reduce=None,
               initial=np.log1p.identity),
    sqrt=dict(name="_numpy_sqrt_",
              operator="Sqrt",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sqrt(__in1)",
              reduce=None,
              initial=np.sqrt.identity),
    square=dict(name="_numpy_square_",
                operator=None,
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = __in1 * __in1",
                reduce=None,
                initial=np.square.identity),
    cbrt=dict(name="_numpy_cbrt_",
              operator="Cbrt",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = cbrt(__in1)",
              reduce=None,
              initial=np.cbrt.identity),
    reciprocal=dict(name="_numpy_reciprocal_",
                    operator="Div",
                    inputs=["__in1"],
                    outputs=["__out"],
                    code="__out = reciprocal(__in1)",
                    reduce=None,
                    initial=np.reciprocal.identity),
    gcd=dict(name="_numpy_gcd_",
             operator="Gcd",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = gcd(__in1, __in2)",
             reduce="lambda a, b: gcd(a, b)",
             initial=np.gcd.identity),
    lcm=dict(name="_numpy_lcm_",
             operator="Lcm",
             inputs=["__in1", "__in2"],
             outputs=["__out"],
             code="__out = lcm(__in1, __in2)",
             reduce="lambda a, b: lcm(a, b)",
             initial=np.lcm.identity),
    sin=dict(name="_numpy_sin_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = sin(__in1)",
             reduce=None,
             initial=np.sin.identity),
    cos=dict(name="_numpy_cos_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = cos(__in1)",
             reduce=None,
             initial=np.cos.identity),
    tan=dict(name="_numpy_tan_",
             operator="Trigonometric",
             inputs=["__in1"],
             outputs=["__out"],
             code="__out = tan(__in1)",
             reduce=None,
             initial=np.tan.identity),
    arcsin=dict(name="_numpy_arcsin_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = asin(__in1)",
                reduce=None,
                initial=np.arcsin.identity),
    arccos=dict(name="_numpy_arccos_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = acos(__in1)",
                reduce=None,
                initial=np.arccos.identity),
    arctan=dict(name="_numpy_arctan_",
                operator="Trigonometric",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = atan(__in1)",
                reduce=None,
                initial=np.arctan.identity),
    sinh=dict(name="_numpy_sinh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = sinh(__in1)",
              reduce=None,
              initial=np.sinh.identity),
    cosh=dict(name="_numpy_cosh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = cosh(__in1)",
              reduce=None,
              initial=np.cosh.identity),
    tanh=dict(name="_numpy_tanh_",
              operator="Trigonometric",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = tanh(__in1)",
              reduce=None,
              initial=np.tanh.identity),
    arcsinh=dict(name="_numpy_arcsinh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = asinh(__in1)",
                 reduce=None,
                 initial=np.arcsinh.identity),
    arccosh=dict(name="_numpy_arccosh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = acosh(__in1)",
                 reduce=None,
                 initial=np.arccos.identity),
    arctanh=dict(name="_numpy_arctanh_",
                 operator="Trigonometric",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = atanh(__in1)",
                 reduce=None,
                 initial=np.arctanh.identity),
    arctan2=dict(name="_numpy_arctan2_",
                 operator="Arctan2",
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = atan2(__in1, __in2)",
                 reduce="lambda a, b: atan2(a, b)",
                 initial=np.arctan2.identity),
    hypot=dict(name="_numpy_hypot_",
               operator="Hypot",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = hypot(__in1, __in2)",
               reduce="lambda a, b: hypot(a, b)",
               initial=np.arctan2.identity),
    degrees=dict(name="_numpy_degrees_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = rad2deg(__in1)",
                 reduce=None,
                 initial=np.degrees.identity),
    rad2deg=dict(name="_numpy_rad2deg_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = rad2deg(__in1)",
                 reduce=None,
                 initial=np.rad2deg.identity),
    radians=dict(name="_numpy_radians_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = deg2rad(__in1)",
                 reduce=None,
                 initial=np.radians.identity),
    deg2rad=dict(name="_numpy_deg2rad_",
                 operator="Angles",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = deg2rad(__in1)",
                 reduce=None,
                 initial=np.deg2rad.identity),
    bitwise_and=dict(name="_numpy_bitwise_and_",
                     operator="BitAnd",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 & __in2",
                     reduce="lambda a, b: a & b",
                     initial=np.bitwise_and.identity),
    bitwise_or=dict(name="_numpy_bitwise_or_",
                    operator="BitOr",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 | __in2",
                    reduce="lambda a, b: a | b",
                    initial=np.bitwise_or.identity),
    bitwise_xor=dict(name="_numpy_bitwise_xor_",
                     operator="BitXor",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 ^ __in2",
                     reduce="lambda a, b: a ^ b",
                     initial=np.bitwise_xor.identity),
    invert=dict(name="_numpy_invert_",
                operator="Invert",
                inputs=["__in1"],
                outputs=["__out"],
                code="__out = ~ __in1",
                reduce=None,
                initial=np.invert.identity),
    left_shift=dict(name="_numpy_left_shift_",
                    operator="LShift",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 << __in2",
                    reduce="lambda a, b: a << b",
                    initial=np.left_shift.identity),
    right_shift=dict(name="_numpy_right_shift_",
                     operator="RShift",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 >> __in2",
                     reduce="lambda a, b: a >> b",
                     initial=np.right_shift.identity),
    greater=dict(name="_numpy_greater_",
                 operator="Gt",
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = __in1 > __in2",
                 reduce="lambda a, b: a > b",
                 initial=np.greater.identity),
    greater_equal=dict(name="_numpy_greater_equal_",
                       operator="GtE",
                       inputs=["__in1", "__in2"],
                       outputs=["__out"],
                       code="__out = __in1 >= __in2",
                       reduce="lambda a, b: a >= b",
                       initial=np.greater_equal.identity),
    less=dict(name="_numpy_less_",
              operator="Lt",
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = __in1 < __in2",
              reduce="lambda a, b: a < b",
              initial=np.less.identity),
    less_equal=dict(name="_numpy_less_equal_",
                    operator="LtE",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 <= __in2",
                    reduce="lambda a, b: a <= b",
                    initial=np.less_equal.identity),
    equal=dict(name="_numpy_equal_",
               operator="Eq",
               inputs=["__in1", "__in2"],
               outputs=["__out"],
               code="__out = __in1 == __in2",
               reduce="lambda a, b: a == b",
               initial=np.equal.identity),
    not_equal=dict(name="_numpy_not_equal_",
                   operator="NotEq",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = __in1 != __in2",
                   reduce="lambda a, b: a != b",
                   initial=np.not_equal.identity),
    logical_and=dict(name="_numpy_logical_and_",
                     operator="And",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = __in1 and __in2",
                     reduce="lambda a, b: a and b",
                     initial=np.logical_and.identity),
    logical_or=dict(name="_numpy_logical_or_",
                    operator="Or",
                    inputs=["__in1", "__in2"],
                    outputs=["__out"],
                    code="__out = __in1 or __in2",
                    reduce="lambda a, b: a or b",
                    initial=np.logical_or.identity),
    logical_xor=dict(name="_numpy_logical_xor_",
                     operator="Xor",
                     inputs=["__in1", "__in2"],
                     outputs=["__out"],
                     code="__out = (not __in1) != (not __in2)",
                     reduce="lambda a, b: (not a) != (not b)",
                     initial=np.logical_xor.identity),
    logical_not=dict(name="_numpy_logical_not_",
                     operator="Not",
                     inputs=["__in1"],
                     outputs=["__out"],
                     code="__out = not __in1",
                     reduce=None,
                     initial=np.logical_not.identity),
    maximum=dict(name="_numpy_maximum_",
                 operator=None,
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = max(__in1, __in2)",
                 reduce="lambda a, b: max(a, b)",
                 initial=-np.inf),  # np.maximum.identity is None
    fmax=dict(name="_numpy_fmax_",
              operator=None,
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = fmax(__in1, __in2)",
              reduce="lambda a, b: fmax(a, b)",
              initial=-np.inf),  # np.fmax.identity is None
    minimum=dict(name="_numpy_minimum_",
                 operator=None,
                 inputs=["__in1", "__in2"],
                 outputs=["__out"],
                 code="__out = min(__in1, __in2)",
                 reduce="lambda a, b: min(a, b)",
                 initial=np.inf),  # np.minimum.identity is None
    fmin=dict(name="_numpy_fmin_",
              operator=None,
              inputs=["__in1", "__in2"],
              outputs=["__out"],
              code="__out = fmin(__in1, __in2)",
              reduce="lambda a, b: fmin(a, b)",
              initial=np.inf),  # np.fmin.identity is None
    isfinite=dict(name="_numpy_isfinite_",
                  operator="FpBoolean",
                  inputs=["__in1"],
                  outputs=["__out"],
                  code="__out = isfinite(__in1)",
                  reduce=None,
                  initial=np.isfinite.identity),
    isinf=dict(name="_numpy_isinf_",
               operator="FpBoolean",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = isinf(__in1)",
               reduce=None,
               initial=np.isinf.identity),
    isnan=dict(name="_numpy_isnan_",
               operator="FpBoolean",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = isnan(__in1)",
               reduce=None,
               initial=np.isnan.identity),
    signbit=dict(name="_numpy_signbit_",
                 operator="SignBit",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = signbit(__in1)",
                 reduce=None,
                 initial=np.signbit.identity),
    copysign=dict(name="_numpy_copysign_",
                  operator="CopySign",
                  inputs=["__in1", "__in2"],
                  outputs=["__out"],
                  code="__out = copysign(__in1, __in2)",
                  reduce="lambda a, b: copysign(a, b)",
                  initial=np.copysign.identity),
    nextafter=dict(name="_numpy_nextafter_",
                   operator="NextAfter",
                   inputs=["__in1", "__in2"],
                   outputs=["__out"],
                   code="__out = nextafter(__in1, __in2)",
                   reduce="lambda a, b: nextafter(a, b)",
                   initial=np.nextafter.identity),
    spacing=dict(name="_numpy_spacing_",
                 operator="Spacing",
                 inputs=["__in1"],
                 outputs=["__out"],
                 code="__out = nextafter(__in1, inf) - __in1",
                 reduce=None,
                 initial=np.spacing.identity),
    modf=dict(name="_numpy_modf_",
              operator="Modf",
              inputs=["__in1"],
              outputs=["__out1", "__out2"],
              code="np_modf(__in1, __out1, __out2)",
              reduce=None,
              initial=np.modf.identity),
    ldexp=dict(
        name="_numpy_ldexp_",
        operator="Ldexp",
        inputs=["__in1", "__in2"],
        outputs=["__out"],
        code="__out = ldexp(__in1, __in2)",
        # NumPy apparently has np.ldexp.reduce, but for any kind of input array
        # it returns "TypeError: No loop matching the specified signature and
        # casting was found for ufunc ldexp". Considering that the method
        # computes __in1 * 2 ** __in2, it is hard to define a reduction.
        reduce=None,
        initial=np.ldexp.identity),
    frexp=dict(name="_numpy_frexp_",
               operator="Frexp",
               inputs=["__in1"],
               outputs=["__out1", "__out2"],
               code="np_frexp(__in1, __out1, __out2)",
               reduce=None,
               initial=np.frexp.identity),
    floor=dict(name="_numpy_floor_",
               operator="Floor",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = floor(__in1)",
               reduce=None,
               initial=np.floor.identity),
    ceil=dict(name="_numpy_ceil_",
              operator="Ceil",
              inputs=["__in1"],
              outputs=["__out"],
              code="__out = ceil(__in1)",
              reduce=None,
              initial=np.ceil.identity),
    trunc=dict(name="_numpy_trunc_",
               operator="Trunc",
               inputs=["__in1"],
               outputs=["__out"],
               code="__out = trunc(__in1)",
               reduce=None,
               initial=np.trunc.identity),
)


def _get_ufunc_impl(visitor: 'ProgramVisitor', ast_node: ast.Call, ufunc_name: str) -> Dict[str, Any]:
    """ Retrieves the implementation details for a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc

        :raises DaCeSyntaxError: When the ufunc implementation is missing
    """

    try:
        return ufuncs[ufunc_name]
    except KeyError:
        raise mem_parser.DaceSyntaxError(visitor, ast_node,
                                         "Missing implementation for NumPy ufunc {f}.".format(f=ufunc_name))


def _validate_ufunc_num_arguments(visitor: 'ProgramVisitor', ast_node: ast.Call, ufunc_name: str, num_inputs: int,
                                  num_outputs: int, num_args: int):
    """ Validates the number of positional arguments in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param num_outputs: Number of ufunc outputs
        :param num_args: Number of positional argumnents in the ufunc call

        :raises DaCeSyntaxError: When validation fails
    """

    if num_args > num_inputs + num_outputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected a maximum of {i} input(s) and {o} output(s), "
            "but a total of {a} arguments were given).".format(f=ufunc_name, i=num_inputs, o=num_outputs, a=num_args))


def _validate_ufunc_inputs(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, ufunc_name: str, num_inputs: int,
                           num_args: int, args: Sequence[UfuncInput]) -> List[UfuncInput]:
    """ Validates the number of type of inputs in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param args: Positional arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of input datanames and constants
    """

    # Validate number of inputs
    if num_args > num_inputs:
        # Assume that the first "num_inputs" arguments are inputs
        inputs = args[:num_inputs]
    elif num_args < num_inputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} inputs, but {a} were given).".format(f=ufunc_name, e=num_inputs, a=num_args))
    else:
        inputs = args
    if isinstance(inputs, (list, tuple)):
        inputs = list(inputs)
    else:
        inputs = [inputs]

    # Validate type of inputs
    for arg in inputs:
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            pass
        elif isinstance(arg, (Number, sp.Basic)):
            pass
        else:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Input arguments in call to numpy.{f} must be of dace.data.Data "
                "type or numerical/boolean constants (invalid argument {a})".format(f=ufunc_name, a=arg))

    return inputs


def _validate_ufunc_outputs(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, ufunc_name: str, num_inputs: int,
                            num_outputs: int, num_args: int, args: Sequence[UfuncInput],
                            kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Validates the number of type of outputs in a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param num_inputs: Number of ufunc inputs
        :param num_outputs: Number of ufunc outputs
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames and None
    """

    # Validate number of outputs
    num_pos_outputs = num_args - num_inputs
    if num_pos_outputs == 0 and "out" not in kwargs.keys():
        outputs = [None] * num_outputs
    elif num_pos_outputs > 0 and "out" in kwargs.keys():
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "You cannot specify 'out' in call to numpy.{f} as both a positional"
            " and keyword argument (positional {p}, keyword {w}).".format(f=ufunc_name,
                                                                          p=args[num_outputs, :],
                                                                          k=kwargs['out']))
    elif num_pos_outputs > 0:
        outputs = list(args[num_inputs:])
        # TODO: Support the following undocumented NumPy behavior?
        # NumPy allows to specify less than `expected_num_outputs` as
        # positional arguments. For example, `np.divmod` has 2 outputs, the
        # quotient and the remainder. `np.divmod(A, B, C)` works, but
        # `np.divmod(A, B, out=C)` or `np.divmod(A, B, out=(C))` doesn't.
        # In the case of output as a positional argument, C will be set to
        # the quotient of the floor division, while a new array will be
        # generated for the remainder.
    else:
        outputs = kwargs["out"]
    if isinstance(outputs, (list, tuple)):
        outputs = list(outputs)
    else:
        outputs = [outputs]
    if len(outputs) != num_outputs:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Invalid number of arguments in call to numpy.{f} "
            "(expected {e} outputs, but {a} were given).".format(f=ufunc_name, e=num_outputs, a=len(outputs)))

    # Validate outputs
    for arg in outputs:
        if arg is None:
            pass
        elif isinstance(arg, str) and arg in sdfg.arrays.keys():
            pass
        else:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Return arguments in call to numpy.{f} must be of "
                "dace.data.Data type.".format(f=ufunc_name))

    return outputs


def _validate_where_kword(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, ufunc_name: str,
                          kwargs: Dict[str, Any]) -> Tuple[bool, Union[str, bool]]:
    """ Validates the 'where' keyword argument passed to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call

        :raises DaceSyntaxError: When validation fails

        :returns: Tuple of a boolean value indicating whether the 'where'
        keyword is defined, and the validated 'where' value
    """

    has_where = False
    where = None
    if 'where' in kwargs.keys():
        where = kwargs['where']
        if isinstance(where, str) and where in sdfg.arrays.keys():
            has_where = True
        elif isinstance(where, (bool, np.bool_)):
            has_where = True
        elif isinstance(where, (list, tuple)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Values for the 'where' keyword that are a sequence of boolean "
                " constants are unsupported. Please, pass these values to the "
                " {n} call through a DaCe boolean array.".format(n=ufunc_name))
        else:
            # NumPy defaults to "where=True" for invalid values for the keyword
            pass

    return has_where, where


def _validate_shapes(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, ufunc_name: str,
                     inputs: List[UfuncInput],
                     outputs: List[UfuncOutput]) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Validates the data shapes of inputs and outputs to a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param ufunc_name: Name of the ufunc
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: Tuple with the output shape, the map, output and input indices
    """

    shapes = []
    for arg in inputs + outputs:
        if isinstance(arg, str):
            array = sdfg.arrays[arg]
            shapes.append(array.shape)
        else:
            shapes.append([])
    try:
        result = _broadcast(shapes)
    except SyntaxError as e:
        raise mem_parser.DaceSyntaxError(
            visitor, ast_node, "Shape validation in numpy.{f} call failed. The following error "
            "occured : {m}".format(f=ufunc_name, m=str(e)))
    return result


def _broadcast(shapes: Sequence[Shape]) -> Tuple[Shape, Tuple[Tuple[str, str], ...], str, List[str]]:
    """ Applies the NumPy ufunc brodacsting rules in a sequence of data shapes
        (see https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting).

        :param shapes: Sequence (list, tuple) of data shapes

        :raises SyntaxError: When broadcasting fails

        :returns: Tuple with the output shape, the map, output and input indices
    """

    map_lengths = dict()
    output_indices = []
    input_indices = [[] for _ in shapes]

    ndims = [len(shape) for shape in shapes]
    max_i = max(ndims)

    def get_idx(i):
        return "__i" + str(max_i - i - 1)

    def to_string(idx):
        return ", ".join(reversed(idx))

    reversed_shapes = [reversed(shape) for shape in shapes]
    for i, dims in enumerate(itertools.zip_longest(*reversed_shapes)):
        output_indices.append(get_idx(i))

        not_none_dims = [d for d in dims if d is not None]
        # Per NumPy broadcasting rules, we need to find the largest dimension.
        # However, `max_dim = max(not_none_dims)` does not work with symbols.
        # Therefore, we sequentially check every not-none dimension.
        # Symbols are assumed to be larger than constants.
        # This will not work properly otherwise.
        # If more than 1 (different) symbols are found, then this fails, because
        # we cannot know which will have the greater size.
        # NOTE: This is a compromise. NumPy broadcasting depends on knowing
        # the exact array sizes. However, symbolic sizes are not known at this
        # point.
        max_dim = 0
        for d in not_none_dims:
            if isinstance(max_dim, Number):
                if isinstance(d, Number):
                    max_dim = max(max_dim, d)
                elif symbolic.issymbolic(d):
                    max_dim = d
                else:
                    raise NotImplementedError
            elif symbolic.issymbolic(max_dim):
                if isinstance(d, Number):
                    pass
                elif symbolic.issymbolic(d):
                    if max_dim != d:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        map_lengths[get_idx(i)] = max_dim
        for j, d in enumerate(dims):
            if d is None:
                pass
            elif d == 1:
                input_indices[j].append('0')
            elif d == max_dim:
                input_indices[j].append(get_idx(i))
            else:
                raise SyntaxError("Operands could not be broadcast together with shapes {}.".format(','.join(
                    str(shapes))))

    out_shape = tuple(reversed([map_lengths[idx] for idx in output_indices]))
    map_indices = [(k, "0:" + str(map_lengths[k])) for k in reversed(output_indices)]
    output_indices = to_string(output_indices)
    input_indices = [to_string(idx) for idx in input_indices]

    if not out_shape:
        out_shape = (1, )
        output_indices = "0"

    return out_shape, map_indices, output_indices, input_indices


def _create_output(sdfg: SDFG,
                   inputs: List[UfuncInput],
                   outputs: List[UfuncOutput],
                   output_shape: Shape,
                   output_dtype: Union[dtypes.typeclass, List[dtypes.typeclass]],
                   storage: dtypes.StorageType = None,
                   force_scalar: bool = False) -> List[UfuncOutput]:
    """ Creates output data for storing the result of a NumPy ufunc call.

        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call
        :param output_shape: Shape of the output data
        :param output_dtype: Datatype of the output data
        :param storage: Storage type of the output data
        :param force_scalar: If True and output shape is (1,) then output
        becomes a dace.data.Scalar, regardless of the data-type of the inputs

        :returns: New outputs of the ufunc call
    """

    # Check if the result is scalar
    is_output_scalar = True
    for arg in inputs:
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            datadesc = sdfg.arrays[arg]
            # If storage is not set, then choose the storage of the first
            # data input.
            if not storage:
                storage = datadesc.storage
            # TODO: What about streams?
            if not isinstance(datadesc, data.Scalar):
                is_output_scalar = False
                break

    # Set storage
    storage = storage or dtypes.StorageType.Default

    # Validate datatypes
    if isinstance(output_dtype, (list, tuple)):
        if len(output_dtype) == 1:
            datatypes = [output_dtype[0]] * len(outputs)
        elif len(output_dtype) == len(outputs):
            datatypes = output_dtype
        else:
            raise ValueError("Missing output datatypes")
    else:
        datatypes = [output_dtype] * len(outputs)

    # Create output data (if needed)
    for i, (arg, datatype) in enumerate(zip(outputs, datatypes)):
        if arg is None:
            if (len(output_shape) == 1 and output_shape[0] == 1 and (is_output_scalar or force_scalar)):
                output_name = sdfg.temp_data_name()
                sdfg.add_scalar(output_name, output_dtype, transient=True, storage=storage)
                outputs[i] = output_name
            else:
                outputs[i], _ = sdfg.add_temp_transient(output_shape, datatype)

    return outputs


def _set_tasklet_params(ufunc_impl: Dict[str, Any],
                        inputs: List[UfuncInput],
                        casting: List[dtypes.typeclass] = None) -> Dict[str, Any]:
    """ Sets the tasklet parameters for a NumPy ufunc call.

        :param ufunc_impl: Information on how the ufunc must be implemented
        :param inputs: Inputs of the ufunc call

        :returns: Dictionary with the (1) tasklet name, (2) input connectors,
                  (3) output connectors, and (4) tasklet code
    """

    # (Deep) copy default tasklet parameters from the ufunc_impl dictionary
    name = ufunc_impl['name']
    inp_connectors = copy.deepcopy(ufunc_impl['inputs'])
    out_connectors = copy.deepcopy(ufunc_impl['outputs'])
    code = ufunc_impl['code']

    # Remove input connectors related to constants
    # and fix constants/symbols in the tasklet code
    for i, arg in reversed(list(enumerate(inputs))):
        inp_conn = inp_connectors[i]
        if casting and casting[i]:
            repl = "{c}({o})".format(c=str(casting[i]).replace('::', '.'), o=inp_conn)
            code = code.replace(inp_conn, repl)
        if isinstance(arg, (Number, sp.Basic)):
            inp_conn = inp_connectors[i]
            code = code.replace(inp_conn, astutils.unparse(arg))
            inp_connectors.pop(i)

    return dict(name=name, inputs=inp_connectors, outputs=out_connectors, code=code)


def _create_subgraph(visitor: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     inputs: List[UfuncInput],
                     outputs: List[UfuncOutput],
                     map_indices: Tuple[str, str],
                     input_indices: List[str],
                     output_indices: str,
                     output_shape: Shape,
                     tasklet_params: Dict[str, Any],
                     has_where: bool = False,
                     where: Union[str, bool] = None):
    """ Creates the subgraph that implements a NumPy ufunc call.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param inputs: Inputs of the ufunc call
        :param outputs: Outputs of the ufunc call
        :param map_indices: Map (if needed) indices
        :param input_indices: Input indices for inner-most memlets
        :param output_indices: Output indices for inner-most memlets
        :param output_shape: Shape of the output
        :param tasklet_params: Dictionary with the tasklet parameters
        :param has_where: True if the 'where' keyword is set
        :param where: Keyword 'where' value
    """

    # Create subgraph
    if list(output_shape) == [1]:
        # No map needed
        if has_where:
            if isinstance(where, (bool, np.bool_)):
                if where == True:
                    pass
                elif where == False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                cond_state = state
                where_data = sdfg.arrays[where]
                if not isinstance(where_data, data.Scalar):
                    name = sdfg.temp_data_name()
                    sdfg.add_scalar(name, where_data.dtype, transient=True)
                    r = cond_state.add_read(where)
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, dace.Memlet("{}[0]".format(r)))
                true_state = sdfg.add_state(label=cond_state.label + '_true')
                state = true_state
                visitor.last_state = state
                cond = name
                cond_else = 'not ({})'.format(cond)
                sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(cond))
        tasklet = state.add_tasklet(**tasklet_params)
        inp_conn_idx = 0
        for arg in inputs:
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                inp_node = state.add_read(arg)
                state.add_edge(inp_node, None, tasklet, tasklet_params['inputs'][inp_conn_idx],
                               dace.Memlet.from_array(arg, sdfg.arrays[arg]))
                inp_conn_idx += 1
        for i, arg in enumerate(outputs):
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                out_node = state.add_write(arg)
                state.add_edge(tasklet, tasklet_params['outputs'][i], out_node, None,
                               dace.Memlet.from_array(arg, sdfg.arrays[arg]))
        if has_where and isinstance(where, str) and where in sdfg.arrays.keys():
            visitor._add_state(label=cond_state.label + '_true')
            sdfg.add_edge(cond_state, visitor.last_state, dace.InterstateEdge(cond_else))
    else:
        # Map needed
        if has_where:
            if isinstance(where, (bool, np.bool_)):
                if where == True:
                    pass
                elif where == False:
                    return
            elif isinstance(where, str) and where in sdfg.arrays.keys():
                nested_sdfg = dace.SDFG(state.label + "_where")
                nested_sdfg_inputs = dict()
                nested_sdfg_outputs = dict()
                nested_sdfg._temp_transients = sdfg._temp_transients

                for idx, arg in enumerate(inputs + [where]):
                    if not (isinstance(arg, str) and arg in sdfg.arrays.keys()):
                        continue
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg.temp_data_name()
                    nested_sdfg_inputs[arg] = (conn_name, input_indices[idx])
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError

                for arg in outputs:
                    arg_data = sdfg.arrays[arg]
                    conn_name = nested_sdfg.temp_data_name()
                    nested_sdfg_outputs[arg] = (conn_name, output_indices)
                    if isinstance(arg_data, data.Scalar):
                        nested_sdfg.add_scalar(conn_name, arg_data.dtype)
                    elif isinstance(arg_data, data.Array):
                        nested_sdfg.add_array(conn_name, [1], arg_data.dtype)
                    else:
                        raise NotImplementedError

                cond_state = nested_sdfg.add_state(label=state.label + "_where_cond", is_start_state=True)
                where_data = sdfg.arrays[where]
                if isinstance(where_data, data.Scalar):
                    name = nested_sdfg_inputs[where]
                elif isinstance(where_data, data.Array):
                    name = nested_sdfg.temp_data_name()
                    nested_sdfg.add_scalar(name, where_data.dtype, transient=True)
                    r = cond_state.add_read(nested_sdfg_inputs[where][0])
                    w = cond_state.add_write(name)
                    cond_state.add_nedge(r, w, dace.Memlet("{}[0]".format(r)))

                sdfg._temp_transients = nested_sdfg._temp_transients

                true_state = nested_sdfg.add_state(label=cond_state.label + '_where_true')
                cond = name
                cond_else = 'not ({})'.format(cond)
                nested_sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(cond))

                tasklet = true_state.add_tasklet(**tasklet_params)
                idx = 0
                for arg in inputs:
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        inp_name, _ = nested_sdfg_inputs[arg]
                        inp_data = nested_sdfg.arrays[inp_name]
                        inp_node = true_state.add_read(inp_name)
                        true_state.add_edge(inp_node, None, tasklet, tasklet_params['inputs'][idx],
                                            dace.Memlet.from_array(inp_name, inp_data))
                        idx += 1
                for i, arg in enumerate(outputs):
                    if isinstance(arg, str) and arg in sdfg.arrays.keys():
                        out_name, _ = nested_sdfg_outputs[arg]
                        out_data = nested_sdfg.arrays[out_name]
                        out_node = true_state.add_write(out_name)
                        true_state.add_edge(tasklet, tasklet_params['outputs'][i], out_node, None,
                                            dace.Memlet.from_array(out_name, out_data))

                false_state = nested_sdfg.add_state(label=state.label + '_where_false')
                nested_sdfg.add_edge(cond_state, false_state, dace.InterstateEdge(cond_else))
                nested_sdfg.add_edge(true_state, false_state, dace.InterstateEdge())

                codenode = state.add_nested_sdfg(nested_sdfg, sdfg, set([n for n, _ in nested_sdfg_inputs.values()]),
                                                 set([n for n, _ in nested_sdfg_outputs.values()]))
                me, mx = state.add_map(state.label + '_map', map_indices)
                for arg in inputs + [where]:
                    if not (isinstance(arg, str) and arg in sdfg.arrays.keys()):
                        continue
                    n = state.add_read(arg)
                    conn, idx = nested_sdfg_inputs[arg]
                    state.add_memlet_path(n,
                                          me,
                                          codenode,
                                          memlet=dace.Memlet("{a}[{i}]".format(a=n, i=idx)),
                                          dst_conn=conn)
                for arg in outputs:
                    n = state.add_write(arg)
                    conn, idx = nested_sdfg_outputs[arg]
                    state.add_memlet_path(codenode,
                                          mx,
                                          n,
                                          memlet=dace.Memlet("{a}[{i}]".format(a=n, i=idx)),
                                          src_conn=conn)
                return

        input_memlets = dict()
        inp_conn_idx = 0
        for arg, idx in zip(inputs, input_indices):
            if isinstance(arg, str) and arg in sdfg.arrays.keys():
                conn = tasklet_params['inputs'][inp_conn_idx]
                input_memlets[conn] = Memlet.simple(arg, idx)
                inp_conn_idx += 1
        output_memlets = {
            out_conn: Memlet.simple(arg, output_indices)
            for arg, out_conn in zip(outputs, tasklet_params['outputs'])
        }
        state.add_mapped_tasklet(tasklet_params['name'],
                                 map_indices,
                                 input_memlets,
                                 tasklet_params['code'],
                                 output_memlets,
                                 external_edges=True)


def _flatten_args(args: Sequence[UfuncInput]) -> Sequence[UfuncInput]:
    """ Flattens arguments of a NumPy ufunc. This is useful in cases where
        one of the arguments is the result of another operation or ufunc, which
        may be a list of Dace data.
    """
    flat_args = []
    for arg in args:
        if isinstance(arg, list):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)
    return flat_args


@oprepo.replaces_ufunc('ufunc')
def implement_ufunc(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                    args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg, ufunc_name, kwargs)

    # Validate data shapes and apply NumPy broadcasting rules
    inp_shapes = copy.deepcopy(inputs)
    if has_where:
        inp_shapes += [where]
    (out_shape, map_indices, out_indices, inp_indices) = _validate_shapes(visitor, ast_node, sdfg, ufunc_name,
                                                                          inp_shapes, outputs)

    # Infer result type
    result_type, casting = _result_type(
        [sdfg.arrays[arg] if isinstance(arg, str) and arg in sdfg.arrays else arg for arg in inputs],
        ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.DTYPE_TO_TYPECLASS.keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor,
                     sdfg,
                     state,
                     inputs,
                     outputs,
                     map_indices,
                     inp_indices,
                     out_indices,
                     out_shape,
                     tasklet_params,
                     has_where=has_where,
                     where=where)

    return outputs


def _validate_keepdims_kword(visitor: 'ProgramVisitor', ast_node: ast.Call, ufunc_name: str, kwargs: Dict[str,
                                                                                                          Any]) -> bool:
    """ Validates the 'keepdims' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param ufunc_name: Name of the ufunc
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: Boolean value of the 'keepdims' keyword argument
    """

    keepdims = False
    if 'keepdims' in kwargs.keys():
        keepdims = kwargs['keepdims']
        if not isinstance(keepdims, (Integral, bool, np.bool_)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Integer or boolean value expected for keyword argument "
                "'keepdims' in reduction operation {f} (got {v}).".format(f=ufunc_name, v=keepdims))
        if not isinstance(keepdims, (bool, np.bool_)):
            keepdims = bool(keepdims)

    return keepdims


def _validate_axis_kword(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, inputs: List[UfuncInput],
                         kwargs: Dict[str, Any], keepdims: bool) -> Tuple[Tuple[int, ...], Union[Shape, None], Shape]:
    """ Validates the 'axis' keyword argument of a NumPy ufunc call.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param inputs: Inputs of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call
        :param keepdims: Boolean value of the 'keepdims' keyword argument

        :raises DaCeSyntaxError: When validation fails

        :returns: The value of the 'axis' keyword argument, the intermediate
        data shape (if needed), and the expected output shape
    """

    # Validate 'axis' keyword
    axis = (0, )
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        inp_shape = sdfg.arrays[inputs[0]].shape
    else:
        inp_shape = [1]
    if 'axis' in kwargs.keys():
        # Set to (0, 1, 2, ...) if the keyword arg value is None
        axis = kwargs['axis'] or tuple(range(len(inp_shape)))
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
    if axis is not None:
        axis = tuple(pystr_to_symbolic(a) for a in axis)
        axis = tuple(normalize_axes(axis, len(inp_shape)))
        if len(axis) > len(inp_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Axis {a} is out of bounds for data of dimension {d}".format(a=axis, d=inp_shape))
        for a in axis:
            if a >= len(inp_shape):
                raise mem_parser.DaceSyntaxError(
                    visitor, ast_node, "Axis {a} is out of bounds for data of dimension {d}".format(a=a, d=inp_shape))
        if keepdims:
            intermediate_shape = [d for i, d in enumerate(inp_shape) if i not in axis]
            expected_out_shape = [d if i not in axis else 1 for i, d in enumerate(inp_shape)]
        else:
            intermediate_shape = None
            expected_out_shape = [d for i, d in enumerate(inp_shape) if i not in axis]
        expected_out_shape = expected_out_shape or [1]
    else:
        if keepdims:
            intermediate_shape = [1]
            expected_out_shape = [1] * len(inp_shape)
        else:
            intermediate_shape = None
            expected_out_shape = [1]

    return axis, intermediate_shape, expected_out_shape


@oprepo.replaces_ufunc('reduce')
def implement_ufunc_reduce(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                           args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'reduce' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'keepdims' keyword
    keepdims = _validate_keepdims_kword(visitor, ast_node, ufunc_name, kwargs)

    # Validate 'axis' keyword
    axis, intermediate_shape, expected_out_shape = _validate_axis_kword(visitor, ast_node, sdfg, inputs, kwargs,
                                                                        keepdims)

    # Validate 'where' keyword
    # Throw a warning that it is currently unsupported.
    if 'where' in kwargs.keys():
        warnings.warn("Keyword argument 'where' in 'reduce' method of NumPy "
                      "ufunc calls is unsupported. It will be ignored.")

    # Validate data shapes and apply NumPy broadcasting rules
    # In the case of reduce we may only validate the broadcasting of the
    # single input with the 'where' value. Since 'where' is currently
    # unsupported, only validate output shape.
    # TODO: Maybe add special error when 'keepdims' is True
    if isinstance(outputs[0], str) and outputs[0] in sdfg.arrays.keys():
        out_shape = sdfg.arrays[outputs[0]].shape
        if len(out_shape) < len(expected_out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} does not have "
                "enough dimensions (output shape {o}, expected shape {e}).".format(f=ufunc_name,
                                                                                   o=out_shape,
                                                                                   e=expected_out_shape))
        if len(out_shape) > len(expected_out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} has too many "
                "dimensions (output shape {o}, expected shape {e}).".format(f=ufunc_name,
                                                                            o=out_shape,
                                                                            e=expected_out_shape))
        if (list(out_shape) != list(expected_out_shape)):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Output parameter for reduction operation {f} has non-reduction"
                " dimension not equal to the input one (output shape {o}, "
                "expected shape {e}).".format(f=ufunc_name, o=out_shape, e=expected_out_shape))
    else:
        out_shape = expected_out_shape

    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str):
        datadesc = sdfg.arrays[arg]
        result_type = datadesc.dtype
    elif isinstance(arg, (Number, np.bool_)):
        result_type = dtypes.DTYPE_TO_TYPECLASS[type(arg)]
    elif isinstance(arg, sp.Basic):
        result_type = _sym_type(arg)

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type, force_scalar=True)
    if keepdims:
        if (len(intermediate_shape) == 1 and intermediate_shape[0] == 1):
            intermediate_name = sdfg.temp_data_name()
            sdfg.add_scalar(intermediate_name, result_type, transient=True)
        else:
            intermediate_name, _ = sdfg.add_temp_transient(intermediate_shape, result_type)
    else:
        intermediate_name = outputs[0]

    # Validate 'initial' keyword
    # This is set to be ufunc.identity, when it exists
    initial = ufunc_impl['initial']
    if 'initial' in kwargs.keys():
        # NumPy documentation says that when 'initial' is set to None,
        # then the first element of the reduction is used. However, it seems
        # that when 'initial' is None and the ufunc has 'identity', then
        # ufunc.identity is the default.
        initial = kwargs['initial'] or initial
        if initial is None:
            if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
                inpdata = sdfg.arrays[inputs[0]]
                # In the input data has more than 1 dimensions and 'initial'
                # is None, then NumPy uses a different 'initial' value for every
                # non-reduced dimension.
                if isinstance(inpdata, data.Array):
                    state.add_mapped_tasklet(
                        name=state.label + "_reduce_initial",
                        map_ranges={
                            "__i{i}".format(i=i): "0:{s}".format(s=s)
                            for i, s in enumerate(inpdata.shape) if i not in axis
                        },
                        inputs={
                            "__inp":
                            dace.Memlet("{a}[{i}]".format(a=inputs[0],
                                                          i=','.join([
                                                              "0" if i in axis else "__i{i}".format(i=i)
                                                              for i in range(len(inpdata.shape))
                                                          ])))
                        },
                        outputs={
                            "__out":
                            dace.Memlet("{a}[{i}]".format(
                                a=intermediate_name,
                                i=','.join(["__i{i}".format(i=i) for i in range(len(inpdata.shape)) if i not in axis])))
                        },
                        code="__out = __inp",
                        external_edges=True)
                else:
                    r = state.add_read(inputs[0])
                    w = state.add_write(intermediate_name)
                    state.add.nedge(r, w, dace.Memlet.from_array(inputs[0], inpdata))
                state = visitor._add_state(state.label + 'b')
            else:
                initial = intermediate_name

    # Special case for infinity
    if np.isinf(initial):
        if np.sign(initial) < 0:
            initial = dtypes.min_value(result_type)
        else:
            initial = dtypes.max_value(result_type)

    # Create subgraph
    if isinstance(inputs[0], str) and inputs[0] in sdfg.arrays.keys():
        _reduce(visitor, sdfg, state, ufunc_impl['reduce'], inputs[0], intermediate_name, axis=axis, identity=initial)
    else:
        tasklet = state.add_tasklet(state.label + "_tasklet", {}, {'__out'}, "__out = {}".format(inputs[0]))
        out_node = state.add_write(intermediate_name)
        datadesc = sdfg.arrays[intermediate_name]
        state.add_edge(tasklet, '__out', out_node, None, dace.Memlet.from_array(intermediate_name, datadesc))

    if keepdims:
        intermediate_node = None
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == intermediate_name:
                intermediate_node = n
                break
        if not intermediate_node:
            raise ValueError("Keyword argument 'keepdims' is True, but "
                             "intermediate access node was not found.")
        out_node = state.add_write(outputs[0])
        state.add_nedge(intermediate_node, out_node, dace.Memlet.from_array(outputs[0], sdfg.arrays[outputs[0]]))

    return outputs


@oprepo.replaces_ufunc('accumulate')
def implement_ufunc_accumulate(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, state: SDFGState,
                               ufunc_name: str, args: Sequence[UfuncInput], kwargs: Dict[str,
                                                                                         Any]) -> List[UfuncOutput]:
    """ Implements the 'accumulate' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = 1
    num_outputs = 1
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # No casting needed
    arg = inputs[0]
    if isinstance(arg, str) and arg in sdfg.arrays.keys():
        datadesc = sdfg.arrays[arg]
        if not isinstance(datadesc, data.Array):
            raise mem_parser.DaceSyntaxError(visitor, ast_node,
                                             "Cannot accumulate on a dace.data.Scalar or dace.data.Stream.")
        out_shape = datadesc.shape
        result_type = datadesc.dtype
    else:
        raise mem_parser.DaceSyntaxError(visitor, ast_node, "Can accumulate only on a dace.data.Array.")

    # Validate 'axis' keyword argument
    axis = 0
    if 'axis' in kwargs.keys():
        axis = kwargs['axis'] or axis
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            axis = axis[0]
        if not isinstance(axis, Integral):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Value of keyword argument 'axis' in 'accumulate' method of {f}"
                " must be an integer (value {v}).".format(f=ufunc_name, v=axis))
        if axis >= len(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Axis {a} is out of bounds for dace.data.Array of dimension "
                "{l}".format(a=axis, l=len(out_shape)))
        # Normalize negative axis
        axis = normalize_axes([axis], len(out_shape))[0]

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Create subgraph
    shape = datadesc.shape
    map_range = {"__i{}".format(i): "0:{}".format(s) for i, s in enumerate(shape) if i != axis}
    input_idx = ','.join(["__i{}".format(i) if i != axis else "0:{}".format(shape[i]) for i in range(len(shape))])
    output_idx = ','.join(["__i{}".format(i) if i != axis else "0:{}".format(shape[i]) for i in range(len(shape))])

    nested_sdfg = dace.SDFG(state.label + "_for_loop")
    nested_sdfg._temp_transients = sdfg._temp_transients
    inpconn = nested_sdfg.temp_data_name()
    outconn = nested_sdfg.temp_data_name()
    shape = [datadesc.shape[axis]]
    strides = [datadesc.strides[axis]]
    nested_sdfg.add_array(inpconn, shape, result_type, strides=strides)
    nested_sdfg.add_array(outconn, shape, result_type, strides=strides)

    init_state = nested_sdfg.add_state(label="init")
    r = init_state.add_read(inpconn)
    w = init_state.add_write(outconn)
    init_state.add_nedge(r, w, dace.Memlet("{a}[{i}] -> {oi}".format(a=inpconn, i='0', oi='0')))

    body_state = nested_sdfg.add_state(label="body")
    r1 = body_state.add_read(inpconn)
    r2 = body_state.add_read(outconn)
    w = body_state.add_write(outconn)
    t = body_state.add_tasklet(name=state.label + "_for_loop_tasklet",
                               inputs=ufunc_impl['inputs'],
                               outputs=ufunc_impl['outputs'],
                               code=ufunc_impl['code'])

    loop_idx = "__i{}".format(axis)
    loop_idx_m1 = "__i{} - 1".format(axis)
    body_state.add_edge(r1, None, t, '__in1', dace.Memlet("{a}[{i}]".format(a=inpconn, i=loop_idx)))
    body_state.add_edge(r2, None, t, '__in2', dace.Memlet("{a}[{i}]".format(a=outconn, i=loop_idx_m1)))
    body_state.add_edge(t, '__out', w, None, dace.Memlet("{a}[{i}]".format(a=outconn, i=loop_idx)))

    init_expr = str(1)
    cond_expr = "__i{i} < {s}".format(i=axis, s=shape[0])
    incr_expr = "__i{} + 1".format(axis)
    nested_sdfg.add_loop(init_state, body_state, None, loop_idx, init_expr, cond_expr, incr_expr)

    sdfg._temp_transients = nested_sdfg._temp_transients

    r = state.add_read(inputs[0])
    w = state.add_write(outputs[0])
    codenode = state.add_nested_sdfg(nested_sdfg, sdfg, {inpconn}, {outconn})
    me, mx = state.add_map(state.label + '_map', map_range)
    state.add_memlet_path(r,
                          me,
                          codenode,
                          memlet=dace.Memlet("{a}[{i}]".format(a=inputs[0], i=input_idx)),
                          dst_conn=inpconn)
    state.add_memlet_path(codenode,
                          mx,
                          w,
                          memlet=dace.Memlet("{a}[{i}]".format(a=outputs[0], i=output_idx)),
                          src_conn=outconn)

    return outputs


@oprepo.replaces_ufunc('outer')
def implement_ufunc_outer(visitor: 'ProgramVisitor', ast_node: ast.Call, sdfg: SDFG, state: SDFGState, ufunc_name: str,
                          args: Sequence[UfuncInput], kwargs: Dict[str, Any]) -> List[UfuncOutput]:
    """ Implements the 'outer' method of a NumPy ufunc.

        :param visitor: ProgramVisitor object handling the ufunc call
        :param ast_node: AST node corresponding to the ufunc call
        :param sdfg: SDFG object
        :param state: SDFG State object
        :param ufunc_name: Name of the ufunc
        :param args: Positional arguments of the ufunc call
        :param kwargs: Keyword arguments of the ufunc call

        :raises DaCeSyntaxError: When validation fails

        :returns: List of output datanames
    """

    # Flatten arguments
    args = _flatten_args(args)

    # Get the ufunc implementation details
    ufunc_impl = _get_ufunc_impl(visitor, ast_node, ufunc_name)

    # Validate number of arguments, inputs, and outputs
    num_inputs = len(ufunc_impl['inputs'])
    num_outputs = len(ufunc_impl['outputs'])
    num_args = len(args)
    _validate_ufunc_num_arguments(visitor, ast_node, ufunc_name, num_inputs, num_outputs, num_args)
    inputs = _validate_ufunc_inputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_args, args)
    outputs = _validate_ufunc_outputs(visitor, ast_node, sdfg, ufunc_name, num_inputs, num_outputs, num_args, args,
                                      kwargs)

    # Validate 'where' keyword
    has_where, where = _validate_where_kword(visitor, ast_node, sdfg, ufunc_name, kwargs)

    # Validate data shapes
    out_shape = []
    map_vars = []
    map_range = dict()
    input_indices = []
    output_idx = None
    for i, arg in enumerate(inputs):
        if isinstance(arg, str) and arg in sdfg.arrays.keys():
            datadesc = sdfg.arrays[arg]
            if isinstance(datadesc, data.Scalar):
                input_idx = '0'
            elif isinstance(datadesc, data.Array):
                shape = datadesc.shape
                out_shape.extend(shape)
                map_vars.extend(["__i{i}_{j}".format(i=i, j=j) for j in range(len(shape))])
                map_range.update({"__i{i}_{j}".format(i=i, j=j): "0:{}".format(sz) for j, sz in enumerate(shape)})
                input_idx = ','.join(["__i{i}_{j}".format(i=i, j=j) for j in range(len(shape))])
                if output_idx:
                    output_idx = ','.join([output_idx, input_idx])
                else:
                    output_idx = input_idx
            else:
                raise mem_parser.DaceSyntaxError(
                    visitor, ast_node, "Unsuported data type {t} in 'outer' method of NumPy ufunc "
                    "{f}.".format(t=type(datadesc), f=ufunc_name))
        elif isinstance(arg, (Number, sp.Basic)):
            input_idx = None
        input_indices.append(input_idx)

    if has_where and not isinstance(where, (bool, np.bool_)):
        where_shape = sdfg.arrays[where].shape
        try:
            bcast_out_shape, _, _, bcast_inp_indices = _broadcast([out_shape, where_shape])
        except SyntaxError:
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "'where' shape {w} could not be broadcast together with 'out' "
                "shape {o}.".format(w=where_shape, o=out_shape))
        if list(bcast_out_shape) != list(out_shape):
            raise mem_parser.DaceSyntaxError(
                visitor, ast_node, "Broadcasting 'where' shape {w} together with expected 'out' "
                "shape {o} resulted in a different output shape {no}. This is "
                "currently unsupported.".format(w=where_shape, o=out_shape, no=bcast_out_shape))
        where_idx = bcast_inp_indices[1]
        for i in range(len(out_shape)):
            where_idx = where_idx.replace("__i{}".format(i), map_vars[i])
        input_indices.append(where_idx)
    else:
        input_indices.append(None)

    # Infer result type
    result_type, casting = _result_type(
        [sdfg.arrays[arg] if isinstance(arg, str) and arg in sdfg.arrays else arg for arg in inputs],
        ufunc_impl['operator'])
    if 'dtype' in kwargs.keys():
        dtype = kwargs['dtype']
        if dtype in dtypes.DTYPE_TO_TYPECLASS.keys():
            result_type = dtype

    # Create output data (if needed)
    outputs = _create_output(sdfg, inputs, outputs, out_shape, result_type)

    # Set tasklet parameters
    tasklet_params = _set_tasklet_params(ufunc_impl, inputs, casting=casting)

    # Create subgraph
    _create_subgraph(visitor,
                     sdfg,
                     state,
                     inputs,
                     outputs,
                     map_range,
                     input_indices,
                     output_idx,
                     out_shape,
                     tasklet_params,
                     has_where=has_where,
                     where=where)

    return outputs


@oprepo.replaces('numpy.reshape')
def reshape(pv: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            arr: str,
            newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
            order='C') -> str:
    if isinstance(arr, (list, tuple)) and len(arr) == 1:
        arr = arr[0]
    desc = sdfg.arrays[arr]

    # "order" determines stride orders
    fortran_strides = False
    if order == 'F' or (order == 'A' and desc.strides[0] == 1):
        # FORTRAN strides
        fortran_strides = True

    # New shape and strides as symbolic expressions
    newshape = [symbolic.pystr_to_symbolic(s) for s in newshape]
    if fortran_strides:
        strides = [data._prod(newshape[:i]) for i in range(len(newshape))]
    else:
        strides = [data._prod(newshape[i + 1:]) for i in range(len(newshape))]

    newarr, newdesc = sdfg.add_view(arr,
                                    newshape,
                                    desc.dtype,
                                    storage=desc.storage,
                                    strides=strides,
                                    allow_conflicts=desc.allow_conflicts,
                                    total_size=desc.total_size,
                                    may_alias=desc.may_alias,
                                    alignment=desc.alignment,
                                    find_new_name=True)

    # Register view with DaCe program visitor
    aset = subsets.Range.from_array(desc)
    vset = subsets.Range.from_array(newdesc)
    pv.views[newarr] = (arr, Memlet(data=arr, subset=aset, other_subset=vset))

    return newarr


@oprepo.replaces_method('Array', 'view')
@oprepo.replaces_method('Scalar', 'view')
@oprepo.replaces_method('View', 'view')
def view(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, dtype, type=None) -> str:
    if type is not None:
        raise ValueError('View to numpy types is not supported')

    desc = sdfg.arrays[arr]

    # Change size of array based on the differences in bytes
    bytemult = desc.dtype.bytes / dtype.bytes
    bytediv = dtype.bytes / desc.dtype.bytes
    contigdim = next(i for i, s in enumerate(desc.strides) if s == 1)

    # For cases that can be recognized, if contiguous dimension is too small
    # raise an exception similar to numpy
    if (not issymbolic(desc.shape[contigdim], sdfg.constants) and bytemult < 1
            and desc.shape[contigdim] % bytediv != 0):
        raise ValueError('When changing to a larger dtype, its size must be a divisor of '
                         'the total size in bytes of the last axis of the array.')

    # Create new shape and strides for view
    newshape = list(desc.shape)
    newstrides = [s * bytemult if i != contigdim else s for i, s in enumerate(desc.strides)]
    newshape[contigdim] *= bytemult

    newarr, _ = sdfg.add_view(arr,
                              newshape,
                              dtype,
                              storage=desc.storage,
                              strides=newstrides,
                              allow_conflicts=desc.allow_conflicts,
                              total_size=desc.total_size * bytemult,
                              may_alias=desc.may_alias,
                              alignment=desc.alignment,
                              find_new_name=True)

    # Register view with DaCe program visitor
    # NOTE: We do not create here a Memlet of the form `A[subset] -> osubset`
    # because the View can be of a different dtype. Adding `other_subset` in
    # such cases will trigger validation error.
    pv.views[newarr] = (arr, Memlet.from_array(arr, desc))

    return newarr


@oprepo.replaces_attribute('Array', 'size')
@oprepo.replaces_attribute('Scalar', 'size')
@oprepo.replaces_attribute('View', 'size')
def size(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> Size:
    desc = sdfg.arrays[arr]
    totalsize = data._prod(desc.shape)
    return totalsize


@oprepo.replaces_attribute('Array', 'flat')
@oprepo.replaces_attribute('Scalar', 'flat')
@oprepo.replaces_attribute('View', 'flat')
def flat(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, order: str = 'C') -> str:
    desc = sdfg.arrays[arr]
    totalsize = data._prod(desc.shape)
    if order not in ('C', 'F'):
        raise NotImplementedError(f'Order "{order}" not yet supported for flattening')

    if order == 'C':
        contig_strides = tuple(data._prod(desc.shape[i + 1:]) for i in range(len(desc.shape)))
    elif order == 'F':
        contig_strides = tuple(data._prod(desc.shape[:i]) for i in range(len(desc.shape)))

    if desc.total_size != totalsize or desc.strides != contig_strides:
        # If data is not contiguous (numpy standard), create copy as explicit map
        # warnings.warn(f'Generating explicit copy for non-contiguous array "{arr}"')
        newarr, _ = sdfg.add_array(arr, [totalsize],
                                   desc.dtype,
                                   storage=desc.storage,
                                   strides=[1],
                                   allow_conflicts=desc.allow_conflicts,
                                   total_size=totalsize,
                                   may_alias=desc.may_alias,
                                   alignment=desc.alignment,
                                   transient=True,
                                   find_new_name=True)
        maprange = {f'__i{i}': (0, s - 1, 1) for i, s in enumerate(desc.shape)}
        out_index = sum(symbolic.pystr_to_symbolic(f'__i{i}') * s for i, s in enumerate(contig_strides))
        state.add_mapped_tasklet(
            'flat',
            maprange,
            dict(__inp=Memlet(data=arr, subset=','.join(maprange.keys()))),
            '__out = __inp',
            dict(__out=Memlet(data=newarr, subset=subsets.Range([(out_index, out_index, 1)]))),
            external_edges=True,
        )
    else:
        newarr, newdesc = sdfg.add_view(arr, [totalsize],
                                        desc.dtype,
                                        storage=desc.storage,
                                        strides=[1],
                                        allow_conflicts=desc.allow_conflicts,
                                        total_size=totalsize,
                                        may_alias=desc.may_alias,
                                        alignment=desc.alignment,
                                        find_new_name=True)
        # Register view with DaCe program visitor
        aset = subsets.Range.from_array(desc)
        vset = subsets.Range.from_array(newdesc)
        pv.views[newarr] = (arr, Memlet(data=arr, subset=aset, other_subset=vset))

    return newarr


@oprepo.replaces_attribute('Array', 'T')
@oprepo.replaces_attribute('View', 'T')
def _ndarray_T(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _transpose(pv, sdfg, state, arr)


@oprepo.replaces_attribute('Array', 'real')
@oprepo.replaces_attribute('Scalar', 'real')
@oprepo.replaces_attribute('View', 'real')
def _ndarray_real(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _real(pv, sdfg, state, arr)


@oprepo.replaces_attribute('Array', 'imag')
@oprepo.replaces_attribute('Scalar', 'imag')
@oprepo.replaces_attribute('View', 'imag')
def _ndarray_imag(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _imag(pv, sdfg, state, arr)


@oprepo.replaces_method('Array', 'copy')
@oprepo.replaces_method('Scalar', 'copy')
@oprepo.replaces_method('View', 'copy')
def _ndarray_copy(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _numpy_copy(pv, sdfg, state, arr)


@oprepo.replaces_method('Array', 'fill')
@oprepo.replaces_method('Scalar', 'fill')
@oprepo.replaces_method('View', 'fill')
def _ndarray_fill(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, value: Number) -> str:
    if not isinstance(value, (Number, np.bool_)):
        raise mem_parser.DaceSyntaxError(pv, None, "Fill value {f} must be a number!".format(f=value))
    return _elementwise(pv, sdfg, state, "lambda x: {}".format(value), arr, arr)


@oprepo.replaces_method('Array', 'reshape')
@oprepo.replaces_method('View', 'reshape')
def _ndarray_reshape(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     arr: str,
                     newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
                     order='C') -> str:
    return reshape(pv, sdfg, state, arr, newshape, order)


@oprepo.replaces_method('Array', 'transpose')
@oprepo.replaces_method('View', 'transpose')
def _ndarray_transpose(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, *axes) -> str:
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1:
        axes = axes[0]
    return _transpose(pv, sdfg, state, arr, axes)


@oprepo.replaces_method('Array', 'flatten')
@oprepo.replaces_method('Scalar', 'flatten')
@oprepo.replaces_method('View', 'flatten')
def _ndarray_flatten(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, order: str = 'C') -> str:
    new_arr = flat(pv, sdfg, state, arr, order)
    # `flatten` always returns a copy
    if isinstance(new_arr, data.View):
        return _ndarray_copy(pv, sdfg, state, new_arr)
    return new_arr


@oprepo.replaces_method('Array', 'ravel')
@oprepo.replaces_method('Scalar', 'ravel')
@oprepo.replaces_method('View', 'ravel')
def _ndarray_ravel(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, order: str = 'C') -> str:
    # `ravel` returns a copy only when necessary (sounds like ndarray.flat)
    return flat(pv, sdfg, state, arr, order)


@oprepo.replaces_method('Array', 'max')
@oprepo.replaces_method('Scalar', 'max')
@oprepo.replaces_method('View', 'max')
def _ndarray_max(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'maximum', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'min')
@oprepo.replaces_method('Scalar', 'min')
@oprepo.replaces_method('View', 'min')
def _ndarray_min(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'minimum', [arr], kwargs)[0]


# TODO: It looks like `_argminmax` does not work with a flattened array.
# @oprepo.replaces_method('Array', 'argmax')
# @oprepo.replaces_method('Scalar', 'argmax')
# @oprepo.replaces_method('View', 'argmax')
# def _ndarray_argmax(pv: 'ProgramVisitor',
#                  sdfg: SDFG,
#                  state: SDFGState,
#                  arr: str,
#                  axis: int = None,
#                  out: str = None) -> str:
#     if not axis:
#         axis = 0
#         arr = flat(pv, sdfg, state, arr)
#     nest, newarr = _argmax(pv, sdfg, state, arr, axis)
#     if out:
#         r = state.add_read(arr)
#         w = state.add_read(newarr)
#         state.add_nedge(r, w, dace.Memlet.from_array(newarr, sdfg.arrays[newarr]))
#     return new_arr


@oprepo.replaces_method('Array', 'conj')
@oprepo.replaces_method('Scalar', 'conj')
@oprepo.replaces_method('View', 'conj')
def _ndarray_conj(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return implement_ufunc(pv, None, sdfg, state, 'conj', [arr], {})[0]


@oprepo.replaces_method('Array', 'sum')
@oprepo.replaces_method('Scalar', 'sum')
@oprepo.replaces_method('View', 'sum')
def _ndarray_sum(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'add', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'mean')
@oprepo.replaces_method('Scalar', 'mean')
@oprepo.replaces_method('View', 'mean')
def _ndarray_mean(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    nest = NestedCall(pv, sdfg, state)
    kwargs = kwargs or dict(axis=None)
    sumarr = implement_ufunc_reduce(pv, None, sdfg, nest.add_state(), 'add', [arr], kwargs)[0]
    desc = sdfg.arrays[arr]
    sz = reduce(lambda x, y: x * y, desc.shape)
    return nest, _elementwise(pv, sdfg, nest.add_state(), "lambda x: x / {}".format(sz), sumarr)


@oprepo.replaces_method('Array', 'prod')
@oprepo.replaces_method('Scalar', 'prod')
@oprepo.replaces_method('View', 'prod')
def _ndarray_prod(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'multiply', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'all')
@oprepo.replaces_method('Scalar', 'all')
@oprepo.replaces_method('View', 'all')
def _ndarray_all(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'logical_and', [arr], kwargs)[0]


@oprepo.replaces_method('Array', 'any')
@oprepo.replaces_method('Scalar', 'any')
@oprepo.replaces_method('View', 'any')
def _ndarray_any(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, kwargs: Dict[str, Any] = None) -> str:
    kwargs = kwargs or dict(axis=None)
    return implement_ufunc_reduce(pv, None, sdfg, state, 'logical_or', [arr], kwargs)[0]


# Datatype converter #########################################################


def _make_datatype_converter(typeclass: str):
    if typeclass == "bool":
        dtype = dace.bool
    elif typeclass in {"int", "float", "complex"}:
        dtype = dtypes.DTYPE_TO_TYPECLASS[eval(typeclass)]
    else:
        dtype = dtypes.DTYPE_TO_TYPECLASS[eval("np.{}".format(typeclass))]

    @oprepo.replaces(typeclass)
    @oprepo.replaces("dace.{}".format(typeclass))
    @oprepo.replaces("numpy.{}".format(typeclass))
    def _converter(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arg: UfuncInput):
        return _datatype_converter(sdfg, state, arg, dtype=dtype)


for typeclass in dtypes.TYPECLASS_STRINGS:
    _make_datatype_converter(typeclass)


def _datatype_converter(sdfg: SDFG, state: SDFGState, arg: UfuncInput, dtype: dtypes.typeclass) -> UfuncOutput:
    """ Out-of-place datatype conversion of the input argument.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param arg: Input argument
        :param dtype: Datatype to convert input argument into

        :returns: dace.data.Array of same size as input or dace.data.Scalar
    """

    # Get shape and indices
    (out_shape, map_indices, out_indices, inp_indices) = _validate_shapes(None, None, sdfg, None, [arg], [None])

    # Create output data
    outputs = _create_output(sdfg, [arg], [None], out_shape, dtype)

    # Set tasklet parameters
    impl = {
        'name': "_convert_to_{}_".format(dtype.to_string()),
        'inputs': ['__inp'],
        'outputs': ['__out'],
        'code': "__out = dace.{}(__inp)".format(dtype.to_string())
    }
    tasklet_params = _set_tasklet_params(impl, [arg])

    # Visitor input only needed when `has_where == True`.
    _create_subgraph(None,
                     sdfg,
                     state, [arg],
                     outputs,
                     map_indices,
                     inp_indices,
                     out_indices,
                     out_shape,
                     tasklet_params,
                     has_where=False,
                     where=None)

    return outputs


@oprepo.replaces_method('Array', 'astype')
@oprepo.replaces_method('Scalar', 'astype')
@oprepo.replaces_method('View', 'astype')
def _ndarray_astype(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, dtype: dace.typeclass) -> str:
    if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
        dtype = dtypes.typeclass(dtype)
    return _datatype_converter(sdfg, state, arr, dtype)[0]


# Replacements that need ufuncs ###############################################
# TODO: Fix by separating to different modules and importing


@oprepo.replaces('dace.dot')
@oprepo.replaces('numpy.dot')
def dot(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op_a: str, op_b: str, op_out=None):

    # TODO: Add support for dot(N-D, 1-D) and dot(N-D, M-D) cases.
    # See https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    # TODO: Add/improve validation

    for op in (op_a, op_b):
        if not isinstance(op, str) or not op in sdfg.arrays.keys():
            raise SyntaxError()

    arr_a = sdfg.arrays[op_a]
    arr_b = sdfg.arrays[op_b]

    if len(arr_a.shape) == 2 and len(arr_b.shape) == 2:
        # Matrix multiplication
        # TODO: `If op_out`, then this is not correct. We need np.matmult,
        # but it is not implemented yet
        return _matmult(pv, sdfg, state, op_a, op_b)

    if (isinstance(arr_a, data.Scalar) or list(arr_a.shape) == [1] or isinstance(arr_b, data.Scalar)
            or list(arr_b.shape) == [1]):
        # Case dot(N-D, 0-D), intepreted as np.multiply(a, b)
        node = ast.Call()
        ufunc_name = 'multiply'
        args = [op_a, op_b]
        if op_out:
            args.append(op_out)
        return ufunc_impl(pv, node, ufunc_name, sdfg, state, args)

    if len(arr_a.shape) > 2 or len(arr_b.shape) > 2:
        raise NotImplementedError

    if arr_a.shape[0] != arr_b.shape[0]:
        raise SyntaxError()

    if op_out:
        if not isinstance(op_out, str) or not op_out in sdfg.arrays.keys():
            raise SyntaxError()
    else:
        # Infer result type
        restype, _ = _result_type([arr_a, arr_b], 'Mul')
        op_out = sdfg.temp_data_name()
        sdfg.add_scalar(op_out, restype, transient=True, storage=arr_a.storage)

    arr_out = sdfg.arrays[op_out]

    from dace.libraries.blas.nodes.dot import Dot  # Avoid import loop

    acc_a = state.add_read(op_a)
    acc_b = state.add_read(op_b)
    acc_out = state.add_write(op_out)

    tasklet = Dot('_Dot_')
    state.add_node(tasklet)
    state.add_edge(acc_a, None, tasklet, '_x', dace.Memlet.from_array(op_a, arr_a))
    state.add_edge(acc_b, None, tasklet, '_y', dace.Memlet.from_array(op_b, arr_b))
    state.add_edge(tasklet, '_result', acc_out, None, dace.Memlet.from_array(op_out, arr_out))

    return op_out


# NumPy linalg replacements ###################################################


@oprepo.replaces('dace.linalg.inv')
@oprepo.replaces('numpy.linalg.inv')
def _inv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, inp_op: str):

    if not isinstance(inp_op, str) or not inp_op in sdfg.arrays.keys():
        raise SyntaxError()

    inp_arr = sdfg.arrays[inp_op]
    out_arr = sdfg.add_temp_transient(inp_arr.shape, inp_arr.dtype, storage=inp_arr.storage)

    from dace.libraries.linalg import Inv

    inp = state.add_read(inp_op)
    out = state.add_write(out_arr[0])
    inv_node = Inv("inv", overwrite_a=False, use_getri=True)

    state.add_memlet_path(inp, inv_node, dst_conn="_ain", memlet=Memlet.from_array(inp_op, inp_arr))
    state.add_memlet_path(inv_node, out, src_conn="_aout", memlet=Memlet.from_array(*out_arr))

    return out_arr[0]


@oprepo.replaces('dace.linalg.solve')
@oprepo.replaces('numpy.linalg.solve')
def _solve(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, op_a: str, op_b: str):

    for op in (op_a, op_b):
        if not isinstance(op, str) or not op in sdfg.arrays.keys():
            raise SyntaxError()

    a_arr = sdfg.arrays[op_a]
    b_arr = sdfg.arrays[op_b]
    out_arr = sdfg.add_temp_transient(b_arr.shape, b_arr.dtype, storage=b_arr.storage)

    from dace.libraries.linalg import Solve

    a_inp = state.add_read(op_a)
    b_inp = state.add_read(op_b)
    out = state.add_write(out_arr[0])
    solve_node = Solve("solve")

    state.add_memlet_path(a_inp, solve_node, dst_conn="_ain", memlet=Memlet.from_array(op_a, a_arr))
    state.add_memlet_path(b_inp, solve_node, dst_conn="_bin", memlet=Memlet.from_array(op_b, b_arr))
    state.add_memlet_path(solve_node, out, src_conn="_bout", memlet=Memlet.from_array(*out_arr))

    return out_arr[0]


@oprepo.replaces('dace.linalg.cholesky')
@oprepo.replaces('numpy.linalg.cholesky')
def _inv(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, inp_op: str):

    if not isinstance(inp_op, str) or not inp_op in sdfg.arrays.keys():
        raise SyntaxError()

    inp_arr = sdfg.arrays[inp_op]
    out_arr = sdfg.add_temp_transient(inp_arr.shape, inp_arr.dtype, storage=inp_arr.storage)

    from dace.libraries.linalg import Cholesky

    inp = state.add_read(inp_op)
    out = state.add_write(out_arr[0])
    chlsky_node = Cholesky("cholesky", lower=True)

    state.add_memlet_path(inp, chlsky_node, dst_conn="_a", memlet=Memlet.from_array(inp_op, inp_arr))
    state.add_memlet_path(chlsky_node, out, src_conn="_b", memlet=Memlet.from_array(*out_arr))

    return out_arr[0]


# CuPy replacements


@oprepo.replaces("cupy._core.core.ndarray")
@oprepo.replaces("cupy.ndarray")
def _define_cupy_local(
    pv: "ProgramVisitor",
    sdfg: SDFG,
    state: SDFGState,
    shape: Shape,
    dtype: typeclass,
):
    """Defines a local array in a DaCe program."""
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    name, _ = sdfg.add_temp_transient(shape, dtype, storage=dtypes.StorageType.GPU_Global)
    return name


@oprepo.replaces('cupy.full')
def _cupy_full(pv: 'ProgramVisitor',
               sdfg: SDFG,
               state: SDFGState,
               shape: Shape,
               fill_value: Union[sp.Expr, Number],
               dtype: dace.typeclass = None):
    """ Creates and array of the specified shape and initializes it with
        the fill value.
    """
    if isinstance(fill_value, (Number, np.bool_)):
        vtype = dtypes.DTYPE_TO_TYPECLASS[type(fill_value)]
    elif isinstance(fill_value, sp.Expr):
        vtype = _sym_type(fill_value)
    else:
        raise mem_parser.DaceSyntaxError(pv, None, "Fill value {f} must be a number!".format(f=fill_value))
    dtype = dtype or vtype
    name, _ = sdfg.add_temp_transient(shape, dtype, storage=dtypes.StorageType.GPU_Global)

    state.add_mapped_tasklet(
        '_cupy_full_', {"__i{}".format(i): "0: {}".format(s)
                        for i, s in enumerate(shape)}, {},
        "__out = {}".format(fill_value),
        dict(__out=dace.Memlet.simple(name, ",".join(["__i{}".format(i) for i in range(len(shape))]))),
        external_edges=True)

    return name


@oprepo.replaces('cupy.zeros')
def _cupy_zeros(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dace.typeclass = dace.float64):
    """ Creates and array of the specified shape and initializes it with zeros.
    """
    return _cupy_full(pv, sdfg, state, shape, 0.0, dtype)


@oprepo.replaces('cupy.empty_like')
def _cupy_empty_like(pv: 'ProgramVisitor',
                     sdfg: SDFG,
                     state: SDFGState,
                     prototype: str,
                     dtype: dace.typeclass = None,
                     shape: Shape = None):
    if prototype not in sdfg.arrays.keys():
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=prototype))
    desc = sdfg.arrays[prototype]
    name, newdesc = sdfg.add_temp_transient_like(desc)
    if dtype is not None:
        newdesc.dtype = dtype
    if shape is not None:
        newdesc.shape = shape
    return name


@oprepo.replaces('cupy.empty')
@oprepo.replaces('cupy_empty')
def _cupy_empty(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, shape: Shape, dtype: dace.typeclass):
    """ Creates an unitialized array of the specificied shape and dtype. """
    return _define_cupy_local(pv, sdfg, state, shape, dtype)
