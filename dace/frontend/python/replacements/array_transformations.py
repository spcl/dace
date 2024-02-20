# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for N-dimensional array transformations.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor, UfuncInput, UfuncOutput
import dace.frontend.python.memlet_parser as mem_parser
from dace import data, dtypes, subsets, symbolic
from dace import Memlet, SDFG, SDFGState

from typing import Tuple, Union

import numpy as np

@oprepo.replaces('numpy.copy')
def _numpy_copy(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str):
    """ Creates a copy of array a.
    """
    if a not in sdfg.arrays.keys():
        raise DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=a))
    # TODO: The whole AddTransientMethod class should be move in replacements.py
    from dace.frontend.python.newast import _add_transient_data
    name, desc = _add_transient_data(sdfg, sdfg.arrays[a])
    rnode = state.add_read(a)
    wnode = state.add_write(name)
    state.add_nedge(rnode, wnode, Memlet.from_array(name, desc))
    return name


@oprepo.replaces_method('Array', 'copy')
@oprepo.replaces_method('Scalar', 'copy')
@oprepo.replaces_method('View', 'copy')
def _ndarray_copy(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _numpy_copy(pv, sdfg, state, arr)


@oprepo.replaces('numpy.flip')
def _numpy_flip(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, axis=None):
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
def _numpy_rot90(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, k=1, axes=(0, 1)):
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


@oprepo.replaces('transpose')
@oprepo.replaces('dace.transpose')
@oprepo.replaces('numpy.transpose')
def _transpose(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, inpname: str, axes=None):

    arr1 = sdfg.arrays[inpname]

    # Reversed list
    if axes is None:
        axes = tuple(range(len(arr1.shape) - 1, -1, -1))
    else:
        if len(axes) != len(arr1.shape) or sorted(axes) != list(range(len(arr1.shape))):
            raise ValueError("axes don't match array")
        axes = tuple(axes)

    if axes == (0, ):  # Special (degenerate) case for 1D "transposition"
        return inpname

    restype = arr1.dtype
    new_shape = [arr1.shape[i] for i in axes]
    outname, arr2 = sdfg.add_temp_transient(new_shape, restype, arr1.storage)

    if axes == (1, 0):  # Special case for 2D transposition
        acc1 = state.add_read(inpname)
        acc2 = state.add_write(outname)
        import dace.libraries.standard  # Avoid import loop
        tasklet = dace.libraries.standard.Transpose('_Transpose_', restype)
        state.add_node(tasklet)
        state.add_edge(acc1, None, tasklet, '_inp', Memlet.from_array(inpname, arr1))
        state.add_edge(tasklet, '_out', acc2, None, Memlet.from_array(outname, arr2))
    else:  # Tensor transpose
        modes = len(arr1.shape)
        idx = axes.index(0)
        # Special case of tensor transposition: matrix transpose + reshape
        if axes[idx:] == list(range(modes - idx)) and axes[:idx] == list(range(axes[-1] + 1, modes)):
            rows = data._prod([arr1.shape[axes[i]] for i in range(idx, len(arr1.shape))])
            cols = data._prod([arr1.shape[axes[i]] for i in range(idx)])
            matrix = _ndarray_reshape(pv, sdfg, state, inpname, [rows, cols])
            trans_matrix = _transpose(pv, sdfg, state, matrix)
            return _ndarray_reshape(pv, sdfg, state, trans_matrix, [arr1.shape[i] for i in axes])

        read = state.add_read(inpname)
        write = state.add_write(outname)
        from dace.libraries.standard import TensorTranspose
        tasklet = TensorTranspose('_TensorTranspose', axes or list(range(len(arr1.shape))))
        state.add_node(tasklet)
        state.add_edge(read, None, tasklet, '_inp_tensor', Memlet.from_array(inpname, arr1))
        state.add_edge(tasklet, '_out_tensor', write, None, Memlet.from_array(outname, arr2))

    return outname


@oprepo.replaces_method('Array', 'transpose')
@oprepo.replaces_method('View', 'transpose')
def _ndarray_transpose(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, *axes) -> str:
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1:
        axes = axes[0]
    return _transpose(pv, sdfg, state, arr, axes)


@oprepo.replaces('numpy.reshape')
def reshape(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    arr: str,
    newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
    order: StringLiteral = StringLiteral('C')
) -> str:
    if isinstance(arr, (list, tuple)) and len(arr) == 1:
        arr = arr[0]
    desc = sdfg.arrays[arr]

    # "order" determines stride orders
    order = str(order)
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


@oprepo.replaces_method('Array', 'reshape')
@oprepo.replaces_method('View', 'reshape')
def _ndarray_reshape(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    arr: str,
    newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
    order: StringLiteral = StringLiteral('C')
) -> str:
    return reshape(pv, sdfg, state, arr, newshape, order)


@oprepo.replaces_method('Array', 'flatten')
@oprepo.replaces_method('Scalar', 'flatten')
@oprepo.replaces_method('View', 'flatten')
def _ndarray_flatten(pv: ProgramVisitor,
                     sdfg: SDFG,
                     state: SDFGState,
                     arr: str,
                     order: StringLiteral = StringLiteral('C')) -> str:
    new_arr = flat(pv, sdfg, state, arr, order)
    # `flatten` always returns a copy
    if isinstance(new_arr, data.View):
        return _ndarray_copy(pv, sdfg, state, new_arr)
    return new_arr


@oprepo.replaces_method('Array', 'ravel')
@oprepo.replaces_method('Scalar', 'ravel')
@oprepo.replaces_method('View', 'ravel')
def _ndarray_ravel(pv: ProgramVisitor,
                   sdfg: SDFG,
                   state: SDFGState,
                   arr: str,
                   order: StringLiteral = StringLiteral('C')) -> str:
    # `ravel` returns a copy only when necessary (sounds like ndarray.flat)
    return flat(pv, sdfg, state, arr, order)


@oprepo.replaces_method('Array', 'view')
@oprepo.replaces_method('Scalar', 'view')
@oprepo.replaces_method('View', 'view')
def view(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, dtype, type=None) -> str:
    if type is not None:
        raise ValueError('View to numpy types is not supported')

    desc = sdfg.arrays[arr]

    # Change size of array based on the differences in bytes
    bytemult = desc.dtype.bytes / dtype.bytes
    bytediv = dtype.bytes / desc.dtype.bytes
    contigdim = next(i for i, s in enumerate(desc.strides) if s == 1)

    # For cases that can be recognized, if contiguous dimension is too small
    # raise an exception similar to numpy
    if (not symbolic.issymbolic(desc.shape[contigdim], sdfg.constants) and bytemult < 1
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


@oprepo.replaces_attribute('Array', 'flat')
@oprepo.replaces_attribute('Scalar', 'flat')
@oprepo.replaces_attribute('View', 'flat')
def flat(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, order: StringLiteral = StringLiteral('C')) -> str:
    desc = sdfg.arrays[arr]
    order = str(order)
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
def _ndarray_T(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _transpose(pv, sdfg, state, arr)


###############################################################################
# Type conversion
###############################################################################


def _make_datatype_converter(typeclass: str):
    if typeclass == "bool":
        dtype = dtypes.bool
    elif typeclass in {"int", "float", "complex"}:
        dtype = dtypes.dtype_to_typeclass(eval(typeclass))
    else:
        dtype = dtypes.dtype_to_typeclass(getattr(np, typeclass))

    @oprepo.replaces(typeclass)
    @oprepo.replaces("dace.{}".format(typeclass))
    @oprepo.replaces("numpy.{}".format(typeclass))
    def _converter(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput):
        return _datatype_converter(sdfg, state, arg, dtype=dtype)


for typeclass in dtypes.TYPECLASS_STRINGS:
    _make_datatype_converter(typeclass)


def _datatype_converter(sdfg: SDFG, state: SDFGState, arg: UfuncInput, dtype: dtypes.typeclass) -> UfuncOutput:
    """ Out-of-place datatype conversion of the input argument.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param arg: Input argument
        :param dtype: Datatype to convert input argument into

        :return: ``dace.data.Array`` of same size as input or ``dace.data.Scalar``
    """
    from dace.frontend.python.replacements import ufunc

    # Get shape and indices
    (out_shape, map_indices, out_indices, inp_indices) = ufunc._validate_shapes(None, None, sdfg, None, [arg], [None])

    # Create output data
    outputs = ufunc._create_output(sdfg, [arg], [None], out_shape, dtype)

    # Set tasklet parameters
    impl = {
        'name':
        "_convert_to_{}_".format(dtype.to_string()),
        'inputs': ['__inp'],
        'outputs': ['__out'],
        'code':
        "__out = {}(__inp)".format(f"dace.{dtype.to_string()}" if dtype not in (dtypes.bool,
                                                                                dtypes.bool_) else dtype.to_string())
    }
    if dtype in (dtypes.bool, dtypes.bool_):
        impl['code'] = "__out = dace.bool_(__inp)"
    tasklet_params = ufunc._set_tasklet_params(impl, [arg])

    # Visitor input only needed when `has_where == True`.
    ufunc._create_subgraph(None,
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
def _ndarray_astype(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, dtype: dtypes.typeclass) -> str:
    if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
        dtype = dtypes.typeclass(dtype)
    return _datatype_converter(sdfg, state, arr, dtype)[0]
