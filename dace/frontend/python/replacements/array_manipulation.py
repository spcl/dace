# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for N-dimensional array transformations.
"""
import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.nested_call import NestedCall
from dace.frontend.python.replacements.utils import ProgramVisitor, UfuncInput, UfuncOutput
import dace.frontend.python.memlet_parser as mem_parser
from dace import data, dtypes, subsets, symbolic
from dace import Memlet, SDFG, SDFGState

import copy
from numbers import Integral, Number
from typing import Any, Optional, List, Sequence, Tuple, Union

import numpy as np


@oprepo.replaces('dace.roll')
@oprepo.replaces('numpy.roll')
def _numpy_roll(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, shift, axis=None) -> str:
    """Circular shift, lowered onto the ``CShift`` library node.

    ``CShift`` rotates in either direction and the node says which: Fortran ``CSHIFT(x, s)(i)``
    reads ``x(mod(i + s, n))`` where ``numpy.roll(x, s)[i]`` reads ``x[(i - s) % n]``. Passing
    :attr:`ShiftDirection.NUMPY` keeps the sign flip inside the expansion instead of open-coding a
    negation here -- a node rotated the wrong way has the right shape, dtype and even the right
    multiset of values, so nothing downstream can catch it.

    numpy takes a TUPLE of shifts and axes and applies them in order; each one becomes its own
    node, and the write node of one is reused as the read node of the next so the chain is ordered
    by dataflow rather than by two access nodes for the same array sitting unordered in one state.
    """
    from dace.libraries.standard.nodes.cshift import CShift, ShiftDirection  # Avoid import loop
    if arr not in sdfg.arrays:
        raise mem_parser.DaceSyntaxError(pv, None, f'numpy.roll argument {arr} is not SDFG data')
    desc = sdfg.arrays[arr]
    if isinstance(desc, data.Scalar):
        return arr
    ndim = len(desc.shape)
    shifts = list(shift) if isinstance(shift, (list, tuple)) else [shift]
    if axis is None:
        # numpy FLATTENS an axis-less roll, which is a reshape only when the operand is contiguous.
        # Rolling the last axis instead would return the right shape holding the wrong numbers.
        if ndim != 1:
            raise NotImplementedError(f'numpy.roll without an axis flattens a {ndim}-D operand; '
                                      f'pass an axis instead')
        axes = [0]
    else:
        axes = list(axis) if isinstance(axis, (list, tuple)) else [axis]
    if len(shifts) == 1 and len(axes) > 1:
        shifts = shifts * len(axes)
    if len(shifts) != len(axes):
        raise mem_parser.DaceSyntaxError(pv, None,
                                         f'numpy.roll got {len(shifts)} shifts for {len(axes)} axes; they must agree')
    axes = [ax if ax >= 0 else ax + ndim for ax in axes]
    if any(not 0 <= ax < ndim for ax in axes):
        raise mem_parser.DaceSyntaxError(pv, None, f'numpy.roll axis out of range for a {ndim}-D operand')

    source, node = arr, state.add_read(arr)
    for amount, ax in zip(shifts, axes):
        out, out_desc = pv.add_temp_transient(desc.shape, desc.dtype, storage=desc.storage)
        write = state.add_write(out)
        cshift = CShift(f'roll_{ax}',
                        dim=ax + 1,
                        shift=symbolic.pystr_to_symbolic(str(amount)),
                        direction=ShiftDirection.NUMPY)
        state.add_edge(node, None, cshift, '_x', Memlet.from_array(source, sdfg.arrays[source]))
        state.add_edge(cshift, '_out', write, None, Memlet.from_array(out, out_desc))
        source, node = out, write
    return source


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

    arr_copy_name = pv.get_target_name()
    # view = _ndarray_reshape(pv, sdfg, state, arr, desc.shape)
    # acpy, _ = sdfg.add_transient(arr_copy_name, desc.shape, desc.dtype, desc.storage, find_new_name=True)
    # vnode = state.add_read(view)
    # anode = state.add_read(acpy)
    # state.add_edge(vnode, None, anode, None, Memlet(f'{view}[{sset}] -> [{dset}]'))

    arr_copy, _ = sdfg.add_temp_transient_like(desc, name=arr_copy_name)
    inpidx = ','.join([f'__i{i}' for i in range(ndim)])
    outidx = ','.join([f'{s} - __i{i} - 1' if a else f'__i{i}' for i, (a, s) in enumerate(zip(axis, desc.shape))])
    state.add_mapped_tasklet(name="_numpy_flip_",
                             map_ranges={
                                 f'__i{i}': f'0:{s}:1'
                                 for i, s in enumerate(desc.shape)
                             },
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

    arr_copy, narr = sdfg.add_temp_transient_like(desc, name=pv.get_target_name())

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
                             map_ranges={
                                 f'__i{i}': f'0:{s}:1'
                                 for i, s in enumerate(desc.shape)
                             },
                             inputs={'__inp': Memlet(f'{arr}[{inpidx}]')},
                             code='__out = __inp',
                             outputs={'__out': Memlet(f'{arr_copy}[{outidx}]')},
                             external_edges=True)

    return arr_copy


@oprepo.replaces('numpy.triu')
def _numpy_triu(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, m: str, k: int = 0) -> str:
    """Copy of ``m`` with the elements below the k-th diagonal zeroed (upper triangle kept). For an input
    with more than two dimensions the mask applies to the final two axes, matching numpy."""
    return _triangle_mask(pv, sdfg, state, m, k, upper=True)


@oprepo.replaces('numpy.tril')
def _numpy_tril(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, m: str, k: int = 0) -> str:
    """Copy of ``m`` with the elements above the k-th diagonal zeroed (lower triangle kept). For an input
    with more than two dimensions the mask applies to the final two axes, matching numpy."""
    return _triangle_mask(pv, sdfg, state, m, k, upper=False)


def _triangle_mask(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, m: str, k: int, upper: bool) -> str:
    if m not in sdfg.arrays:
        raise mem_parser.DaceSyntaxError(pv, None, "Prototype argument {a} is not SDFG data!".format(a=m))
    desc = sdfg.arrays[m]
    if not isinstance(desc, (data.Array, data.View)):
        raise mem_parser.DaceSyntaxError(pv, None, "numpy.triu/numpy.tril only support Arrays and Views!")
    ndim = len(desc.shape)
    if ndim < 2:
        raise mem_parser.DaceSyntaxError(pv, None, "numpy.triu/numpy.tril require an at least two-dimensional input.")

    out_name, _ = sdfg.add_temp_transient_like(desc, name=pv.get_target_name())

    # Element (i, j) on the last two axes lies on diagonal ``j - i``; triu keeps ``j - i >= k`` (zeroing
    # below the k-th diagonal), tril keeps ``j - i <= k``. Every masked-out element is set to zero.
    row, col = f'__i{ndim - 2}', f'__i{ndim - 1}'
    cmp = '>=' if upper else '<='
    idx = ','.join(f'__i{i}' for i in range(ndim))
    state.add_mapped_tasklet(name='triu' if upper else 'tril',
                             map_ranges={
                                 f'__i{i}': f'0:{s}:1'
                                 for i, s in enumerate(desc.shape)
                             },
                             inputs={'__inp': Memlet(f'{m}[{idx}]')},
                             code=f'__out = __inp if ({col} - {row} {cmp} ({k})) else 0',
                             outputs={'__out': Memlet(f'{out_name}[{idx}]')},
                             external_edges=True)
    return out_name


@oprepo.replaces('transpose')
@oprepo.replaces('dace.transpose')
@oprepo.replaces('numpy.transpose')
def _transpose(pv: ProgramVisitor,
               sdfg: SDFG,
               state: SDFGState,
               inpname: str,
               axes=None,
               outname: Optional[str] = None) -> str:

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
    if outname is None:
        outname = pv.get_target_name()
    outname, arr2 = sdfg.add_transient(outname, new_shape, restype, arr1.storage, find_new_name=True)

    if axes == (1, 0):  # 2D transposition
        # A unit extent used to be routed around the library node with a hand-written index-swap
        # map, on two grounds that no longer hold: ``blas_helpers.matrix_view`` stopped squeezing,
        # so validation accepts a ``(N, 1)``, and the BLAS expansions now hand a one-element
        # operand to the pure single-element tasklet instead of building an omatcopy call around a
        # scalar (``linalg/nodes/transpose._is_single_element``). Every 2D shape goes to the node.
        acc1 = state.add_read(inpname)
        acc2 = state.add_write(outname)
        import dace.libraries.linalg  # Avoid import loop
        tasklet = dace.libraries.linalg.Transpose('_Transpose_', restype)
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
        from dace.libraries.linalg import TensorTranspose  # Avoid import loop
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


@oprepo.replaces('dace.moveaxis')
@oprepo.replaces('numpy.moveaxis')
def _moveaxis(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, source, destination) -> str:
    """``numpy.moveaxis``, lowered as the permutation it is.

    NumPy hands back a strided view. DaCe materializes instead, for the same reason
    :func:`broadcast_to` does: a permuted-stride operand reaching a library node is read with that
    node's own leading dimension, so a BLAS or tensor call built around it computes the wrong
    numbers without raising. ``TensorTranspose`` already moves the data, so the whole replacement
    is the axis arithmetic plus a delegation.
    """
    ndim = len(sdfg.arrays[arr].shape)
    src_axes = normalize_axes(source, ndim, 'source')
    dst_axes = normalize_axes(destination, ndim, 'destination')
    if len(src_axes) != len(dst_axes):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")

    axes = [a for a in range(ndim) if a not in src_axes]
    for dst, src in sorted(zip(dst_axes, src_axes)):
        axes.insert(dst, src)

    return _transpose(pv, sdfg, state, arr, axes)


def normalize_axes(axes, ndim: int, name: str) -> List[int]:
    """``axes`` as a list of non-negative indices, rejecting duplicates and out-of-range entries."""
    if isinstance(axes, Integral):
        axes = [axes]
    out = []
    for axis in axes:
        axis = int(axis)
        if axis < -ndim or axis >= ndim:
            raise ValueError(f"axis {axis} in `{name}` is out of bounds for an array of dimension {ndim}")
        out.append(axis + ndim if axis < 0 else axis)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in `{name}`")
    return out


@oprepo.replaces('numpy.broadcast_to')
def broadcast_to(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str,
                 shape: Union[str, symbolic.SymbolicType, Sequence[Union[str, symbolic.SymbolicType]]]) -> str:
    """Replicate ``arr`` across ``shape`` by the NumPy broadcasting rule.

    NumPy returns a zero-stride VIEW; DaCe materializes a transient instead, because a
    stride of 0 makes every write to the result alias and there is no way to tell here
    whether the caller only reads it.
    """
    from dace.libraries.standard.nodes import Broadcast  # Avoid import loop

    if isinstance(arr, (list, tuple)) and len(arr) == 1:
        arr = arr[0]
    desc = sdfg.arrays[arr]
    if isinstance(shape, (str, symbolic.symbol)) or isinstance(shape, Integral):
        shape = [shape]
    newshape = [symbolic.pystr_to_symbolic(s) for s in shape]
    if len(newshape) < len(desc.shape):
        raise ValueError(f'Cannot broadcast a rank-{len(desc.shape)} array to the '
                         f'rank-{len(newshape)} shape {tuple(newshape)}')

    out, out_desc = sdfg.add_transient(pv.get_target_name(), newshape, desc.dtype, desc.storage, find_new_name=True)
    node = Broadcast('broadcast_to', dim=None)
    state.add_node(node)
    state.add_edge(state.add_read(arr), None, node, '_src', Memlet.from_array(arr, desc))
    state.add_edge(node, '_dst', state.add_write(out), None, Memlet.from_array(out, out_desc))
    # The node's own validate() is what rejects a shape that does not broadcast; run it here
    # so the error names the numpy call instead of surfacing at expansion time.
    node.validate(sdfg, state)
    return out


@oprepo.replaces('numpy.ravel')
def ravel(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, order: StringLiteral = StringLiteral('C')) -> str:
    """``np.ravel`` is the free-function spelling of the ``.ravel()`` method."""
    return flat(pv, sdfg, state, arr, order)


@oprepo.replaces('numpy.squeeze')
def squeeze(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, axis=None) -> str:
    """``np.squeeze``: drop length-1 axes.

    Dropping is what the call asks for, so it is not the blind
    :meth:`~dace.subsets.Range.squeeze` that cannot tell an indexed axis from a sliced one -- an
    axis named here that is not provably 1 is an error, exactly as in NumPy.
    """
    shape = list(sdfg.arrays[arr].shape)
    if axis is None:
        keep = [i for i, extent in enumerate(shape) if symbolic.equal(extent, 1) is not True]
    else:
        axes = normalize_axes(axis, len(shape), 'axis')
        for ax in axes:
            if symbolic.equal(shape[ax], 1) is not True:
                raise ValueError(f'numpy.squeeze: axis {ax} has extent {shape[ax]}, which is not 1')
        keep = [i for i in range(len(shape)) if i not in axes]
    newshape = [shape[i] for i in keep] or [1]
    return reshape(pv, sdfg, state, arr, newshape)


@oprepo.replaces('numpy.expand_dims')
def expand_dims(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, axis) -> str:
    """``np.expand_dims``: insert length-1 axes. A reshape, not a replication."""
    shape = list(sdfg.arrays[arr].shape)
    # The inserted axes are positions in the RESULT, so the range is one wider than the source.
    axes = sorted(normalize_axes(axis, len(shape) + 1, 'axis'))
    for ax in axes:
        shape.insert(ax, 1)
    return reshape(pv, sdfg, state, arr, shape)


@oprepo.replaces('numpy.swapaxes')
def swapaxes(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, axis1: int, axis2: int) -> str:
    """``np.swapaxes`` is a transpose whose permutation swaps two entries."""
    ndim = len(sdfg.arrays[arr].shape)
    ax1 = normalize_axes(axis1, ndim, 'axis1')[0]
    ax2 = normalize_axes(axis2, ndim, 'axis2')[0]
    perm = list(range(ndim))
    perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
    return _transpose(pv, sdfg, state, arr, axes=perm)


@oprepo.replaces('numpy.rollaxis')
def rollaxis(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, axis: int, start: int = 0) -> str:
    """``np.rollaxis`` in terms of ``moveaxis``, with NumPy's own start adjustment."""
    ndim = len(sdfg.arrays[arr].shape)
    ax = normalize_axes(axis, ndim, 'axis')[0]
    dst = int(start) + ndim + 1 if int(start) < 0 else int(start)
    if dst > ax:  # numpy: the axis is removed before it is re-inserted
        dst -= 1
    return _moveaxis(pv, sdfg, state, arr, ax, dst)


@oprepo.replaces('numpy.fliplr')
def fliplr(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    if len(sdfg.arrays[arr].shape) < 2:
        raise ValueError('numpy.fliplr needs an array of at least rank 2')
    return _numpy_flip(pv, sdfg, state, arr, axis=1)


@oprepo.replaces('numpy.flipud')
def flipud(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    if len(sdfg.arrays[arr].shape) < 1:
        raise ValueError('numpy.flipud needs an array of at least rank 1')
    return _numpy_flip(pv, sdfg, state, arr, axis=0)


@oprepo.replaces('numpy.asarray')
@oprepo.replaces('numpy.ascontiguousarray')
def asarray(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, dtype: dtypes.typeclass = None) -> str:
    """``np.asarray`` / ``np.ascontiguousarray`` on data already inside the SDFG.

    Every DaCe array is contiguous in its own strides, so both are a copy -- and a copy rather than
    a no-op because NumPy's caller is entitled to write the result without touching the source.
    """
    from dace.frontend.python.replacements.array_creation import _numpy_copy  # Avoid import loop

    out = _numpy_copy(pv, sdfg, state, arr)
    if dtype is not None and dtype != sdfg.arrays[out].dtype:
        return _ndarray_astype(pv, sdfg, state, out, dtype)
    return out


@oprepo.replaces('numpy.atleast_1d')
def atleast_1d(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _atleast_nd(pv, sdfg, state, arr, 1)


@oprepo.replaces('numpy.atleast_2d')
def atleast_2d(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _atleast_nd(pv, sdfg, state, arr, 2)


@oprepo.replaces('numpy.atleast_3d')
def atleast_3d(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _atleast_nd(pv, sdfg, state, arr, 3)


def _atleast_nd(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, rank: int) -> str:
    """Shared body of ``atleast_{1,2,3}d``: pad the shape to ``rank`` the way NumPy does."""
    shape = list(sdfg.arrays[arr].shape)
    if len(shape) >= rank:
        return arr
    if rank == 1:
        shape = [1]
    elif rank == 2:
        shape = [1] * (2 - len(shape)) + shape
    else:  # atleast_3d puts the NEW trailing axis last, and a 1-D input becomes (1, n, 1)
        shape = ([1] + shape if len(shape) == 1 else shape) + [1] * (3 - max(len(shape), 2))
    return reshape(pv, sdfg, state, arr, shape)


@oprepo.replaces('numpy.copyto')
def copyto(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dst: str, src: str) -> str:
    """``np.copyto(dst, src)``: an in-place copy edge, so the caller's array is the one written."""
    if dst not in sdfg.arrays or src not in sdfg.arrays:
        raise ValueError('numpy.copyto expects two arrays')
    dst_desc, src_desc = sdfg.arrays[dst], sdfg.arrays[src]
    if not symbolic.shapes_equal(list(src_desc.shape), list(dst_desc.shape)):
        raise ValueError(f'numpy.copyto: shapes {tuple(src_desc.shape)} and {tuple(dst_desc.shape)} do not match')
    state.add_edge(state.add_read(src), None, state.add_write(dst), None, Memlet.from_array(src, src_desc))
    return dst


@oprepo.replaces('numpy.diagonal')
def diagonal(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, offset: int = 0) -> str:
    """The ``offset``-th diagonal of a rank-2 array, as a Map rather than a strided view.

    NumPy returns a read-only view with stride ``rows + 1``; DaCe materializes, because a strided
    operand handed to a library node is read with that node's own leading dimension.
    """
    desc = sdfg.arrays[arr]
    if len(desc.shape) != 2:
        raise ValueError('numpy.diagonal is supported for rank-2 arrays')
    if not isinstance(offset, Integral):
        raise ValueError('numpy.diagonal needs a compile-time offset')
    rows, cols = desc.shape
    off = int(offset)
    # A positive offset starts on row 0 and column `off`; a negative one starts on row `-off`.
    length = symbolic.pystr_to_symbolic(f'min({rows}, {cols} - {off})' if off >= 0 else f'min({rows} + {off}, {cols})')
    row0, col0 = (0, off) if off >= 0 else (-off, 0)
    out, out_desc = sdfg.add_transient(pv.get_target_name(), [length], desc.dtype, desc.storage, find_new_name=True)
    state.add_mapped_tasklet(f'diagonal_{off}', {'__d': f'0:{length}'},
                             {'__inp': Memlet(f'{arr}[__d + {row0}, __d + {col0}]')},
                             '__out = __inp', {'__out': Memlet(f'{out}[__d]')},
                             external_edges=True)
    return out


@oprepo.replaces('numpy.diag')
def diag(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, k: int = 0) -> str:
    """``np.diag``: extract the diagonal of a matrix, or build a matrix from a vector."""
    from dace.frontend.python.replacements.array_creation import _numpy_full  # Avoid import loop

    desc = sdfg.arrays[arr]
    if len(desc.shape) == 2:
        return diagonal(pv, sdfg, state, arr, k)
    if len(desc.shape) != 1:
        raise ValueError('numpy.diag takes a rank-1 or rank-2 array')
    if not isinstance(k, Integral):
        raise ValueError('numpy.diag needs a compile-time k')
    n = desc.shape[0] + abs(int(k))
    out = _numpy_full(pv, sdfg, state, [n, n], 0, desc.dtype)
    row0, col0 = (0, int(k)) if k >= 0 else (-int(k), 0)
    # The zero fill and the diagonal write are two statements on one array: the fill has to be
    # complete before the diagonal lands, which the state boundary is what guarantees.
    state = pv.last_block
    state.add_mapped_tasklet(f'diag_{k}', {'__d': f'0:{desc.shape[0]}'}, {'__inp': Memlet(f'{arr}[__d]')},
                             '__out = __inp', {'__out': Memlet(f'{out}[__d + {row0}, __d + {col0}]')},
                             external_edges=True)
    return out


@oprepo.replaces('numpy.pad')
def pad(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, pad_width, mode='constant', **kwargs) -> str:
    """``np.pad`` in constant mode: fill the padded shape, then copy the original into the interior.

    Only ``mode='constant'`` lowers. The edge modes read a mirrored or clamped index, which is a
    different kernel and not what any of the padding call sites here ask for.
    """
    from dace.frontend.python.replacements.array_creation import _numpy_full  # Avoid import loop

    if str(mode) != 'constant':
        raise NotImplementedError(f'numpy.pad supports mode="constant", not mode="{mode}"')
    fill = kwargs.get('constant_values', 0)
    if isinstance(fill, (list, tuple)):
        raise NotImplementedError('numpy.pad with per-axis constant_values is not supported')
    desc = sdfg.arrays[arr]
    ndim = len(desc.shape)

    widths = pad_width
    if isinstance(widths, Integral):
        widths = [(int(widths), int(widths))] * ndim
    elif isinstance(widths, (list, tuple)) and widths and isinstance(widths[0], Integral):
        widths = [(int(widths[0]), int(widths[-1]))] * ndim
    else:
        widths = [(int(lo), int(hi)) for lo, hi in widths]
    if len(widths) != ndim:
        raise ValueError(f'numpy.pad: {len(widths)} pad widths for a rank-{ndim} array')

    out_shape = [extent + lo + hi for extent, (lo, hi) in zip(desc.shape, widths)]
    out = _numpy_full(pv, sdfg, state, out_shape, fill, desc.dtype)
    # The fill and the interior copy are two writes to one array; the state boundary the fill
    # opened is what orders them.
    state = pv.last_block
    interior = ', '.join(f'{lo}:{lo} + {extent}' for extent, (lo, _) in zip(desc.shape, widths))
    state.add_edge(
        state.add_read(arr), None, state.add_write(out), None,
        Memlet(data=arr, subset=subsets.Range.from_array(desc), other_subset=subsets.Range.from_string(interior)))
    return out


@oprepo.replaces('numpy.fill_diagonal')
def fill_diagonal(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, val) -> str:
    """``np.fill_diagonal`` writes IN PLACE, so the caller's array is the one returned."""
    desc = sdfg.arrays[arr]
    if len(desc.shape) != 2:
        raise ValueError('numpy.fill_diagonal is supported for rank-2 arrays')
    n = f'min({desc.shape[0]}, {desc.shape[1]})'
    if isinstance(val, str) and val in sdfg.arrays:
        state.add_mapped_tasklet('fill_diagonal', {'__d': f'0:{n}'}, {'__inp': Memlet(f'{val}[0]')},
                                 '__out = __inp', {'__out': Memlet(f'{arr}[__d, __d]')},
                                 external_edges=True)
    else:
        state.add_mapped_tasklet('fill_diagonal', {'__d': f'0:{n}'}, {},
                                 f'__out = {val}', {'__out': Memlet(f'{arr}[__d, __d]')},
                                 external_edges=True)
    return arr


@oprepo.replaces('numpy.diagflat')
def diagflat(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, k: int = 0) -> str:
    """``np.diagflat`` is :func:`diag` of the flattened input."""
    return diag(pv, sdfg, state, flat(pv, sdfg, state, arr), k)


@oprepo.replaces('numpy.diff')
def diff(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, n: int = 1, axis: int = -1) -> str:
    """``np.diff``: the adjacent difference along ``axis``, applied ``n`` times."""
    if not isinstance(n, Integral) or int(n) < 0:
        raise ValueError('numpy.diff needs a compile-time non-negative n')
    if int(n) == 0:
        return arr
    desc = sdfg.arrays[arr]
    ndim = len(desc.shape)
    ax = normalize_axes(axis, ndim, 'axis')[0]

    out_shape = list(desc.shape)
    out_shape[ax] = desc.shape[ax] - 1
    out, out_desc = sdfg.add_transient(pv.get_target_name(), out_shape, desc.dtype, desc.storage, find_new_name=True)
    rng = {f'__i{d}': f'0:{out_shape[d]}' for d in range(ndim)}
    hi = ', '.join(f'__i{d} + 1' if d == ax else f'__i{d}' for d in range(ndim))
    lo = ', '.join(f'__i{d}' for d in range(ndim))
    state.add_mapped_tasklet('diff',
                             rng, {
                                 '__hi': Memlet(f'{arr}[{hi}]'),
                                 '__lo': Memlet(f'{arr}[{lo}]')
                             },
                             '__out = __hi - __lo', {'__out': Memlet(f'{out}[{lo}]')},
                             external_edges=True)
    return diff(pv, pv.sdfg, pv.last_block, out, int(n) - 1, ax) if int(n) > 1 else out


@oprepo.replaces('numpy.ediff1d')
def ediff1d(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    """``np.ediff1d`` is :func:`diff` of the flattened input."""
    return diff(pv, sdfg, state, flat(pv, sdfg, state, arr))


@oprepo.replaces('numpy.meshgrid')
def meshgrid(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *arrays, indexing='xy') -> list:
    """``np.meshgrid`` for 1-D inputs: one ``Broadcast`` per output axis.

    ``indexing='xy'`` transposes the first two axes against ``'ij'``, which is the whole difference
    between the two spellings and the one a stencil gets wrong silently.
    """

    if str(indexing) not in ('xy', 'ij'):
        raise ValueError(f"numpy.meshgrid: indexing must be 'xy' or 'ij', got {indexing!r}")
    names = [a for a in arrays]
    for name in names:
        if name not in sdfg.arrays or len(sdfg.arrays[name].shape) != 1:
            raise NotImplementedError('numpy.meshgrid is supported for rank-1 inputs')
    lengths = [sdfg.arrays[n].shape[0] for n in names]
    grid = list(lengths)
    if str(indexing) == 'xy' and len(names) >= 2:
        grid[0], grid[1] = grid[1], grid[0]

    out_names = []
    for k, name in enumerate(names):
        desc = sdfg.arrays[name]
        # Which grid axis this operand varies along -- 'xy' swaps the first two.
        axis = k
        if str(indexing) == 'xy' and k < 2:
            axis = 1 - k
        out, out_desc = sdfg.add_transient(pv.get_target_name(), grid, desc.dtype, desc.storage, find_new_name=True)
        rng = {f'__o{d}': f'0:{grid[d]}' for d in range(len(grid))}
        state.add_mapped_tasklet(
            f'meshgrid_{k}',
            rng, {'__inp': Memlet(f'{name}[__o{axis}]')},
            '__out = __inp', {'__out': Memlet('{}[{}]'.format(out, ', '.join(f'__o{d}' for d in range(len(grid)))))},
            external_edges=True)
        out_names.append(out)
    return out_names


@oprepo.replaces('numpy.trace')
def trace(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, offset: int = 0) -> str:
    """``np.trace`` is the sum of :func:`diagonal`."""
    from dace.frontend.python.replacements.reduction import _sum  # Avoid import loop

    nest = NestedCall(pv, sdfg, state)
    return nest, nest(_sum)(nest(diagonal)(arr, offset))


@oprepo.replaces('numpy.tile')
def tile(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, reps) -> str:
    """``np.tile`` for a rank-1 array repeated a whole number of times.

    Tiling REPEATS THE ARRAY (``a b a b``), where :func:`repeat` repeats each element
    (``a a b b``); both are one ``Broadcast`` plus a reshape, differing only in which side of the
    source axis the new one is inserted.
    """
    from dace.libraries.standard.nodes import Broadcast  # Avoid import loop

    desc = sdfg.arrays[arr]
    if isinstance(reps, (list, tuple)):
        if len(reps) != 1:
            raise ValueError('numpy.tile is supported for a single repetition count')
        reps = reps[0]
    if len(desc.shape) != 1:
        raise ValueError('numpy.tile is supported for rank-1 arrays')
    count = symbolic.pystr_to_symbolic(reps) if isinstance(reps, str) else reps

    out, out_desc = sdfg.add_transient(pv.get_target_name(), [count, desc.shape[0]],
                                       desc.dtype,
                                       desc.storage,
                                       find_new_name=True)
    node = Broadcast('tile', dim=1)  # the copies axis goes FIRST, so the source repeats whole
    state.add_node(node)
    state.add_edge(state.add_read(arr), None, node, '_src', Memlet.from_array(arr, desc))
    state.add_edge(node, '_dst', state.add_write(out), None, Memlet.from_array(out, out_desc))
    node.validate(sdfg, state)
    return reshape(pv, sdfg, state, out, [desc.shape[0] * count])


@oprepo.replaces('numpy.repeat')
def repeat(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, repeats: Any, axis: Optional[int] = None) -> str:
    """``np.repeat``: SPREAD a new axis of length ``repeats`` next to the repeated one, then merge
    the pair back with a reshape.

    ``repeats`` is one count for the whole array. A per-element count makes the result extent
    data-dependent, which no static descriptor can carry, so it is refused rather than lowered to a
    callback that only appears to work.
    """
    from dace.libraries.standard.nodes import Broadcast  # Avoid import loop

    if isinstance(arr, (list, tuple)) and len(arr) == 1:
        arr = arr[0]
    if not isinstance(arr, str) or arr not in sdfg.arrays:
        raise ValueError('numpy.repeat expects an array to repeat')
    if isinstance(repeats, str) and repeats in sdfg.arrays:
        raise ValueError('numpy.repeat with a per-element repeats array is unsupported: the result '
                         'extent would be data-dependent')
    count = symbolic.pystr_to_symbolic(repeats) if isinstance(repeats, str) else repeats

    if axis is None:
        arr = flat(pv, sdfg, state, arr)
        axis = 0
    desc = sdfg.arrays[arr]
    ndim = len(desc.shape)
    if not isinstance(axis, Integral):
        raise ValueError('numpy.repeat needs a compile-time axis')
    ax = int(axis) + ndim if axis < 0 else int(axis)
    if not 0 <= ax < ndim:
        raise ValueError(f'numpy.repeat: axis {axis} is out of range for a rank-{ndim} array')

    spread_shape = list(desc.shape)
    spread_shape.insert(ax + 1, count)
    out, out_desc = sdfg.add_transient(pv.get_target_name(), spread_shape, desc.dtype, desc.storage, find_new_name=True)
    node = Broadcast('repeat', dim=ax + 2)  # 1-based position of the axis SPREAD inserts
    state.add_node(node)
    state.add_edge(state.add_read(arr), None, node, '_src', Memlet.from_array(arr, desc))
    state.add_edge(node, '_dst', state.add_write(out), None, Memlet.from_array(out, out_desc))
    node.validate(sdfg, state)

    merged = list(desc.shape)
    merged[ax] = desc.shape[ax] * count
    return reshape(pv, sdfg, state, out, merged)


@oprepo.replaces('numpy.reshape')
def reshape(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            arr: str,
            newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
            order: StringLiteral = StringLiteral('C'),
            strides: Optional[Any] = None) -> str:
    if isinstance(arr, (list, tuple)) and len(arr) == 1:
        arr = arr[0]
    desc = sdfg.arrays[arr]

    # "order" determines stride orders. ``'A'`` asks whether the source is FORTRAN-contiguous, which
    # a unit first stride does not answer -- every array whose leading axis walks by one has it.
    order = str(order)
    fortran_strides = order == 'F' or (order == 'A' and desc.is_packed_fortran_strides())

    # New shape and strides as symbolic expressions
    newshape = [symbolic.pystr_to_symbolic(s) for s in newshape]
    undecided = False
    if strides is None:
        # numpy's own rule: a view exactly when the new shape factors into the source's
        # stride-contiguous groups. Reinterpreting a source as packed reads the elements a slice
        # skips -- densenet's im2col reshapes a strided window, so every tap came out of the zero
        # padding -- and copying where numpy aliases drops a write through the result instead.
        strides, decided = data.core.nocopy_reshape_strides(desc.shape, desc.strides, newshape, fortran_strides)
        if strides is None:
            from dace.frontend.python.replacements.array_creation import _numpy_copy  # Avoid import loop
            source = arr  # the name a write through the result would have had to reach
            arr = _numpy_copy(pv, sdfg, state, arr)
            desc = sdfg.arrays[arr]
            # A PROVABLE copy is what numpy itself does, writes to the result included -- they land in
            # the copy there too. Only an UNDECIDED one has to refuse a write, because view and copy
            # are then two different programs and nothing here can pick between them.
            undecided = not decided
            # The copy is packed, so the view over it is the packed formula. Re-running the factoring
            # would answer "undecided" all over again whenever the extents are what made it undecided.
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
    if undecided:
        pv.detached_reshapes[newarr] = source

    return newarr


@oprepo.replaces_method('Array', 'reshape')
@oprepo.replaces_method('View', 'reshape')
def _ndarray_reshape(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    arr: str,
    *newshape: Union[str, symbolic.SymbolicType, Tuple[Union[str, symbolic.SymbolicType]]],
    order: StringLiteral = StringLiteral('C')
) -> str:
    if len(newshape) == 0:
        raise TypeError('reshape() takes at least 1 argument (0 given)')
    if len(newshape) == 1 and isinstance(newshape, (list, tuple)):
        newshape = newshape[0]
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
        from dace.frontend.python.replacements.array_creation import _ndarray_copy  # Avoid circular import
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

    orig_bytes = desc.dtype.bytes
    view_bytes = dtype.bytes

    if view_bytes < orig_bytes and orig_bytes % view_bytes != 0:
        raise ValueError("When changing to a smaller dtype, its size must be a divisor of "
                         "the size of original dtype")

    contigdim = next(i for i, s in enumerate(desc.strides) if s == 1)

    # For cases that can be recognized, if contiguous dimension is too small
    # raise an exception similar to numpy
    if (not symbolic.issymbolic(desc.shape[contigdim], sdfg.constants) and orig_bytes < view_bytes
            and desc.shape[contigdim] * orig_bytes % view_bytes != 0):
        raise ValueError('When changing to a larger dtype, its size must be a divisor of '
                         'the total size in bytes of the last axis of the array.')

    # Create new shape and strides for view
    # NOTE: we change sizes by using `(old_size * orig_bytes) // view_bytes`
    # Thus, the changed size will be an integer due to integer division.
    # If the division created a fraction, the view wouldn't be valid in the first place.
    # So, we assume the division will always yield an integer, and, hence,
    # the integer division is correct.
    # Also, keep in mind that `old_size * (orig_bytes // view_bytes)` is different.
    # E.g., if `orig_bytes == 1 and view_bytes == 2`: `old_size * (1 // 2) == old_size * 0`.
    newshape = list(desc.shape)
    # int_floor, never `//`: on a symbolic stride `//` builds sympy `floor(expr / d)`, whose argument
    # sym2cpp prints WITHOUT the floor, leaving each term of the sum to truncate on its own.
    newstrides = [
        symbolic.int_floor(s * orig_bytes, view_bytes) if i != contigdim else s for i, s in enumerate(desc.strides)
    ]
    # don't use `*=`, because it will break the bracket
    newshape[contigdim] = symbolic.int_floor(newshape[contigdim] * orig_bytes, view_bytes)

    newarr, _ = sdfg.add_view(arr,
                              newshape,
                              dtype,
                              storage=desc.storage,
                              strides=newstrides,
                              allow_conflicts=desc.allow_conflicts,
                              total_size=symbolic.int_floor(desc.total_size * orig_bytes, view_bytes),
                              may_alias=desc.may_alias,
                              alignment=desc.alignment,
                              find_new_name=True)

    # Register view with DaCe program visitor
    # NOTE: We do not create here a Memlet of the form `A[subset] -> [osubset]`
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
        # np.dtype resolves numpy names and the ml_dtypes-registered low-precision
        # names (bfloat16 / float8_e4m3fn / float8_e5m2) alike.
        dtype = dtypes.dtype_to_typeclass(np.dtype(typeclass).type)

    @oprepo.replaces(typeclass)
    @oprepo.replaces("dace.{}".format(typeclass))
    @oprepo.replaces("numpy.{}".format(typeclass))
    def _converter(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput):
        return _datatype_converter(sdfg, state, arg, dtype=dtype, name_hint=pv.get_target_name())


for typeclass in dtypes.TYPECLASS_STRINGS:
    _make_datatype_converter(typeclass)


def _datatype_converter(sdfg: SDFG, state: SDFGState, arg: UfuncInput, dtype: dtypes.typeclass,
                        name_hint: str) -> UfuncOutput:
    """ Out-of-place datatype conversion of the input argument.

        :param sdfg: SDFG object
        :param state: SDFG State object
        :param arg: Input argument
        :param dtype: Datatype to convert input argument into
        :param name_hint: Name hint for the output array

        :return: ``dace.data.Array`` of same size as input or ``dace.data.Scalar``
    """
    from dace.frontend.python.replacements import ufunc

    # Get shape and indices
    (out_shape, map_indices, out_indices, inp_indices) = ufunc._validate_shapes(None, None, sdfg, None, [arg], [None])

    # Create output data
    outputs = ufunc._create_output(sdfg, [arg], [None], out_shape, dtype, name_hint=name_hint)

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
def _ndarray_astype(pv: ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    arr: str,
                    dtype: dtypes.typeclass,
                    copy: bool = True) -> str:
    # The copy is FORCED, whichever value is passed. ``copy`` is a permission in numpy rather than
    # an instruction -- ``copy=False`` allows eliding a copy, never forbids one -- so always
    # materialising is a legal reading of both. It is also the only one available: the elision
    # numpy performs returns the operand ITSELF when the dtype already matches, and an SDFG has no
    # way to hand back an alias in place of a converted array. Accepting the keyword and copying
    # anyway is therefore the honest behaviour; without the parameter at all the call raised
    # TypeError and refused the program outright.
    if isinstance(dtype, type) and dtype in dtypes._CONSTANT_TYPES[:-1]:
        dtype = dtypes.typeclass(dtype)
    return _datatype_converter(sdfg, state, arr, dtype, pv.get_target_name())[0]


@oprepo.replaces('numpy.concatenate')
def _concat(visitor: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            arrays: Tuple[Any],
            axis: Optional[int] = 0,
            out: Optional[Any] = None,
            *,
            dtype=None,
            casting: str = 'same_kind'):
    if dtype is not None and out is not None:
        raise ValueError('Arguments dtype and out cannot be given together')
    if casting != 'same_kind':
        raise NotImplementedError('The casting argument is currently unsupported')
    if not isinstance(arrays, (tuple, list)):
        raise ValueError('List of arrays is not iterable, cannot compile concatenation')
    if axis is not None and not isinstance(axis, Integral):
        raise ValueError('Axis is not a compile-time evaluatable integer, cannot compile concatenation')
    if len(arrays) == 1:
        return arrays[0]
    for i in range(len(arrays)):
        if arrays[i] not in sdfg.arrays:
            raise TypeError(f'Index {i} is not an array')
    if out is not None:
        if out not in sdfg.arrays:
            raise TypeError('Output is not an array')
        dtype = sdfg.arrays[out].dtype

    descs = [sdfg.arrays[arr] for arr in arrays]
    shape = list(descs[0].shape)

    if axis is None:  # Flatten arrays, then concatenate
        arrays = [flat(visitor, sdfg, state, arr) for arr in arrays]
        descs = [sdfg.arrays[arr] for arr in arrays]
        shape = list(descs[0].shape)
        axis = 0
    else:
        # Check shapes for validity
        first_shape = copy.copy(shape)
        first_shape[axis] = 0
        for i, d in enumerate(descs[1:]):
            other_shape = list(d.shape)
            other_shape[axis] = 0
            if not symbolic.shapes_equal(other_shape, first_shape):
                raise ValueError(f'Array shapes do not match at index {i}')

    shape[axis] = sum(desc.shape[axis] for desc in descs)
    if out is None:
        if dtype is None:
            dtype = descs[0].dtype
        name, odesc = sdfg.add_transient(visitor.get_target_name(),
                                         shape,
                                         dtype,
                                         storage=descs[0].storage,
                                         lifetime=descs[0].lifetime,
                                         find_new_name=True)
    else:
        name = out
        odesc = sdfg.arrays[out]

    if out is None:
        # Fast path: a single write node with per-array subset edges is enough
        # for most cases. A single write access node with multiple incoming direct
        # edges can be mis-optimized when the output is also an input/argument
        # (see numpy.concatenate(..., out=...)), so we use explicit tasklets there.
        w = state.add_write(name)
        offset = 0
        subset = subsets.Range.from_array(odesc)
        for arr, desc in zip(arrays, descs):
            r = state.add_read(arr)
            subset = copy.deepcopy(subset)
            subset[axis] = (offset, offset + desc.shape[axis] - 1, 1)
            state.add_edge(r, None, w, None, Memlet(data=name, subset=subset))
            offset += desc.shape[axis]
    else:
        # The output array is reused from the caller; materialize the copy with
        # per-array tasklets so the simplifier cannot drop a partial write.
        offset = 0
        for arr, desc in zip(arrays, descs):
            map_ranges = {}
            inpidx = []
            outidx = []
            for i, s in enumerate(desc.shape):
                var = f'__i{i}'
                map_ranges[var] = f'0:{s}:1'
                inpidx.append(var)
                if i == axis:
                    outidx.append(f'{offset} + {var}')
                else:
                    outidx.append(var)
            inpidx = ','.join(inpidx)
            outidx = ','.join(outidx)
            state.add_mapped_tasklet(name='_concat_copy_',
                                     map_ranges=map_ranges,
                                     inputs={'__inp': Memlet(f'{arr}[{inpidx}]')},
                                     code='__out = __inp',
                                     outputs={'__out': Memlet(f'{name}[{outidx}]')},
                                     external_edges=True,
                                     propagate=True)
            offset += desc.shape[axis]

    return name


@oprepo.replaces('numpy.append')
def append(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str, values: Any, axis: Optional[int] = None) -> str:
    """``np.append`` is ``np.concatenate`` over two operands; the default ``axis=None`` flattens both.

    A scalar ``values`` is materialized as a length-1 array first, which is what NumPy's own
    ``append`` does before concatenating.
    """
    from dace.frontend.python.replacements.array_creation import _numpy_full  # Avoid import loop

    if isinstance(values, (Number, np.bool_)) or symbolic.issymbolic(values):
        values = _numpy_full(pv, sdfg, state, [1], values, sdfg.arrays[arr].dtype)
    return _concat(pv, sdfg, state, (arr, values), axis=axis)


@oprepo.replaces('numpy.stack')
def _stack(visitor: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState,
           arrays: Tuple[Any],
           axis: int = 0,
           out: Any = None,
           *,
           dtype=None,
           casting: str = 'same_kind'):
    if dtype is not None and out is not None:
        raise ValueError('Arguments dtype and out cannot be given together')
    if casting != 'same_kind':
        raise NotImplementedError('The casting argument is currently unsupported')
    if not isinstance(arrays, (tuple, list)):
        raise ValueError('List of arrays is not iterable, cannot compile stack call')
    if not isinstance(axis, Integral):
        raise ValueError('Axis is not a compile-time evaluatable integer, cannot compile stack call')

    for i in range(len(arrays)):
        if arrays[i] not in sdfg.arrays:
            raise TypeError(f'Index {i} is not an array')

    descs = [sdfg.arrays[a] for a in arrays]
    shape = descs[0].shape
    for i, d in enumerate(descs[1:]):
        if not symbolic.shapes_equal(d.shape, shape):
            raise ValueError(f'Array shapes are not equal ({shape} != {d.shape} at index {i})')

    if axis > len(shape):
        raise ValueError(f'axis {axis} is out of bounds for array of dimension {len(shape)}')
    if axis < 0:
        naxis = len(shape) + 1 + axis
        if naxis < 0 or naxis > len(shape):
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {len(shape)}')
        axis = naxis

    # Stacking is implemented as a reshape followed by concatenation
    reshaped = []
    for arr, desc in zip(arrays, descs):
        # Make a reshaped view with the inserted dimension
        new_shape = [0] * (len(shape) + 1)
        new_strides = [0] * (len(shape) + 1)
        for i in range(len(shape) + 1):
            if i == axis:
                new_shape[i] = 1
                new_strides[i] = desc.strides[i - 1] if i != 0 else desc.strides[i]
            elif i < axis:
                new_shape[i] = shape[i]
                new_strides[i] = desc.strides[i]
            else:
                new_shape[i] = shape[i - 1]
                new_strides[i] = desc.strides[i - 1]

        rname = reshape(visitor, sdfg, state, arr, new_shape, strides=new_strides)
        reshaped.append(rname)

    return _concat(visitor, sdfg, state, reshaped, axis, out, dtype=dtype, casting=casting)


@oprepo.replaces('numpy.vstack')
@oprepo.replaces('numpy.row_stack')
def _vstack(visitor: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            tup: Tuple[Any],
            *,
            dtype=None,
            casting: str = 'same_kind'):
    if not isinstance(tup, (tuple, list)):
        raise ValueError('List of arrays is not iterable, cannot compile stack call')
    if tup[0] not in sdfg.arrays:
        raise TypeError(f'Index 0 is not an array')

    # In the 1-D case, stacking is performed along the first axis
    if len(sdfg.arrays[tup[0]].shape) == 1:
        return _stack(visitor, sdfg, state, tup, axis=0, out=None, dtype=dtype, casting=casting)
    # Otherwise, concatenation is performed
    return _concat(visitor, sdfg, state, tup, axis=0, out=None, dtype=dtype, casting=casting)


@oprepo.replaces('numpy.hstack')
@oprepo.replaces('numpy.column_stack')
def _hstack(visitor: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            tup: Tuple[Any],
            *,
            dtype=None,
            casting: str = 'same_kind'):
    if not isinstance(tup, (tuple, list)):
        raise ValueError('List of arrays is not iterable, cannot compile stack call')
    if tup[0] not in sdfg.arrays:
        raise TypeError(f'Index 0 is not an array')

    # In the 1-D case, concatenation is performed along the first axis
    if len(sdfg.arrays[tup[0]].shape) == 1:
        return _concat(visitor, sdfg, state, tup, axis=0, out=None, dtype=dtype, casting=casting)

    return _concat(visitor, sdfg, state, tup, axis=1, out=None, dtype=dtype, casting=casting)


@oprepo.replaces('numpy.dstack')
def _dstack(visitor: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            tup: Tuple[Any],
            *,
            dtype=None,
            casting: str = 'same_kind'):
    if not isinstance(tup, (tuple, list)):
        raise ValueError('List of arrays is not iterable, cannot compile a stack call')
    if tup[0] not in sdfg.arrays:
        raise TypeError(f'Index 0 is not an array')
    if len(sdfg.arrays[tup[0]].shape) < 3:
        raise NotImplementedError('dstack is not implemented for arrays that are smaller than 3D')

    return _concat(visitor, sdfg, state, tup, axis=2, out=None, dtype=dtype, casting=casting)


def _split_core(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, ary: str,
                indices_or_sections: Union[int, Sequence[symbolic.SymbolicType], str], axis: int, allow_uneven: bool):
    # Argument checks
    if not isinstance(ary, str) or ary not in sdfg.arrays:
        raise TypeError('Split object must be an array')
    if not isinstance(axis, Integral):
        raise ValueError('Cannot determine split dimension, axis is not a compile-time evaluatable integer')

    desc = sdfg.arrays[ary]

    # Test validity of axis
    orig_axis = axis
    if axis < 0:
        axis = len(desc.shape) + axis
    if axis < 0 or axis >= len(desc.shape):
        raise ValueError(f'axis {orig_axis} is out of bounds for array of dimension {len(desc.shape)}')

    # indices_or_sections may only be an integer (not symbolic), list of integers, list of symbols, or an array
    if isinstance(indices_or_sections, str):
        raise ValueError('Array-indexed split cannot be compiled due to data-dependent sizes. '
                         'Consider using numpy.reshape instead.')
    elif isinstance(indices_or_sections, (list, tuple)):
        if any(isinstance(i, str) for i in indices_or_sections):
            raise ValueError('Array-indexed split cannot be compiled due to data-dependent sizes. '
                             'Use symbolic values as an argument instead.')
        # Sequence is given
        sections = indices_or_sections
    elif isinstance(indices_or_sections, Integral):  # Constant integer given
        if indices_or_sections <= 0:
            raise ValueError('Number of sections must be larger than zero.')

        # If uneven sizes are not allowed and ary shape is numeric, check evenness
        if not allow_uneven and not symbolic.issymbolic(desc.shape[axis]):
            if desc.shape[axis] % indices_or_sections != 0:
                raise ValueError('Array split does not result in an equal division. Consider using numpy.array_split '
                                 'instead.')
        if indices_or_sections > desc.shape[axis]:
            raise ValueError('Cannot compile array split as it will result in empty arrays.')

        # Sequence is not given, compute sections
        # Mimic behavior of array_split in numpy: Sections are [s+1 x N%s], s, ..., s
        size = desc.shape[axis] // indices_or_sections
        remainder = desc.shape[axis] % indices_or_sections
        sections = []
        offset = 0
        for _ in range(min(remainder, indices_or_sections)):
            offset += size + 1
            sections.append(offset)
        for _ in range(remainder, indices_or_sections - 1):
            offset += size
            sections.append(offset)

    elif symbolic.issymbolic(indices_or_sections):
        raise ValueError('Symbolic split cannot be compiled due to output tuple size being unknown. '
                         'Consider using numpy.reshape instead.')
    else:
        raise TypeError(f'Unsupported type {type(indices_or_sections)} for indices_or_sections in numpy.split')

    # Split according to sections
    r = state.add_read(ary)
    result = []
    offset = 0
    outname = visitor.get_target_name()
    for i, section in enumerate(sections):
        shape = list(desc.shape)
        shape[axis] = section - offset
        name, _ = sdfg.add_transient(f'{outname}_{i}',
                                     shape,
                                     desc.dtype,
                                     storage=desc.storage,
                                     lifetime=desc.lifetime,
                                     find_new_name=True)
        # Add copy
        w = state.add_write(name)
        subset = subsets.Range.from_array(desc)
        subset[axis] = (offset, offset + shape[axis] - 1, 1)
        offset += shape[axis]
        state.add_nedge(r, w, Memlet(data=ary, subset=subset))
        result.append(name)

    # Add final section
    shape = list(desc.shape)
    shape[axis] -= offset
    name, _ = sdfg.add_transient(f'{outname}_{len(sections)}',
                                 shape,
                                 desc.dtype,
                                 storage=desc.storage,
                                 lifetime=desc.lifetime,
                                 find_new_name=True)
    w = state.add_write(name)
    subset = subsets.Range.from_array(desc)
    subset[axis] = (offset, offset + shape[axis] - 1, 1)
    state.add_nedge(r, w, Memlet(data=ary, subset=subset))
    result.append(name)

    # Always return a list of results, even if the size is 1
    return result


@oprepo.replaces('numpy.split')
def _split(visitor: ProgramVisitor,
           sdfg: SDFG,
           state: SDFGState,
           ary: str,
           indices_or_sections: Union[symbolic.SymbolicType, List[symbolic.SymbolicType], str],
           axis: int = 0):
    return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis, allow_uneven=False)


@oprepo.replaces('numpy.array_split')
def _array_split(visitor: ProgramVisitor,
                 sdfg: SDFG,
                 state: SDFGState,
                 ary: str,
                 indices_or_sections: Union[symbolic.SymbolicType, List[symbolic.SymbolicType], str],
                 axis: int = 0):
    return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis, allow_uneven=True)


@oprepo.replaces('numpy.dsplit')
def _dsplit(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, ary: str,
            indices_or_sections: Union[symbolic.SymbolicType, List[symbolic.SymbolicType], str]):
    if isinstance(ary, str) and ary in sdfg.arrays:
        if len(sdfg.arrays[ary].shape) < 3:
            raise ValueError('Array dimensionality must be 3 or above for dsplit')
    return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis=2, allow_uneven=False)


@oprepo.replaces('numpy.hsplit')
def _hsplit(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, ary: str,
            indices_or_sections: Union[symbolic.SymbolicType, List[symbolic.SymbolicType], str]):
    if isinstance(ary, str) and ary in sdfg.arrays:
        # In case of a 1D array, split with axis=0
        if len(sdfg.arrays[ary].shape) <= 1:
            return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis=0, allow_uneven=False)
    return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis=1, allow_uneven=False)


@oprepo.replaces('numpy.vsplit')
def _vsplit(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, ary: str,
            indices_or_sections: Union[symbolic.SymbolicType, List[symbolic.SymbolicType], str]):
    return _split_core(visitor, sdfg, state, ary, indices_or_sections, axis=0, allow_uneven=False)
