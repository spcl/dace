# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains linear algebra function and operator replacements.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

import ast
from numbers import Integral
from typing import Optional, Sequence, Union
import warnings

import numpy as np


@oprepo.replaces_operator('Array', 'MatMult')
@oprepo.replaces_operator('View', 'MatMult')
@oprepo.replaces_operator('Array', 'MatMult', 'View')
@oprepo.replaces_operator('View', 'MatMult', 'Array')
def _matmult(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):

    from dace.libraries.blas.nodes.matmul import MatMul  # Avoid import loop

    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    if len(arr1.shape) > 1 and len(arr2.shape) > 1:  # matrix * matrix

        res = symbolic.equal(arr1.shape[-1], arr2.shape[-2])
        if res is None:
            warnings.warn(
                f'Last mode of first tesnsor/matrix {arr1.shape[-1]} and second-last mode of '
                f'second tensor/matrix {arr2.shape[-2]} may not match', UserWarning)
        elif not res:
            raise SyntaxError('Matrix dimension mismatch %s != %s' % (arr1.shape[-1], arr2.shape[-2]))

        from dace.libraries.blas.nodes.matmul import _get_batchmm_opts

        # Determine batched multiplication (supports N-D tensors)
        bopt = _get_batchmm_opts(arr1.shape, arr1.strides, arr2.shape, arr2.strides, None, None)
        if bopt:
            # Multi-dimensional batch: use batch_dims if available, otherwise use flattened batch size
            batch_dims = bopt.get('batch_dims', [bopt['b']])
            output_shape = tuple(batch_dims) + (arr1.shape[-2], arr2.shape[-1])
        else:
            output_shape = (arr1.shape[-2], arr2.shape[-1])

    elif len(arr1.shape) == 2 and len(arr2.shape) == 1:  # matrix * vector

        res = symbolic.equal(arr1.shape[-1], arr2.shape[0])
        if res is None:
            warnings.warn(
                f'Number of matrix columns {arr1.shape[-1]} and length of vector {arr2.shape[0]} '
                f'may not match', UserWarning)
        elif not res:
            raise SyntaxError("Number of matrix columns {} must match"
                              "size of vector {}.".format(arr1.shape[1], arr2.shape[0]))

        output_shape = (arr1.shape[0], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 2:  # vector * matrix

        res = symbolic.equal(arr1.shape[0], arr2.shape[0])
        if res is None:
            warnings.warn(
                f'Length of vector {arr1.shape[0]} and number of matrix rows {arr2.shape[0]} '
                f'may not match', UserWarning)
        elif not res:
            raise SyntaxError("Size of vector {} must match number of matrix "
                              "rows {} must match".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (arr2.shape[1], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 1:  # vector * vector

        res = symbolic.equal(arr1.shape[0], arr2.shape[0])
        if res is None:
            warnings.warn(
                f'Length of first vector {arr1.shape[0]} and length of second vector {arr2.shape[0]} '
                f'may not match', UserWarning)
        elif not res:
            raise SyntaxError("Vectors in vector product must have same size: "
                              "{} vs. {}".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (1, )

    else:  # Dunno what this is, bail

        raise SyntaxError("Cannot multiply arrays with shapes: {} and {}".format(arr1.shape, arr2.shape))

    type1 = arr1.dtype.type
    type2 = arr2.dtype.type
    restype = dtypes.dtype_to_typeclass(np.result_type(type1, type2).type)

    op3, arr3 = sdfg.add_transient(visitor.get_target_name(), output_shape, restype, arr1.storage, find_new_name=True)

    acc1 = state.add_read(op1)
    acc2 = state.add_read(op2)
    acc3 = state.add_write(op3)

    tasklet = MatMul('_MatMult_')
    state.add_node(tasklet)
    state.add_edge(acc1, None, tasklet, '_a', Memlet.from_array(op1, arr1))
    state.add_edge(acc2, None, tasklet, '_b', Memlet.from_array(op2, arr2))
    state.add_edge(tasklet, '_c', acc3, None, Memlet.from_array(op3, arr3))

    return op3


@oprepo.replaces('dace.dot')
@oprepo.replaces('numpy.dot')
def dot(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str, op_out=None):
    from dace.frontend.python.replacements.ufunc import implement_ufunc
    from dace.frontend.python.replacements.operators import result_type

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
        return implement_ufunc(pv, node, sdfg, state, ufunc_name, args)

    if len(arr_a.shape) > 2 or len(arr_b.shape) > 2:
        raise NotImplementedError

    if arr_a.shape[0] != arr_b.shape[0]:
        raise SyntaxError()

    if op_out:
        if not isinstance(op_out, str) or not op_out in sdfg.arrays.keys():
            raise SyntaxError()
    else:
        # Infer result type
        restype, _ = result_type([arr_a, arr_b], 'Mul')
        op_out = pv.get_target_name()
        op_out, _ = sdfg.add_scalar(op_out, restype, transient=True, storage=arr_a.storage, find_new_name=True)

    arr_out = sdfg.arrays[op_out]

    from dace.libraries.blas.nodes.dot import Dot  # Avoid import loop

    acc_a = state.add_read(op_a)
    acc_b = state.add_read(op_b)
    acc_out = state.add_write(op_out)

    tasklet = Dot('_Dot_')
    state.add_node(tasklet)
    state.add_edge(acc_a, None, tasklet, '_x', Memlet.from_array(op_a, arr_a))
    state.add_edge(acc_b, None, tasklet, '_y', Memlet.from_array(op_b, arr_b))
    state.add_edge(tasklet, '_result', acc_out, None, Memlet.from_array(op_out, arr_out))

    return op_out


@oprepo.replaces('dace.linalg.inv')
@oprepo.replaces('numpy.linalg.inv')
def _inv(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, inp_op: str):

    if not isinstance(inp_op, str) or not inp_op in sdfg.arrays.keys():
        raise SyntaxError()

    inp_arr = sdfg.arrays[inp_op]
    out_arr = sdfg.add_transient(pv.get_target_name(),
                                 inp_arr.shape,
                                 inp_arr.dtype,
                                 storage=inp_arr.storage,
                                 find_new_name=True)

    from dace.libraries.linalg import Inv

    inp = state.add_read(inp_op)
    out = state.add_write(out_arr[0])
    inv_node = Inv("inv", overwrite_a=False, use_getri=True)

    state.add_memlet_path(inp, inv_node, dst_conn="_ain", memlet=Memlet.from_array(inp_op, inp_arr))
    state.add_memlet_path(inv_node, out, src_conn="_aout", memlet=Memlet.from_array(*out_arr))

    return out_arr[0]


@oprepo.replaces('dace.linalg.solve')
@oprepo.replaces('numpy.linalg.solve')
def _solve(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str):

    for op in (op_a, op_b):
        if not isinstance(op, str) or not op in sdfg.arrays.keys():
            raise SyntaxError()

    a_arr = sdfg.arrays[op_a]
    b_arr = sdfg.arrays[op_b]
    out_arr = pv.add_temp_transient(b_arr.shape, b_arr.dtype, storage=b_arr.storage)

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
def _inv(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, inp_op: str):

    if not isinstance(inp_op, str) or not inp_op in sdfg.arrays.keys():
        raise SyntaxError()

    inp_arr = sdfg.arrays[inp_op]
    out_arr = pv.add_temp_transient(inp_arr.shape, inp_arr.dtype, storage=inp_arr.storage)

    from dace.libraries.linalg import Cholesky

    inp = state.add_read(inp_op)
    out = state.add_write(out_arr[0])
    chlsky_node = Cholesky("cholesky", lower=True)

    state.add_memlet_path(inp, chlsky_node, dst_conn="_a", memlet=Memlet.from_array(inp_op, inp_arr))
    state.add_memlet_path(chlsky_node, out, src_conn="_b", memlet=Memlet.from_array(*out_arr))

    return out_arr[0]


@oprepo.replaces('dace.tensordot')
@oprepo.replaces('numpy.tensordot')
def _tensordot(pv: 'ProgramVisitor',
               sdfg: SDFG,
               state: SDFGState,
               op_a: str,
               op_b: str,
               axes: Union[int, Sequence[int]] = 2,
               out_axes: Sequence[int] = None):

    # NOTE: `out_axes` is a non-standard extension to `numpy.tensordot`, allowing trasposition of the output

    for op in (op_a, op_b):
        if not isinstance(op, str) or not op in sdfg.arrays.keys():
            raise SyntaxError()

    arr_a = sdfg.arrays[op_a]
    arr_b = sdfg.arrays[op_b]

    if isinstance(axes, Integral):
        left_axes = list(range(len(arr_a.shape) - axes, len(arr_a.shape)))
        right_axes = list(range(0, axes))
    else:
        left_axes = axes[0]
        right_axes = axes[1]

    # Some validation (more detailed validation is done inside the TensorDot library node)
    if any(a >= len(arr_a.shape) or a < 0 for a in left_axes):
        raise ValueError("Axes for left tensor are out-of-bounds.")
    if any(a >= len(arr_b.shape) or a < 0 for a in right_axes):
        raise ValueError("Axes for right tensor are out-of-bounds.")
    if len(left_axes) != len(right_axes):
        raise ValueError("The input tensors must have the same number of contracting modes.")
    if any(arr_a.shape[l] != arr_b.shape[r] for l, r in zip(left_axes, right_axes)):
        raise ValueError("The input tensors' contracting modes must have the same length.")

    dot_shape = [s for i, s in enumerate(arr_a.shape) if i not in left_axes]
    dot_shape.extend([s for i, s in enumerate(arr_b.shape) if i not in right_axes])

    if out_axes:
        if list(sorted(out_axes)) != list(range(len(dot_shape))):
            raise ValueError("Output axes is not a permutation of the output's modes.")
        dot_shape = [dot_shape[i] for i in out_axes]

    op_c, arr_c = pv.add_temp_transient(dot_shape, arr_a.dtype, storage=arr_a.storage)

    from dace.libraries.linalg import TensorDot
    a = state.add_read(op_a)
    b = state.add_read(op_b)
    c = state.add_write(op_c)
    tasklet = TensorDot("_TensorDot_", left_axes, right_axes, out_axes)
    state.add_edge(a, None, tasklet, '_left_tensor', Memlet.from_array(op_a, arr_a))
    state.add_edge(b, None, tasklet, '_right_tensor', Memlet.from_array(op_b, arr_b))
    state.add_edge(tasklet, '_out_tensor', c, None, Memlet.from_array(op_c, arr_c))

    return op_c


@oprepo.replaces('numpy.einsum')
def _einsum(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            einsum_string: StringLiteral,
            *arrays: str,
            dtype: Optional[dtypes.typeclass] = None,
            optimize: bool = False,
            output: Optional[str] = None,
            alpha: Optional[symbolic.SymbolicType] = 1.0,
            beta: Optional[symbolic.SymbolicType] = 0.0):
    from dace.frontend.common.einsum import create_einsum_sdfg
    return create_einsum_sdfg(sdfg,
                              state,
                              str(einsum_string),
                              *arrays,
                              dtype=dtype,
                              optimize=optimize,
                              output=output,
                              output_name=pv.get_target_name(),
                              alpha=alpha,
                              beta=beta)
