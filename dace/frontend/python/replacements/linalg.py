# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains linear algebra function and operator replacements.
"""
import dace  # noqa
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import data, dtypes, symbolic, Memlet, SDFG, SDFGState

import ast
from numbers import Integral
from string import ascii_letters
from typing import Optional, Sequence, Union
import warnings

import numpy as np


def check_batched_matmul_support(visitor: ProgramVisitor, shape_a: Sequence, shape_b: Sequence) -> None:
    """
    Refuses batched matrix multiplications whose batch dimensions do not broadcast.

    NumPy aligns the leading dimensions to the RIGHT and stretches any of extent 1. The pure
    expansion of ``BatchedMatMul`` reproduces that by reading a stretched dimension at 0; the BLAS
    expansions walk a fixed batch stride and refuse a broadcast themselves. What no expansion can
    do is combine dimensions that genuinely differ, so only that is rejected here.

    Supported: ``(*B, M, K) @ (*B, K, N)``, either operand a plain ``(M, K)`` / ``(K, N)`` matrix,
    and broadcast forms such as ``(B, 1, M, K) @ (1, C, K, N) -> (B, C, M, N)``.

    :param visitor: The program visitor, used to locate the error in the source.
    :param shape_a: Shape of the first operand.
    :param shape_b: Shape of the second operand.
    :raise DaceSyntaxError: If a pair of batch dimensions neither matches nor broadcasts.
    """
    batch_a, batch_b = tuple(shape_a[:-2]), tuple(shape_b[:-2])
    if not batch_a or not batch_b:
        return
    offset = len(batch_a) - len(batch_b)
    paired = zip(batch_a[offset:], batch_b) if offset >= 0 else zip(batch_a, batch_b[-offset:])
    for dim_a, dim_b in paired:
        if symbolic.equal(dim_a, dim_b) is False and symbolic.equal(dim_a, 1) is not True and symbolic.equal(
                dim_b, 1) is not True:
            raise DaceSyntaxError(
                visitor, None, f'Batched matmul of shapes {tuple(shape_a)} and {tuple(shape_b)} is not supported: '
                f'batch dimensions {batch_a} and {batch_b} do not broadcast -- {dim_a} and {dim_b} differ and '
                'neither is 1.')


@oprepo.replaces_operator('Array', 'MatMult')
@oprepo.replaces_operator('View', 'MatMult')
@oprepo.replaces_operator('Array', 'MatMult', 'View')
@oprepo.replaces_operator('View', 'MatMult', 'Array')
def _matmult(visitor: ProgramVisitor, sdfg: SDFG, state: SDFGState, op1: str, op2: str):

    from dace.libraries.blas.nodes.matmul import MatMul  # Avoid import loop

    arr1 = sdfg.arrays[op1]
    arr2 = sdfg.arrays[op2]

    if len(arr1.shape) > 1 and len(arr2.shape) > 1:  # matrix * matrix

        # Equalize first: the two dims can reach here as different sympy instances of one name,
        # which symbolic.equal cannot decide and reports as a spurious inconclusive mismatch.
        # pystr_to_symbolic also converts a plain int dim, which equalize_symbols_across itself
        # does not accept (it indexes .free_symbols on its arguments).
        d1 = symbolic.pystr_to_symbolic(arr1.shape[-1])
        d2 = symbolic.pystr_to_symbolic(arr2.shape[-2])
        res = symbolic.equal(*symbolic.equalize_symbols_across(d1, d2))
        if res is None:
            warnings.warn(
                f'Last mode of first tesnsor/matrix {arr1.shape[-1]} and second-last mode of '
                f'second tensor/matrix {arr2.shape[-2]} may not match', UserWarning)
        elif not res:
            raise SyntaxError('Matrix dimension mismatch %s != %s' % (arr1.shape[-1], arr2.shape[-2]))

        from dace.libraries.blas.nodes.matmul import _get_batchmm_opts

        check_batched_matmul_support(visitor, arr1.shape, arr2.shape)

        # Determine batched multiplication (supports N-D tensors)
        bopt = _get_batchmm_opts(arr1.shape, arr1.strides, arr2.shape, arr2.strides, None, None)
        if bopt:
            # Multi-dimensional batch: use batch_dims if available, otherwise use flattened batch size
            batch_dims = bopt.get('batch_dims', [bopt['b']])
            output_shape = tuple(batch_dims) + (arr1.shape[-2], arr2.shape[-1])
        else:
            output_shape = (arr1.shape[-2], arr2.shape[-1])

    elif len(arr1.shape) == 2 and len(arr2.shape) == 1:  # matrix * vector

        d1 = symbolic.pystr_to_symbolic(arr1.shape[-1])
        d2 = symbolic.pystr_to_symbolic(arr2.shape[0])
        res = symbolic.equal(*symbolic.equalize_symbols_across(d1, d2))
        if res is None:
            warnings.warn(
                f'Number of matrix columns {arr1.shape[-1]} and length of vector {arr2.shape[0]} '
                f'may not match', UserWarning)
        elif not res:
            raise SyntaxError("Number of matrix columns {} must match"
                              "size of vector {}.".format(arr1.shape[1], arr2.shape[0]))

        output_shape = (arr1.shape[0], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 2:  # vector * matrix

        d1 = symbolic.pystr_to_symbolic(arr1.shape[0])
        d2 = symbolic.pystr_to_symbolic(arr2.shape[0])
        res = symbolic.equal(*symbolic.equalize_symbols_across(d1, d2))
        if res is None:
            warnings.warn(
                f'Length of vector {arr1.shape[0]} and number of matrix rows {arr2.shape[0]} '
                f'may not match', UserWarning)
        elif not res:
            raise SyntaxError("Size of vector {} must match number of matrix "
                              "rows {} must match".format(arr1.shape[0], arr2.shape[0]))

        output_shape = (arr2.shape[1], )

    elif len(arr1.shape) == 1 and len(arr2.shape) == 1:  # vector * vector

        d1 = symbolic.pystr_to_symbolic(arr1.shape[0])
        d2 = symbolic.pystr_to_symbolic(arr2.shape[0])
        res = symbolic.equal(*symbolic.equalize_symbols_across(d1, d2))
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


@oprepo.replaces('dace.matmul')
@oprepo.replaces('numpy.matmul')
def matmul(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str) -> str:
    """
    ``numpy.matmul(a, b)``. PEP 465 defines it as exactly what ``a @ b`` computes, so the function
    spelling delegates to the operator implementation instead of growing a second one.

    Supported ranks: 1-D x 1-D, 1-D x 2-D, 2-D x 1-D, and N-D x M-D with N, M >= 2, whose leading
    dimensions are batched under the restriction of :func:`check_batched_matmul_support`. Mixing a
    1-D operand with an operand of rank >= 3 is refused, since the promote-batch-demote result
    NumPy computes for it has no implementation here.
    """
    for op in (op_a, op_b):
        if not isinstance(op, str) or op not in sdfg.arrays:
            raise DaceSyntaxError(pv, None, f'Operand "{op}" of numpy.matmul is not an SDFG array')

    rank_a = len(sdfg.arrays[op_a].shape)
    rank_b = len(sdfg.arrays[op_b].shape)
    if (rank_a == 1) != (rank_b == 1) and max(rank_a, rank_b) > 2:
        raise DaceSyntaxError(
            pv, None, f'numpy.matmul of a {rank_a}-D and a {rank_b}-D operand is not supported '
            '(a 1-D operand may only be combined with a 1-D or 2-D one)')

    return _matmult(pv, sdfg, state, op_a, op_b)


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

    if symbolic.inequal_symbols(arr_a.shape[0], arr_b.shape[0]):
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
    left_dims = [arr_a.shape[l] for l in left_axes]
    right_dims = [arr_b.shape[r] for r in right_axes]
    if not symbolic.shapes_equal(left_dims, right_dims):
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


EINSUM_LETTERS = ascii_letters


def einsum_subscripts(ranks: Sequence[int], contracted: int, funcname: str) -> list[str]:
    """Hands out one distinct einsum letter per tensor mode, sharing the LAST ``contracted`` modes.

    :param ranks: Rank of each operand.
    :param contracted: Number of trailing modes the operands contract over pairwise.
    :param funcname: Name reported in the refusal.
    :return: One subscript string per operand.
    :raise ValueError: If the operands together need more modes than einsum has letters.
    """
    free = sum(r - contracted for r in ranks)
    if free + contracted > len(EINSUM_LETTERS):
        raise ValueError(f'{funcname} of rank-{ranks} operands needs more modes than einsum has letters')
    shared = EINSUM_LETTERS[:contracted]
    subs, pos = [], contracted
    for rank in ranks:
        subs.append(EINSUM_LETTERS[pos:pos + rank - contracted] + shared)
        pos += rank - contracted
    return subs


@oprepo.replaces('numpy.outer')
def outer(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str, out=None) -> str:
    """``np.outer(a, b)``: both operands are FLATTENED first, then multiplied against each other.

    That is exactly the ``outer`` method of the ``multiply`` ufunc, which already lowers to a pair
    of nested maps around an elementwise product, so this is the flattening plus a delegation.
    """
    from dace.frontend.python.replacements.array_manipulation import flat  # Avoid import loop
    from dace.frontend.python.replacements.ufunc import implement_ufunc_outer  # Avoid import loop

    if out is not None:
        raise ValueError('numpy.outer(out=...) is not supported; assign the result instead')
    for op in (op_a, op_b):
        if not isinstance(op, str) or op not in sdfg.arrays:
            raise ValueError(f'Operand "{op}" of numpy.outer is not an SDFG array')

    flat_a = flat(pv, sdfg, state, op_a)
    flat_b = flat(pv, sdfg, state, op_b)
    return implement_ufunc_outer(pv, ast.Call(), sdfg, state, 'multiply', [flat_a, flat_b], {})[0]


@oprepo.replaces('numpy.inner')
def inner(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str) -> str:
    """``np.inner(a, b)``: a contraction over the LAST mode of BOTH operands.

    That differs from :func:`dot`, which contracts the last mode of ``a`` against the second-to-last
    of ``b``; the two agree only when ``b`` is 1-D. The result keeps ``a``'s leading modes followed
    by ``b``'s, which is what the einsum subscripts below spell out.
    """
    from dace.frontend.python.replacements.operators import result_type  # Avoid import loop

    for op in (op_a, op_b):
        if not isinstance(op, str) or op not in sdfg.arrays:
            raise ValueError(f'Operand "{op}" of numpy.inner is not an SDFG array')
    desc_a, desc_b = sdfg.arrays[op_a], sdfg.arrays[op_b]

    if isinstance(desc_a, data.Scalar) or isinstance(desc_b, data.Scalar):
        from dace.frontend.python.replacements.ufunc import implement_ufunc  # Avoid import loop
        return implement_ufunc(pv, ast.Call(), sdfg, state, 'multiply', [op_a, op_b])[0]

    rank_a, rank_b = len(desc_a.shape), len(desc_b.shape)
    if symbolic.inequal_symbols(desc_a.shape[-1], desc_b.shape[-1]):
        raise ValueError(f'numpy.inner: last modes {desc_a.shape[-1]} and {desc_b.shape[-1]} must match')
    if rank_a == 1 and rank_b == 1:
        return dot(pv, sdfg, state, op_a, op_b)

    restype, _ = result_type([desc_a, desc_b], 'Mul')
    sub_a, sub_b = einsum_subscripts([rank_a, rank_b], 1, 'numpy.inner')
    spec = f'{sub_a},{sub_b}->{sub_a[:-1]}{sub_b[:-1]}'
    return _einsum(pv, sdfg, state, StringLiteral(spec), op_a, op_b, dtype=restype)


@oprepo.replaces('numpy.kron')
def kron(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, op_a: str, op_b: str) -> str:
    """``np.kron(a, b)``: the outer product with the two operands' modes INTERLEAVED, then merged
    pairwise by a reshape -- ``out[i*P + k, j*Q + l] = a[i, j] * b[k, l]``.

    The interleaving is the whole content of the operation, so it is spelled directly as the einsum
    output subscript ``ikjl``; a rank-1 pair needs no interleaving and is the plain outer product.
    NumPy right-aligns operands of unequal rank by PREPENDING length-1 modes, reproduced here by a
    reshape (which keeps every existing mode, unlike a squeeze).
    """
    from dace.frontend.python.replacements.array_manipulation import reshape  # Avoid import loop
    from dace.frontend.python.replacements.operators import result_type  # Avoid import loop

    for op in (op_a, op_b):
        if not isinstance(op, str) or op not in sdfg.arrays:
            raise ValueError(f'Operand "{op}" of numpy.kron is not an SDFG array')
    desc_a, desc_b = sdfg.arrays[op_a], sdfg.arrays[op_b]
    if isinstance(desc_a, data.Scalar) or isinstance(desc_b, data.Scalar):
        raise ValueError('numpy.kron of a 0-D operand is not supported; multiply instead')

    shape_a, shape_b = list(desc_a.shape), list(desc_b.shape)
    ndim = max(len(shape_a), len(shape_b))
    if len(shape_a) < ndim:
        op_a = reshape(pv, sdfg, state, op_a, [1] * (ndim - len(shape_a)) + shape_a)
        shape_a = list(sdfg.arrays[op_a].shape)
    if len(shape_b) < ndim:
        op_b = reshape(pv, sdfg, state, op_b, [1] * (ndim - len(shape_b)) + shape_b)
        shape_b = list(sdfg.arrays[op_b].shape)

    merged = [da * db for da, db in zip(shape_a, shape_b)]
    if ndim == 1:
        return reshape(pv, sdfg, state, outer(pv, sdfg, state, op_a, op_b), merged)

    restype, _ = result_type([desc_a, desc_b], 'Mul')
    sub_a, sub_b = einsum_subscripts([ndim, ndim], 0, 'numpy.kron')
    interleaved = ''.join(a + b for a, b in zip(sub_a, sub_b))
    spec = f'{sub_a},{sub_b}->{interleaved}'
    return reshape(pv, sdfg, state, _einsum(pv, sdfg, state, StringLiteral(spec), op_a, op_b, dtype=restype), merged)


# out[k] = a[i] * b[j] - a[m] * b[n], one row per component of the 3-vector result.
CROSS_TERMS = ((1, 2, 2, 1), (2, 0, 0, 2), (0, 1, 1, 0))


def cross_component_code(comp: int, dim_a: int, dim_b: int) -> str:
    """Body of one output component of :func:`cross`, with out-of-range 2-vector reads as zero."""
    i, j, m, n = CROSS_TERMS[comp]
    plus = f'__a{i} * __b{j}' if i < dim_a and j < dim_b else ''
    minus = f'__a{m} * __b{n}' if m < dim_a and n < dim_b else ''
    if plus and minus:
        return f'{plus} - {minus}'
    if minus:
        return f'-({minus})'
    return plus or '0'


@oprepo.replaces('numpy.cross')
def cross(pv: ProgramVisitor,
          sdfg: SDFG,
          state: SDFGState,
          op_a: str,
          op_b: str,
          axisa: int = -1,
          axisb: int = -1,
          axisc: int = -1,
          axis: int | None = None) -> str:
    """``np.cross(a, b)`` over the LAST mode, broadcasting the leading modes.

    The cross product is a fixed three-term expression, not a contraction, so it lowers to one
    data-parallel map over the broadcast leading modes whose tasklet reads the components directly.
    A 2-vector operand is the deprecated NumPy special case with an implied zero third component;
    two of them yield the single ``z`` component and therefore an output with NO trailing mode --
    that mode is INDEXED away, which is why it disappears where a length-1 slice would have stayed.
    """
    from dace.frontend.python.replacements.operators import result_type  # Avoid import loop
    from dace.frontend.python.replacements.ufunc import _broadcast  # Avoid import loop

    for op in (op_a, op_b):
        if not isinstance(op, str) or op not in sdfg.arrays:
            raise ValueError(f'Operand "{op}" of numpy.cross is not an SDFG array')
    desc_a, desc_b = sdfg.arrays[op_a], sdfg.arrays[op_b]
    rank_a, rank_b = len(desc_a.shape), len(desc_b.shape)
    if isinstance(desc_a, data.Scalar) or isinstance(desc_b, data.Scalar):
        raise ValueError('numpy.cross needs operands of rank >= 1')
    if axis is not None:
        axisa = axisb = axisc = axis
    if axisa not in (-1, rank_a - 1) or axisb not in (-1, rank_b - 1):
        raise ValueError('numpy.cross is supported for vectors along the LAST mode only')

    dims = []
    for op, desc in ((op_a, desc_a), (op_b, desc_b)):
        dim = desc.shape[-1]
        if not isinstance(dim, Integral) and not (symbolic.issymbolic(dim) and dim.is_Integer):
            raise ValueError(f'numpy.cross needs a compile-time last mode on "{op}", got {dim}')
        dims.append(int(dim))
        if dims[-1] not in (2, 3):
            raise ValueError(f'numpy.cross needs a last mode of 2 or 3 on "{op}", got {dims[-1]}')
    dim_a, dim_b = dims
    ncomp = 1 if dim_a == 2 and dim_b == 2 else 3
    if ncomp == 3 and axisc not in (-1, rank_a - 1, rank_b - 1):
        raise ValueError('numpy.cross is supported for a result along the LAST mode only')

    lead_a, lead_b = list(desc_a.shape[:-1]), list(desc_b.shape[:-1])
    if lead_a or lead_b:
        out_lead, map_range, out_idx, (idx_a, idx_b) = _broadcast([lead_a, lead_b])
        map_range = dict(map_range)
    else:
        out_lead, map_range, out_idx, idx_a, idx_b = (), {}, '', '', ''

    restype, _ = result_type([desc_a, desc_b], 'Mul')
    out_shape = list(out_lead) + ([3] if ncomp == 3 else [])
    if out_shape:
        out, _ = sdfg.add_transient(pv.get_target_name(), out_shape, restype, desc_a.storage, find_new_name=True)
    else:
        out, _ = sdfg.add_scalar(pv.get_target_name(),
                                 restype,
                                 transient=True,
                                 storage=desc_a.storage,
                                 find_new_name=True)

    def indexed(name: str, lead: str, comp: str) -> Memlet:
        return Memlet.simple(name, ', '.join([p for p in (lead, comp) if p]) or '0')

    inputs = {f'__a{i}': indexed(op_a, idx_a, str(i)) for i in range(dim_a)}
    inputs.update({f'__b{i}': indexed(op_b, idx_b, str(i)) for i in range(dim_b)})
    # A 2x2 cross yields only the z component, whose terms are the third row of CROSS_TERMS.
    comps = range(3) if ncomp == 3 else (2, )
    outputs = {f'__c{k}': indexed(out, out_idx, str(k) if ncomp == 3 else '') for k in comps}
    code = '\n'.join(f'__c{k} = {cross_component_code(k, dim_a, dim_b)}' for k in comps)

    if map_range:
        state.add_mapped_tasklet('cross', map_range, inputs, code, outputs, external_edges=True)
        return out

    tasklet = state.add_tasklet('cross', {k: None for k in inputs}, {k: None for k in outputs}, code)
    read_a, read_b, write = state.add_read(op_a), state.add_read(op_b), state.add_write(out)
    for conn, memlet in inputs.items():
        state.add_edge(read_a if conn.startswith('__a') else read_b, None, tasklet, conn, memlet)
    for conn, memlet in outputs.items():
        state.add_edge(tasklet, conn, write, None, memlet)
    return out
