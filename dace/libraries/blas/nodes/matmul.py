# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import properties, symbolic
from copy import deepcopy as dc
from typing import Any, Dict
import warnings
from math import prod


def _get_matmul_operands(node, state, sdfg, name_lhs="_a", name_rhs="_b", name_out="_c"):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_lhs = None
    res_rhs = None
    for edge in state.all_edges(node):
        if edge.dst_conn in [name_lhs, name_rhs]:
            size = edge.data.subset.size()
            squeezed = dc(edge.data.subset)
            squeezed_dims = squeezed.squeeze()
            squeezed_size = squeezed.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            strides = list(outer_array.strides)
            squeezed_strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed_dims]
            res = edge, outer_array, size, strides, squeezed_size, squeezed_strides
            if edge.dst_conn == name_lhs:
                res_lhs = res
            else:
                res_rhs = res
        elif edge.src_conn == name_out:
            size = edge.data.subset.size()
            squeezed = dc(edge.data.subset)
            squeezed_dims = squeezed.squeeze()
            squeezed_size = squeezed.size()
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            strides = list(outer_array.strides)
            squeezed_strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed_dims]
            res_out = edge, outer_array, size, strides, squeezed_size, squeezed_strides
    for res, name in ((res_lhs, name_lhs), (res_rhs, name_rhs), (res_out, name_out)):
        if res is None:
            raise ValueError("Matrix multiplication connector "
                             "\"{}\" not found.".format(name))
    return res_lhs, res_rhs, res_out


def _get_batchmm_opts(a_shape, a_strides, b_shape, b_strides, c_shape, c_strides) -> Dict[str, Any]:
    """
    Detects whether a matrix multiplication is a batched matrix multiplication
    and returns its parameters (strides, batch size), or an empty dictionary if
    batched multiplication is not detected.

    Supports N-dimensional tensors where all leading dimensions (except last 2) are treated
    as batch dimensions. For example:
    - [b, m, k] @ [b, k, n] -> [b, m, n]
    - [b1, b2, m, k] @ [b1, b2, k, n] -> [b1, b2, m, n]  (flattened to [b1*b2, m, k])
    - [b, m, k] @ [k, n] -> [b, m, n]
    - [m, k] @ [b, k, n] -> [b, m, n]

    :param a_shape: Shape of the first tensor.
    :param a_strides: Strides of the first tensor.
    :param b_shape: Shape of the second tensor.
    :param b_strides: Strides of the second tensor.
    :param c_shape: Shape of the output tensor (optional).
    :param c_strides: Strides of the output tensor (optional).
    :return: A dictionary with the following keys: sa,sb,sc (strides for a, b,
             and c); and b (batch size). Empty dict if not batched.
    """
    # Both inputs must be at least 2D, and at least one must have batch dimensions
    if len(a_shape) <= 2 and len(b_shape) <= 2:
        return {}

    # Calculate batch dimensions (all dimensions except last 2)
    a_batch_dims = a_shape[:-2] if len(a_shape) > 2 else ()
    b_batch_dims = b_shape[:-2] if len(b_shape) > 2 else ()
    c_batch_dims = c_shape[:-2] if (c_shape and len(c_shape) > 2) else ()

    # Determine the output batch shape using broadcasting rules
    # Start with the longer batch shape and validate compatibility
    if len(a_batch_dims) >= len(b_batch_dims):
        result_batch_dims = list(a_batch_dims)
        shorter_dims = b_batch_dims
        longer_dims = a_batch_dims
    else:
        result_batch_dims = list(b_batch_dims)
        shorter_dims = a_batch_dims
        longer_dims = b_batch_dims

    # Validate broadcasting compatibility for batch dimensions
    if shorter_dims:
        offset = len(longer_dims) - len(shorter_dims)
        for i, (s_dim, l_dim) in enumerate(zip(shorter_dims, longer_dims[offset:])):
            res = symbolic.equal(s_dim, l_dim)
            if res is False and s_dim != 1 and l_dim != 1:
                raise ValueError(f'Batch dimension mismatch: {s_dim} vs {l_dim} at position {i}')
            if res is None:
                warnings.warn(f'Batch dimension {s_dim} may not match {l_dim} at position {i}', UserWarning)
            # Use the non-1 dimension for broadcasting
            if s_dim == 1 and l_dim != 1:
                result_batch_dims[offset + i] = l_dim
            elif l_dim == 1 and s_dim != 1:
                result_batch_dims[offset + i] = s_dim

    # Calculate total flattened batch size
    batch_size = prod(result_batch_dims) if result_batch_dims else 1

    # Calculate strides for batched operations
    # For a tensor with shape [B1, B2, ..., M, K], the stride for batched operations
    # should be M*K (the size of each matrix) to iterate through all matrices in the flattened batch
    stride_a = 0
    stride_b = 0
    stride_c = 0

    if len(a_shape) > 2:
        # Stride for accessing each matrix: product of last two dimensions (M x K)
        stride_a = prod(a_shape[-2:])

    if len(b_shape) > 2:
        # Stride for accessing each matrix: product of last two dimensions (K x N)
        stride_b = prod(b_shape[-2:])

    if c_shape and len(c_shape) > 2:
        # Stride for accessing each matrix: product of last two dimensions (M x N)
        stride_c = prod(c_shape[-2:])
        # Validate output batch dimensions
        for i, (c_dim, r_dim) in enumerate(zip(c_batch_dims, result_batch_dims)):
            res = symbolic.equal(c_dim, r_dim)
            if res is False:
                raise ValueError(f'Output batch dimension mismatch: {c_dim} vs {r_dim} at position {i}')

    if batch_size == 1 and not result_batch_dims:
        return {}

    return {'sa': stride_a, 'sb': stride_b, 'sc': stride_c, 'b': batch_size, 'batch_dims': result_batch_dims}


def _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func) -> Dict[str, Any]:
    """ Get option map for GEMM code generation (with column-major order). """
    # Avoid import loops
    from dace.codegen.common import sym2cpp
    from dace.libraries.blas.blas_helpers import get_gemm_opts

    (_, _, ashape, astride, _, _), (_, _, bshape, bstride, _, _), (_, _, cshape, cstride, _,
                                                                   _) = _get_matmul_operands(node, state, sdfg)

    if getattr(node, 'transA', False):
        ashape = list(reversed(ashape))
        astride = list(reversed(astride))
    if getattr(node, 'transB', False):
        bshape = list(reversed(bshape))
        bstride = list(reversed(bstride))

    opt = get_gemm_opts(astride, bstride, cstride)
    bopt = _get_batchmm_opts(ashape, astride, bshape, bstride, cshape, cstride)

    opt['x'] = '_a'
    opt['y'] = '_b'
    opt['xdtype'] = adesc.dtype
    opt['ydtype'] = bdesc.dtype
    opt['cdtype'] = cdesc.dtype
    opt['M'] = sym2cpp(ashape[-2])
    opt['N'] = sym2cpp(bshape[-1])
    opt['K'] = sym2cpp(ashape[-1])
    opt['lda'] = sym2cpp(opt['lda'])
    opt['ldb'] = sym2cpp(opt['ldb'])
    opt['ldc'] = sym2cpp(opt['ldc'])

    if opt['swap']:
        if bopt:
            bopt['sa'], bopt['sb'] = bopt['sb'], bopt['sa']
        opt['lda'], opt['ldb'] = opt['ldb'], opt['lda']
        opt['x'], opt['y'] = opt['y'], opt['x']
        opt['xdtype'], opt['ydtype'] = opt['ydtype'], opt['xdtype']
        opt['ta'], opt['tb'] = opt['tb'], opt['ta']
        opt['M'], opt['N'] = opt['N'], opt['M']

    opt['alpha'] = alpha
    opt['beta'] = beta
    opt['dtype'] = cdtype
    opt['func'] = func
    if bopt:
        opt['stride_a'] = sym2cpp(bopt['sa'])
        opt['stride_b'] = sym2cpp(bopt['sb'])
        opt['stride_c'] = sym2cpp(bopt['sc'])
        opt['BATCH'] = sym2cpp(bopt['b'])
    else:
        opt['BATCH'] = None

    return opt


@dace.library.expansion
class SpecializeMatMul(dace.transformation.transformation.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        a, b, c = _get_matmul_operands(node, state, sdfg)
        size_a = a[4]
        size_b = b[4]
        size_c = c[4]

        # Check if this is a batched operation (at least one input has 3+ dimensions)
        is_batched = len(size_a) >= 3 or len(size_b) >= 3

        if len(size_c) == 2 and ((len(size_a) == 2 and len(size_b) == 2) or (len(a[2]) == 2 and len(b[2]) == 2)):
            # Matrix and matrix -> GEMM
            from dace.libraries.blas.nodes.gemm import Gemm
            beta = node.beta
            cin = True
            if '_cin' not in node.in_connectors:
                cin = False
            if c[0].data.wcr:
                from dace.frontend import operations
                redtype = operations.detect_reduction_type(c[0].data.wcr)
                if redtype == dace.dtypes.ReductionType.Sum:
                    beta = 1.0
                    cin = False
                else:
                    warnings.warn("Unsupported WCR in output of MatMul "
                                  "library node: {}".format(c[0].data.wcr))
            gemm = Gemm(node.name + 'gemm', location=node.location, alpha=node.alpha, beta=beta, cin=cin)
            return gemm
        elif is_batched and len(size_a) >= 2 and len(size_b) >= 2:
            # Batched matrix multiplication with broadcasting support
            # Handles: [b, m, k] @ [b, k, n], [b, m, k] @ [k, n], [m, k] @ [b, k, n], [b1, b2, m, k] @ [b1, b2, k, n], etc.
            from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
            result = BatchedMatMul(node.name + 'bmm', location=node.location)
        elif len(size_a) == 2 and len(size_b) == 1:
            # Matrix and vector -> GEMV
            from dace.libraries.blas.nodes.gemv import Gemv
            # Rename inputs to match dot naming
            a[0].dst_conn = "_A"
            b[0].dst_conn = "_x"
            c[0].src_conn = "_y"
            result = Gemv(node.name + 'gemv', location=node.location)
        elif len(size_a) == 1 and len(size_b) == 2:
            # Vector and matrix -> GEMV with transposed matrix
            from dace.libraries.blas.nodes.gemv import Gemv
            # Rename inputs to match dot naming
            a[0].dst_conn = "_x"
            b[0].dst_conn = "_A"
            c[0].src_conn = "_y"
            result = Gemv(node.name + 'gemvt', location=node.location, transA=True)
        elif len(size_a) == 1 and len(size_b) == 1:
            # Vector and vector -> dot product
            from dace.libraries.blas.nodes.dot import Dot
            # Rename inputs to match dot naming
            a[0].dst_conn = "_x"
            b[0].dst_conn = "_y"
            c[0].src_conn = "_result"
            result = Dot(node.name + 'dot', location=node.location)
        else:
            raise NotImplementedError("Matrix multiplication not implemented "
                                      "for shapes: {} and {}".format(size_a, size_b))

        result.alpha = node.alpha
        result.beta = node.beta
        return result


@dace.library.node
class MatMul(dace.sdfg.nodes.LibraryNode):
    """This is a "meta-node" which delegates to different implementations of
       matrix multiplication in the mathematical sense to the appropriate
       computational operators, namely GEMM, batched matrix multiplication,
       GEMV, and DOT."""

    # Global properties
    implementations = {
        "specialize": SpecializeMatMul,
    }
    default_implementation = "specialize"

    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")

    def __init__(self, name, location=None, alpha=1, beta=0):
        self.alpha = alpha
        self.beta = beta
        super().__init__(name, location=location, inputs={"_a", "_b"}, outputs={"_c"})
