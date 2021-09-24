# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from copy import deepcopy as dc
from typing import Any, Dict, Optional
import warnings


def _get_matmul_operands(node,
                         state,
                         sdfg,
                         name_lhs="_a",
                         name_rhs="_b",
                         name_out="_c"):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_lhs = None
    res_rhs = None
    for edge in state.all_edges(node):
        if edge.dst_conn in [name_lhs, name_rhs]:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [
                s for i, s in enumerate(outer_array.strides) if i in squeezed
            ]
            res = edge, outer_array, size, strides
            if edge.dst_conn == name_lhs:
                res_lhs = res
            else:
                res_rhs = res
        elif edge.src_conn == name_out:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_output_arraynode(state, edge).data)
            strides = [
                s for i, s in enumerate(outer_array.strides) if i in squeezed
            ]
            res_out = edge, outer_array, size, strides
    for res, name in ((res_lhs, name_lhs), (res_rhs, name_rhs), (res_out,
                                                                 name_out)):
        if res is None:
            raise ValueError("Matrix multiplication connector "
                             "\"{}\" not found.".format(name))
    return res_lhs, res_rhs, res_out


def _get_batchmm_opts(a_shape, a_strides, b_shape, b_strides, c_shape,
                      c_strides) -> Dict[str, Any]:
    """
    Detects whether a matrix multiplication is a batched matrix multiplication
    and returns its parameters (strides, batch size), or an empty dictionary if
    batched multiplication is not detected.
    :param a: Data descriptor for the first tensor.
    :param b: Data descriptor for the second tensor.
    :param c: Data descriptor for the output tensor (optional).
    :return: A dictionary with the following keys: sa,sb,sc (strides for a, b,
             and c); and b (batch size).
    """
    if len(a_shape) > 3 or len(b_shape) > 3 or (c_shape and len(c_shape) > 3):
        raise ValueError('Tensor dimensions too large for (batched) matrix '
                         'multiplication')
    if len(a_shape) <= 2 and len(b_shape) <= 2:
        return {}

    batch = None
    stride_a, stride_b, stride_c = 0, 0, 0
    if len(a_shape) == 3:
        batch = a_shape[0]
        stride_a = a_strides[0]
    if len(b_shape) == 3:
        if batch and batch != b_shape[0]:
            raise ValueError('Batch size mismatch for matrix multiplication')
        batch = b_shape[0]
        stride_b = b_strides[0]
    if c_shape and len(c_shape) == 3:
        if batch and batch != c_shape[0]:
            raise ValueError('Batch size mismatch for matrix multiplication')
        batch = c_shape[0]
        stride_c = c_strides[0]

    if batch is None:
        return {}

    return {'sa': stride_a, 'sb': stride_b, 'sc': stride_c, 'b': batch}


def _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta,
                           cdtype, func) -> Dict[str, Any]:
    """ Get option map for GEMM code generation (with column-major order). """
    # Avoid import loops
    from dace.codegen.targets.common import sym2cpp
    from dace.libraries.blas.blas_helpers import get_gemm_opts

    (_, _, ashape, astride), (_, _, bshape, bstride), (
        _, _, cshape, cstride) = _get_matmul_operands(node, state, sdfg)

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
        size_a = a[2]
        size_b = b[2]
        if len(size_a) == 2 and len(size_b) == 2:
            # Matrix and matrix -> GEMM
            from dace.libraries.blas.nodes.gemm import Gemm
            beta = 0.0
            cin = True
            if c[0].data.wcr:
                from dace.frontend import operations
                redtype = operations.detect_reduction_type(c[0].data.wcr)
                if redtype == dace.dtypes.ReductionType.Sum:
                    beta = 1.0
                    cin = False
                else:
                    warnings.warn("Unsupported WCR in output of MatMul "
                                  "library node: {}".format(c[0].data.wcr))
            gemm = Gemm(node.name + 'gemm',
                        location=node.location,
                        alpha=1.0,
                        beta=beta,
                        cin=cin)
            return gemm
        elif len(size_b) == 3 and (len(size_a) in [2, 3]):
            # Batched matrix and matrix -> batched matrix multiplication
            from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
            batched = BatchedMatMul(node.name + 'bmm',
                                    location=node.location)
            return batched
        elif len(size_a) == 2 and len(size_b) == 1:
            # Matrix and vector -> GEMV
            from dace.libraries.blas.nodes.gemv import Gemv
            # Rename inputs to match dot naming
            a[0].dst_conn = "_A"
            b[0].dst_conn = "_x"
            c[0].src_conn = "_y"
            gemv = Gemv(node.name + 'gemv', location=node.location)
            return gemv
        elif len(size_a) == 1 and len(size_b) == 2:
            # Vector and matrix -> GEMV with transposed matrix
            from dace.libraries.blas.nodes.gemv import Gemv
            # Rename inputs to match dot naming
            a[0].dst_conn = "_x"
            b[0].dst_conn = "_A"
            c[0].src_conn = "_y"
            gemv = Gemv(node.name + 'gemvt',
                        location=node.location,
                        transA=True)
            return gemv
        elif len(size_a) == 1 and len(size_b) == 1:
            # Vector and vector -> dot product
            from dace.libraries.blas.nodes.dot import Dot
            # Rename inputs to match dot naming
            a[0].dst_conn = "_x"
            b[0].dst_conn = "_y"
            c[0].src_conn = "_result"
            dot = Dot(node.name + 'dot', location=node.location)
            return dot
        else:
            raise NotImplementedError("Matrix multiplication not implemented "
                                      "for shapes: {} and {}".format(
                                          size_a, size_b))


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

    def __init__(self, name, location=None):
        super().__init__(name,
                         location=location,
                         inputs={"_a", "_b"},
                         outputs={"_c"})
