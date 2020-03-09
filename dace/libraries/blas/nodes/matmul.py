from copy import deepcopy as dc
import numpy as np
from dace.config import Config
from dace.frontend.common.op_impl import gpu_transform_tasklet
import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments


def _get_matmul_inputs(node, state, sdfg):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_a = None
    res_b = None
    for edge in state.in_edges(node):
        if edge.dst_conn in ["_a", "_b"]:
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            res = edge, outer_array, (size[0], size[1])
            if edge.dst_conn == "_a":
                res_a = res
            else:
                res_b = res
    if res_a is None or res_b is None:
        raise ValueError("Matrix multiplication input connectors \"_a\" and "
                         "\"_b\" not found.")
    return res_a, res_b


@dace.library.expansion
class ExpandMatMulPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        ((edge_a, outer_array_a, shape_a),
         (edge_b, outer_array_b,
          shape_b)) = _get_matmul_inputs(node, parent_state, parent_sdfg)

        if (len(shape_a) != 2 or len(shape_b) != 2
                or shape_a[1] != shape_b[0]):
            raise SyntaxError('Matrix sizes must match')
        shape_c = (shape_a[0], shape_b[1])

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a,
                                                         dtype_b).type]

        if outer_array_a.storage != outer_array_b.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, storage=storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, storage=storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, storage=storage)

        state.add_mapped_tasklet(
            '_MatMult_', {
                '__i%d' % i: '0:%s' % s
                for i, s in enumerate(
                    [array_a.shape[0], array_b.shape[1], array_a.shape[1]])
            }, {
                '__a': dace.Memlet.simple("_a", '__i0, __i2'),
                '__b': dace.Memlet.simple("_b", '__i2, __i1')
            },
            '__c = __a * __b', {
                '__c':
                dace.Memlet.simple("_c",
                                   '__i0, __i1',
                                   wcr_str='lambda x, y: x + y',
                                   wcr_identity=0)
            },
            external_edges=True)

        sdfg.parent = parent_sdfg
        sdfg.parent_sdfg = parent_sdfg

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandMatMulPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandMatMulMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "sgemm"
            alpha = "1.0f"
            beta = "0.0f"
        elif dtype == dace.float64:
            func = "dgemm"
            alpha = "1.0"
            beta = "0.0"
        elif dtype == dace.complex64:
            func = "cgemm"
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex64Zero()"
        elif dtype == dace.complex128:
            func = "zgemm"
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex128Zero()"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        (_, _, (m, k)), (_, _, (_, n)) = _get_matmul_inputs(node, state, sdfg)
        code = ("cblas_{f}(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
                "{m}, {n}, {k}, {a}, _a, {k}, _b, {n}, {b}, _c, {n});").format(
                    f=func, m=m, n=n, k=k, a=alpha, b=beta)
        tasklet = dace.graph.nodes.Tasklet(node.name,
                                           node.in_connectors,
                                           node.out_connectors,
                                           code,
                                           language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class MatMul(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {"pure": ExpandMatMulPure, "MKL": ExpandMatMulMKL}
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name,
                         location=location,
                         inputs={'_a', '_b'},
                         outputs={'_c'})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError(
                "Expected exactly two inputs to matrix-matrix product")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        if len(size0) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        if len(size1) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        if size0[1] != size1[0]:
            raise ValueError(
                "Inputs to matrix-matrix product must agree in the k-dimension"
            )
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size2 = out_subset.size()
        if len(size2) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        if list(size2) != [size0[0], size1[1]]:
            raise ValueError(
                "Output to matrix-matrix product must agree in the m and n dimensions"
            )
