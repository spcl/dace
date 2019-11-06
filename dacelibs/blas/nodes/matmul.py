import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandMatMulPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype):

        m = dace.symbol("m")
        n = dace.symbol("n")
        k = dace.symbol("k")

        @dace.program
        def matmul(_a: dtype[m, k], _b: dtype[k, n], _c: dtype[m, n]):
            _c[:] = _a @ _b

        return matmul.to_sdfg()

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(state, sdfg)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandMatMulPure.make_sdfg(node.dtype)


@dace.library.expansion
class ExpandMatMulOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(state, sdfg)
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
            alpha = "dacelib::blas::BlasConstants::Complex64Pone()"
            beta = "dacelib::blas::BlasConstants::Complex64Zero()"
        elif dtype == dace.complex128:
            func = "zgemm"
            alpha = "dacelib::blas::BlasConstants::Complex128Pone()"
            beta = "dacelib::blas::BlasConstants::Complex128Zero()"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        code = ("cblas_{f}(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
                "m, n, k, {a}, _a, m, _b, k, {b}, beta, _c, m);").format(
                    f=func, a=alpha, b=beta)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandMatMulMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        return ExpandMatMulOpenBLAS.expansion(node, state, sdfg)


@dace.library.expansion
class ExpandMatMulCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(state, sdfg)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "Sgemm"
            alpha = "dacelib::blas::CublasHelper::FloatPone()"
            beta = "dacelib::blas::CublasHelper::FloatZero()"
        elif dtype == dace.float64:
            func = "Dgemm"
            alpha = "dacelib::blas::CublasHelper::DoublePone()"
            beta = "dacelib::blas::CublasHelper::DoubleZero()"
        elif dtype == dace.complex64:
            func = "Cgemm"
            alpha = "dacelib::blas::CublasHelper::Complex64Pone()"
            beta = "dacelib::blas::CublasHelper::Complex64Zero()"
        elif dtype == dace.complex128:
            func = "Zgemm"
            alpha = "dacelib::blas::CublasHelper::Complex128Pone()"
            beta = "dacelib::blas::CublasHelper::Complex128Zero()"
        else:
            raise ValueError("Unsupported type for cuBLAS dot product: " +
                             str(dtype))
        code = ("cublasStatus_t _result = cublas{f}(__dace_cublas_handle, "
                "CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, {a}, _b, m, _a, k, {b}, "
                "_c, m);").format(f=func, a=alpha, b=beta)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class MatMul(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandMatMulPure,
        "OpenBLAS": ExpandMatMulOpenBLAS,
        "MKL": ExpandMatMulMKL,
        "cuBLAS": ExpandMatMulCuBLAS,
    }
    default_implementation = "pure"

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dtype = dtype

    def validate(self, state, sdfg):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to matrix-matrix product")
        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        in_memlets[0].subset.squeeze()
        size0 = in_memlets[0].subset.size()
        if len(size0) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        in_memlets[1].subset.squeeze()
        size1 = in_memlets[1].subset.size()
        if len(size1) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        if size0[1] != size1[0]:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")
        out_memlet.subset.squeeze()
        size2 = out_memlet.subset.size()
        if len(size2) != 2:
            raise ValueError(
                "matrix-matrix product only supported on matrices")
        if list(size2) != [size0[0], size1[1]]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n dimensions")
