from copy import deepcopy as dc
from dace.config import Config
from dace.frontend.common.op_impl import gpu_transform_tasklet
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

        default_implementation = Config.get('frontend', 'implementation')
        Config.set('frontend', 'implementation', value='sdfg')
        sdfg = matmul.to_sdfg()
        Config.set('frontend', 'implementation', value=default_implementation)
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandMatMulPure.make_sdfg(node.dtype)


@dace.library.expansion
class ExpandMatMulOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

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
            alpha = "dacelib::blas::BlasConstants::Get().Complex64Pone()"
            beta = "dacelib::blas::BlasConstants::Get().Complex64Zero()"
        elif dtype == dace.complex128:
            func = "zgemm"
            alpha = "dacelib::blas::BlasConstants::Get().Complex128Pone()"
            beta = "dacelib::blas::BlasConstants::Get().Complex128Zero()"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        for _, _, _, dst_conn, memlet in state.in_edges(node):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                m = size[0]
                k = size[1]
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                n = size[1]
        code = ("cblas_{f}(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
                "{m}, {n}, {k}, {a}, _a, {k}, _b, {n}, {b}, _c, {n});").format(
                    f=func, m=m, n=n, k=k, a=alpha, b=beta)
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
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "Sgemm"
            cast = ""
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).FloatPone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).FloatZero()"
        elif dtype == dace.float64:
            func = "Dgemm"
            cast = ""
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).DoublePone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).DoubleZero()"
        elif dtype == dace.complex64:
            func = "Cgemm"
            cast = "(cuComplex*)"
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex64Pone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex64Zero()"
        elif dtype == dace.complex128:
            func = "Zgemm"
            cast = "(cuDoubleComplex*)"
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex128Pone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex128Zero()"
        else:
            raise ValueError("Unsupported type for cuBLAS dot product: " +
                             str(dtype))
        for _, _, _, dst_conn, memlet in state.in_edges(node):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                m = size[0]
                k = size[1]
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                n = size[1]
        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                "cublasStatus_t _result = cublas{f}(__dace_cublas_handle, "
                "CUBLAS_OP_N, CUBLAS_OP_N, {m}, {n}, {k}, {a}, {c}_a, {m}, "
                "{c}_b, {k}, {b}, {c}_c, {m});").format(
                    f=func, c=cast, m=m, n=n, k=k, a=alpha, b=beta)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        nested_sdfg = dace.SDFG('_cuBLAS_MatMul_')
        A = nested_sdfg.add_array('_a', (m, k), dtype)
        B = nested_sdfg.add_array('_b', (k, n), dtype)
        C = nested_sdfg.add_array('_c', (m, n), dtype)
        AT = nested_sdfg.add_transient('_aT', (k, m), dtype)
        BT = nested_sdfg.add_transient('_bT', (n, k), dtype)
        CT = nested_sdfg.add_transient('_cT', (n, m), dtype)
        nested_state = nested_sdfg.add_state('_cuBLAS_MatMul_')
        acc1 = nested_state.add_read('_a')
        acc2 = nested_state.add_read('_b')
        acc3 = nested_state.add_write('_c')
        acc4 = nested_state.add_access('_aT')
        acc5 = nested_state.add_access('_bT')
        acc6 = nested_state.add_access('_cT')
        from .. import Transpose
        trans_a = Transpose('_Transpose_a', dtype)
        trans_b = Transpose('_Transpose_b', dtype)
        trans_c = Transpose('_Transpose_c', dtype)
        nested_state.add_edge(acc1, None, trans_a, '_inp', dace.Memlet.from_array('_a', A))
        nested_state.add_edge(trans_a, '_out', acc4, None, dace.Memlet.from_array('_aT', AT))
        nested_state.add_edge(acc2, None, trans_b, '_inp', dace.Memlet.from_array('_b', B))
        nested_state.add_edge(trans_b, '_out', acc5, None, dace.Memlet.from_array('_bT', BT))
        nested_state.add_edge(acc6, None, trans_c, '_inp', dace.Memlet.from_array('_cT', CT))
        nested_state.add_edge(trans_c, '_out', acc3, None, dace.Memlet.from_array('_c', C))
        nested_state.add_edge(acc4, None, tasklet, '_a', dace.Memlet.from_array('_aT', AT))
        nested_state.add_edge(acc5, None, tasklet, '_b', dace.Memlet.from_array('_bT', BT))
        nested_state.add_edge(tasklet, '_c', acc6, None, dace.Memlet.from_array('_cT', CT))
        nested_node = dace.graph.nodes.NestedSDFG('_cuBLAS_MatMul_', nested_sdfg, {'_a', '_b'}, {'_c'})
        gpu_transform_tasklet(nested_sdfg, nested_state, tasklet)
        return nested_node

    # @staticmethod
    # def postprocessing(sdfg, state, expansion):
    #     gpu_transform_tasklet(sdfg, state, expansion)


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

    def __init__(self, name, dtype=None, location=None):
        super().__init__(
            name, location=location, inputs={'_a', '_b'}, outputs={'_c'})
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
