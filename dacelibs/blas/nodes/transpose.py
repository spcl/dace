from copy import deepcopy as dc
from dace.config import Config
from dace.frontend.common.op_impl import gpu_transform_tasklet
import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandTransposePure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype):

        m = dace.symbol("m")
        n = dace.symbol("n")

        @dace.program
        def _transpose(_inp: dtype[m, n], _out: dtype[n, m]):
            _out[:] = transpose(_inp)

        default_implementation = Config.get('frontend', 'implementation')
        Config.set('frontend', 'implementation', value='sdfg')
        sdfg = _transpose.to_sdfg()
        Config.set('frontend', 'implementation', value=default_implementation)
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandTransposePure.make_sdfg(node.dtype)


@dace.library.expansion
class ExpandTransposeOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "somatcopy"
            alpha = "1.0f"
        elif dtype == dace.float64:
            func = "domatcopy"
            alpha = "1.0"
        elif dtype == dace.complex64:
            func = "comatcopy"
            alpha = "dacelib::blas::BlasConstants::Get().Complex64Pone()"
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "dacelib::blas::BlasConstants::Get().Complex128Pone()"
        else:
            raise ValueError("Unsupported type for OpenBLAS omatcopy: " +
                             str(dtype))
        for _, _, _, dst_conn, memlet in state.in_edges(node):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                m = size[0]
                n = size[1]
        code = ("cblas_{f}(CblasRowMajor, CblasTrans, {m}, {n}, {a}, _inp, "
                "{n}, _out, {m});").format(f=func, m=m, n=n, a=alpha)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandTransposeMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "somatcopy"
            alpha = "1.0f"
        elif dtype == dace.float64:
            func = "domatcopy"
            alpha = "1.0"
        elif dtype == dace.complex64:
            func = "comatcopy"
            alpha = "dacelib::blas::BlasConstants::Get().Complex64Pone()"
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "dacelib::blas::BlasConstants::Get().Complex128Pone()"
        else:
            raise ValueError("Unsupported type for MKL omatcopy extension: " +
                             str(dtype))
        for _, _, _, dst_conn, memlet in state.in_edges(node):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                m = size[0]
                n = size[1]
        code = ("mkl_{f}('R', 'T', {m}, {n}, {a}, _inp, "
                "{n}, _out, {m});").format(f=func, m=m, n=n, a=alpha)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandTransposeCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "Sgeam"
            cast = ""
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).FloatPone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).FloatZero()"
        elif dtype == dace.float64:
            func = "Dgeam"
            cast = ""
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).DoublePone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).DoubleZero()"
        elif dtype == dace.complex64:
            func = "Cgeam"
            cast = "(cuComplex*)"
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex64Pone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex64Zero()"
        elif dtype == dace.complex128:
            func = "Zgeam"
            cast = "(cuDoubleComplex*)"
            alpha = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex128Pone()"
            beta = "dacelib::blas::CublasConstants::Get(__dace_cuda_device).Complex128Zero()"
        else:
            raise ValueError("Unsupported type for cuBLAS geam: " +
                             str(dtype))
        for _, _, _, dst_conn, memlet in state.in_edges(node):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                size = subset.size()
                m = size[0]
                n = size[1]
        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                "cublasStatus_t _result = cublas{f}(__dace_cublas_handle, "
                "CUBLAS_OP_T, CUBLAS_OP_N, {m}, {n}, {a}, {c}_inp, {n}, "
                "{b}, {c}_inp, {m}, {c}_out, {m});").format(
                    f=func, c=cast, m=m, n=n, a=alpha, b=beta)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        gpu_transform_tasklet(sdfg, state, expansion)


@dace.library.node
class Transpose(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandTransposePure,
        "OpenBLAS": ExpandTransposeOpenBLAS,
        "MKL": ExpandTransposeMKL,
        "cuBLAS": ExpandTransposeCuBLAS,
    }
    default_implementation = "pure"

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    # location = dace.properties.Property(
    #     dtype=str,
    #     desc="Execution location descriptor (e.g., GPU identifier)",
    #     allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(
            name, location=location, inputs={'_inp'}, outputs={'_out'})
        self.dtype = dtype
        # self.location = location

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError(
                "Expected exactly one input to transpose operation")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                in_size = subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from transpose operation")
        out_memlet = out_edges[0].data
        if len(in_size) != 2:
            raise ValueError(
                "Transpose operation only supported on matrices")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        out_size = out_subset.size()
        if len(out_size) != 2:
            raise ValueError(
                "Transpose operation only supported on matrices")
        if list(out_size) != [in_size[1], in_size[0]]:
            raise ValueError(
                "Output to transpose operation must agree in the m and n dimensions"
            )
