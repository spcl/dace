import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype):

        n = dace.symbol("n")

        @dace.program
        def dot(_x: dtype[n], _y: dtype[n], _result: dtype[1]):
            @dace.map
            def product(i: _[0:n]):
                x_in << _x[i]
                y_in << _y[i]
                result_out >> _result(1, lambda a, b: a + b)
                result_out = x_in * y_in

        return dot.to_sdfg()

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandDotPure.make_sdfg(node.dtype)


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "sdot"
        elif dtype == dace.float64:
            func = "ddot"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        code = "_result = cblas_{}(n, _x, 1, _y, 1);".format(func)
        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandDotMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        return ExpandDotOpenBLAS.expansion(node, state, sdfg)


@dace.library.expansion
class ExpandDotCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "Sdot"
        elif dtype == dace.float64:
            func = "Ddot"
        else:
            raise ValueError("Unsupported type for cuBLAS dot product: " +
                             str(dtype))

        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                "cublas{func}(__dace_cublas_handle, n, ___x.ptr<1>(), 1, "
                "___y.ptr<1>(), 1, ___result.ptr<1>());".format(func=func))

        tasklet = dace.graph.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.node
class Dot(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, *args, **kwargs):
        super().__init__(
            name, *args, inputs={"_x", "_y"}, outputs={"_result"}, **kwargs)
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()
        veclen = in_memlets[0].veclen
        if len(size) != 1:
            raise ValueError(
                "dot product only supported on 1-dimensional arrays")
        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to dot product must have equal size")
        if out_memlet.subset.num_elements() != 1 or out_memlet.veclen != 1:
            raise ValueError("Output of dot product must be a single element")
        if veclen != in_memlets[1].veclen:
            raise ValueError(
                "Vector lengths of inputs to dot product must be identical")
        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on dot product memlets not supported")
