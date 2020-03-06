import functools
from copy import deepcopy as dc
from dace.config import Config
from dace.frontend.common.op_impl import gpu_transform_tasklet
import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments


def _get_transpose_input(node, state, sdfg):
    """Returns the transpose input edge, array, and shape."""
    for edge in state.in_edges(node):
        if edge.dst_conn == "_inp":
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            return edge, outer_array, (size[0], size[1])
    raise ValueError("Transpose input connector \"_inp\" not found.")


@dace.library.expansion
class ExpandTransposePure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        in_edge, outer_array, in_shape = _get_transpose_input(
            node, parent_state, parent_sdfg)
        out_shape = (in_shape[1], in_shape[0])
        dtype = node.dtype

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        _, in_array = sdfg.add_array("_inp",
                                     in_shape,
                                     dtype,
                                     storage=outer_array.storage)
        _, out_array = sdfg.add_array("_out",
                                      out_shape,
                                      dtype,
                                      storage=outer_array.storage)

        num_elements = functools.reduce(lambda x, y: x * y, in_array.shape)
        if num_elements == 1:
            inp = state.add_read("_inp")
            out = state.add_write("_out")
            tasklet = state.add_tasklet("transpose", {"__inp"}, {"__out"},
                                        "__out = __inp")
            state.add_edge(inp, None, tasklet, "__inp",
                           dace.memlet.Memlet.from_array("_inp", in_array))
            state.add_edge(tasklet, "__out", out, None,
                           dace.memlet.Memlet.from_array("_out", outarr))
        else:
            state.add_mapped_tasklet(
                name="transpose",
                map_ranges={
                    "__i%d" % i: "0:%s" % n
                    for i, n in enumerate(in_array.shape)
                },
                inputs={
                    "__inp":
                    dace.memlet.Memlet.simple(
                        "_inp", ",".join(
                            ["__i%d" % i for i in range(len(in_array.shape))]))
                },
                code="__out = __inp",
                outputs={
                    "__out":
                    dace.memlet.Memlet.simple(
                        "_out", ",".join([
                            "__i%d" % i
                            for i in range(len(in_array.shape) - 1, -1, -1)
                        ]))
                },
                external_edges=True)

        sdfg.parent = parent_sdfg
        sdfg.parent_sdfg = parent_sdfg

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandTransposePure.make_sdfg(node, state, sdfg)


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
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
        else:
            raise ValueError("Unsupported type for MKL omatcopy extension: " +
                             str(dtype))
        _, _, (m, n) = _get_transpose_input(node, state, sdfg)
        code = ("mkl_{f}('R', 'T', {m}, {n}, {a}, _inp, "
                "{n}, _out, {m});").format(f=func, m=m, n=n, a=alpha)
        tasklet = dace.graph.nodes.Tasklet(node.name,
                                           node.in_connectors,
                                           node.out_connectors,
                                           code,
                                           language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Transpose(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandTransposePure,
        "MKL": ExpandTransposeMKL,
    }
    default_implementation = None

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name,
                         location=location,
                         inputs={'_inp'},
                         outputs={'_out'})
        self.dtype = dtype

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
            raise ValueError("Transpose operation only supported on matrices")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        out_size = out_subset.size()
        if len(out_size) != 2:
            raise ValueError("Transpose operation only supported on matrices")
        if list(out_size) != [in_size[1], in_size[0]]:
            raise ValueError(
                "Output to transpose operation must agree in the m and n dimensions"
            )
