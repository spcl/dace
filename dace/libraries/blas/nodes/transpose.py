# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import functools
from copy import deepcopy as dc
from dace.config import Config
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.blas import blas_helpers
from dace.transformation.transformation import ExpandTransformation
from .. import environments
import warnings


def _get_transpose_input(node, state, sdfg):
    """Returns the transpose input edge, array, and shape."""
    for edge in state.in_edges(node):
        if edge.dst_conn == "_inp":
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            return edge, outer_array, (size[0], size[1])
    raise ValueError("Transpose input connector \"_inp\" not found.")


def _get_transpose_output(node, state, sdfg):
    """Returns the transpose output edge, array, and shape."""
    for edge in state.out_edges(node):
        if edge.src_conn == "_out":
            subset = dc(edge.data.subset)
            subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            return edge, outer_array, (size[0], size[1])
    raise ValueError("Transpose output connector \"_out\" not found.")


@dace.library.expansion
class ExpandTransposePure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        in_edge, in_outer_array, in_shape = _get_transpose_input(node, parent_state, parent_sdfg)
        out_edge, out_outer_array, out_shape = _get_transpose_output(node, parent_state, parent_sdfg)
        dtype = node.dtype

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        _, in_array = sdfg.add_array("_inp",
                                     in_shape,
                                     dtype,
                                     strides=in_outer_array.strides,
                                     storage=in_outer_array.storage)
        _, out_array = sdfg.add_array("_out",
                                      out_shape,
                                      dtype,
                                      strides=out_outer_array.strides,
                                      storage=out_outer_array.storage)

        num_elements = functools.reduce(lambda x, y: x * y, in_array.shape)
        if num_elements == 1:
            inp = state.add_read("_inp")
            out = state.add_write("_out")
            tasklet = state.add_tasklet("transpose", {"__inp"}, {"__out"}, "__out = __inp")
            state.add_edge(inp, None, tasklet, "__inp", dace.memlet.Memlet.from_array("_inp", in_array))
            state.add_edge(tasklet, "__out", out, None, dace.memlet.Memlet.from_array("_out", out_array))
        else:
            state.add_mapped_tasklet(
                name="transpose",
                map_ranges={"__i%d" % i: "0:%s" % n
                            for i, n in enumerate(in_array.shape)},
                inputs={
                    "__inp": dace.memlet.Memlet.simple("_inp",
                                                       ",".join(["__i%d" % i for i in range(len(in_array.shape))]))
                },
                code="__out = __inp",
                outputs={
                    "__out":
                    dace.memlet.Memlet.simple("_out",
                                              ",".join(["__i%d" % i for i in range(len(in_array.shape) - 1, -1, -1)]))
                },
                external_edges=True)

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
            cast = ''
        elif dtype == dace.float64:
            func = "domatcopy"
            alpha = "1.0"
            cast = ''
        elif dtype == dace.complex64:
            func = "comatcopy"
            alpha = "*(MKL_Complex8*)dace::blas::BlasConstants::Get().Complex64Pone()"
            cast = '(MKL_Complex8*)'
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "*(MKL_Complex16*)dace::blas::BlasConstants::Get().Complex128Pone()"
            cast = '(MKL_Complex16*)'
        else:
            warnings.warn("Unsupported type for MKL omatcopy extension: " + str(dtype) + ", falling back to pure")
            return ExpandTransposePure.expansion(node, state, sdfg)

        _, _, (m, n) = _get_transpose_input(node, state, sdfg)
        code = ("mkl_{f}('R', 'T', {m}, {n}, {a}, {cast}_inp, "
                "{n}, {cast}_out, {m});").format(f=func, m=m, n=n, a=alpha, cast=cast)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


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
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
        elif dtype == dace.complex128:
            func = "zomatcopy"
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
        else:
            raise ValueError("Unsupported type for OpenBLAS omatcopy extension: " + str(dtype))
        _, _, (m, n) = _get_transpose_input(node, state, sdfg)
        # Adaptations for BLAS API
        order = 'CblasRowMajor'
        trans = 'CblasTrans'
        code = ("cblas_{f}({o}, {t}, {m}, {n}, {a}, _inp, "
                "{n}, _out, {m});").format(f=func, o=order, t=trans, m=m, n=n, a=alpha)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandTransposeCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
        node.validate(sdfg, state)
        dtype = node.dtype

        func, cdtype, factort = blas_helpers.cublas_type_metadata(dtype)
        func = func + 'geam'

        alpha = f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()"
        beta = f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()"
        _, _, (m, n) = _get_transpose_input(node, state, sdfg)

        code = (environments.cublas.cuBLAS.handle_setup_code(node) + f"""cublas{func}(
                    __dace_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    {m}, {n}, {alpha}, ({cdtype}*)_inp, {n}, {beta}, ({cdtype}*)_inp, {m}, ({cdtype}*)_out, {m});
                """)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.node
class Transpose(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandTransposePure,
        "MKL": ExpandTransposeMKL,
        "OpenBLAS": ExpandTransposeOpenBLAS,
        "cuBLAS": ExpandTransposeCuBLAS
    }
    default_implementation = None

    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name, location=location, inputs={'_inp'}, outputs={'_out'})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to transpose operation")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_inp':
                subset = dc(memlet.subset)
                subset.squeeze()
                in_size = subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from transpose operation")
        out_memlet = out_edges[0].data
        if len(in_size) != 2:
            raise ValueError("Transpose operation only supported on matrices")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        out_size = out_subset.size()
        if len(out_size) != 2:
            raise ValueError("Transpose operation only supported on matrices")
        if list(out_size) != [in_size[1], in_size[0]]:
            raise ValueError("Output to transpose operation must agree in the m and n dimensions")
