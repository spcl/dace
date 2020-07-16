import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments
import numpy as np


@dace.library.expansion
class ExpandGetrfPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of GETRF.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGetrfPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGetrfMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        func = to_blastype(dtype.type).lower() + 'getrf'
        (_, adesc, ashape,
         astrides), (_, bdesc, bshape,
                     bstrides), _ = _get_matmul_operands(node, state, sdfg)
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc,
                                     alpha, beta, cdesc.dtype.ctype, func)

        # Adaptations for MKL/BLAS API
        opt['layout'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        code = ("LAPACKE_{func}(CblasColMajor, {ta}, {tb}, "
                "{M}, {N}, {K}, {alpha}, {x}, {lda}, {y}, {ldb}, {beta}, "
                "_c, {ldc});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGetrfPure,
        "MKL": ExpandGetrfMKL,
        "cuBLAS": ExpandGetrfCuBLAS
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self,
                 name,
                 dtype=None,
                 location=None):
        super().__init__(name,
                         location=location,
                         inputs={"_a_in", "_ipiv"},
                         outputs={"_a_out", "_info"})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected 2 inputs to GETRF")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a_in":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_in = subset.size()
            if dst_conn == "_ipiv":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_ipiv = subset.size()

        if len(size_a_in) != 2 or len(size_ipiv) != 1:
            raise ValueError(
                "GETRF supported only on a matrix input A "
                "and an 1D array for pivot indices")

        expected_size_ipiv = min(size_a_in)
        if size_ipiv[0] < expected_size_ipiv:
            raise ValueError("1D array for pivot indices must have length at "
                             "least min(M, N)")

        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError(
                "Expected exactly two outputs from GETRF")
        for _, src_conn, _, _, memlet in state.out_edges(self):
            if src_conn == "_a_out":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_out = subset.size()
            if src_conn == "_info":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_info = subset.size()

        # TODO: Need to validate that input and output are the same matrix

        if len(size_a_out) != 2:
            raise ValueError("GETRF output must be a matrix")
