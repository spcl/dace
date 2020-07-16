import copy
from typing import Any, Dict
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, get_leading_dimension
from .. import environments
import numpy as np


def _get_getrf_opts(inout_dict: Dict[str, Any]) -> Dict[str, Any]:
    
    opt = dict()
    opt['M'] = inout_dict['_a_out'][0]
    opt['N'] = inout_dict['_a_out'][1]
    opt['a'] = '_a_out'
    opt['lda'] = get_leading_dimension(inout_dict['_a_out'], row_major=True)
    opt['ipiv'] = '_ipiv'
    opt['info'] = '_info'

    return opt


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
        inout_dict = node.validate(sdfg, state)
        dtype = node.dtype
        func = to_blastype(dtype.type).lower() + 'getrf'
        opt = _get_getrf_opts(inout_dict)
        opt['func'] = func

        code = ("{info} = LAPACKE_{func}(LAPACK_ROW_MAJOR, {M}, {N}, {a}, "
                "{lda}, {ipiv});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Getrf(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGetrfPure,
        "MKL": ExpandGetrfMKL,
        # "cuBLAS": ExpandGetrfCuBLAS
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
                         inputs={"_a_in"},
                         outputs={"_a_out", "_ipiv", "_info"})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected 1 input to GETRF")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a_in":
                srcname = memlet.data
                srcsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_in = subset.size()

        if len(size_a_in) != 2:
            raise ValueError("GETRF supported only on a matrix input A")

        out_edges = state.out_edges(self)
        if len(out_edges) != 3:
            raise ValueError(
                "Expected 3 outputs from GETRF")
        dstname = None
        for _, src_conn, _get_getrf_opts, _, memlet in state.out_edges(self):
            if src_conn == "_a_out":
                dstname = memlet.data
                dstsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_out = subset.size()
            if src_conn == "_ipiv":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_ipiv = subset.size()
            if src_conn == "_info":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_info = subset.size()

        if srcname != dstname:
            raise ValueError("GETRF input and output must be the same matrix A")

        if len(size_a_out) != 2:
            raise ValueError("GETRF output must be a matrix")

        if srcsubset != dstsubset:
            raise ValueError("GETRF input and output subsets do not match")

        expected_size_ipiv = min(size_a_in)
        if size_ipiv[0] < expected_size_ipiv:
            raise ValueError("1D array for pivot indices must have length at "
                             "least min(M, N)")

        ret = {
            "_a_in": size_a_in,
            "_a_out": size_a_out,
            "_ipiv": size_ipiv,
            "_info": size_info
        }

        return ret
