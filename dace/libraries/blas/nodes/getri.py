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


def _get_getri_opts(inout_dict: Dict[str, Any]) -> Dict[str, Any]:
    
    opt = dict()
    opt['N'] = inout_dict['_a_out'][0]
    opt['a'] = '_a_out'
    opt['lda'] = get_leading_dimension(inout_dict['_a_out'], row_major=True)
    opt['ipiv'] = '_ipiv'
    opt['info'] = '_info'

    return opt


@dace.library.expansion
class ExpandGetriPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of GETRI.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGetriPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGetriMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        inout_dict = node.validate(sdfg, state)
        dtype = node.dtype
        func = to_blastype(dtype.type).lower() + 'getri'
        opt = _get_getri_opts(inout_dict)
        opt['func'] = func

        code = ("{info} = LAPACKE_{func}(LAPACK_ROW_MAJOR, {N}, {a}, "
                "{lda}, {ipiv});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Getri(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGetriPure,
        "MKL": ExpandGetriMKL,
        # "cuBLAS": ExpandGetriCuBLAS
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
            raise ValueError("Expected 2 inputs to GETRI")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a_in":
                srcname = memlet.data
                srcsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_in = subset.size()
            if dst_conn == "_ipiv":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_ipiv = subset.size()

        if len(size_a_in) != 2:
            raise ValueError("GETRI supported only on a matrix input A")

        if size_a_in[0] != size_a_in[1]:
            raise ValueError("GETRI supported only on square matrices")

        expected_size_ipiv = min(size_a_in)
        if size_ipiv[0] < expected_size_ipiv:
            raise ValueError("1D array for pivot indices must have length at N")

        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError(
                "Expected 2 outputs from GETRI")
        for _, src_conn, _get_getrf_opts, _, memlet in state.out_edges(self):
            if src_conn == "_a_out":
                dstname = memlet.data
                dstsubset = memlet.subset
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a_out = subset.size()
            if src_conn == "_info":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_info = subset.size()

        if srcname != dstname:
            raise ValueError("GETRI input and output must be the same matrix A")

        if len(size_a_out) != 2:
            raise ValueError("GETRI output must be a matrix")

        if srcsubset != dstsubset:
            raise ValueError("GETRI input and output subsets do not match")

        ret = {
            "_a_in": size_a_in,
            "_a_out": size_a_out,
            "_ipiv": size_ipiv,
            "_info": size_info
        }

        return ret
