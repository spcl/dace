# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandGetriPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GETRI.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandGetriOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x),  desc_ipiv, desc_result = node.validate(
            parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(lapack_complex_float*)"
        elif lapack_dtype == 'z':
            cast = "(lapack_complex_double*)"
        if desc_x.dtype.veclen > 1:
            raise(NotImplementedError)


        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getri(LAPACK_ROW_MAJOR, {rows_x}, {cast}_xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetriMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x),  desc_ipiv, desc_result = node.validate(
            parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(MKL_Complex8*)"
        elif lapack_dtype == 'z':
            cast = "(MKL_Complex16*)"
        if desc_x.dtype.veclen > 1:
            raise(NotImplementedError)


        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getri(LAPACK_ROW_MAJOR, {rows_x}, {cast}_xin, {stride_x}, _ipiv);"
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
        "OpenBLAS": ExpandGetriOpenBLAS,
        "MKL": ExpandGetriMKL,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_xin", "_ipiv"},
                         outputs={"_xout", "_res"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, ipiv, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to getri")
        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError("Expected exactly two outputs from getri")

        desc_xin, desc_xout, desc_ipiv, desc_res = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_xin":
                desc_xin = sdfg.arrays[e.data.data]
            if e.dst_conn == "_ipiv":
                desc_ipiv = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_xout":
                desc_xout = sdfg.arrays[e.data.data]
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]
        
        if desc_xin.dtype.base_type != desc_xout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")
        if desc_ipiv.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Pivot input must be an integer array!")

        in_memlets = [None] * 2
        for _, _, _, conn, data in in_edges:
            if conn == '_xin':
                in_memlets[0] = data
            elif conn == '_ipiv':
                in_memlets[1] = data
        
        # Squeeze input memlets
        squeezed_xin = copy.deepcopy(in_memlets[0].subset)
        dims_xin = squeezed_xin.squeeze()
        squeezed_ipiv = copy.deepcopy(in_memlets[1].subset)
        dims_ipiv = squeezed_ipiv.squeeze()

        stride_x = desc_xin.strides[dims_xin[0]]
        shape_xin = squeezed_xin.size()
        rows_x = shape_xin[0]
        cols_x = shape_xin[1]

        return (desc_xin, stride_x, rows_x, cols_x), desc_ipiv, desc_res
