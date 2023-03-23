# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandGetrsPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GETRS.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandGetrsOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a, cols_a), (desc_rhs, stride_rhs, rows_rhs,
                                             cols_rhs), desc_ipiv, desc_res = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(lapack_complex_float*)"
        elif lapack_dtype == 'z':
            cast = "(lapack_complex_double*)"
        if desc_a.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getrs(LAPACK_ROW_MAJOR, 'N', {rows_a}, {cols_rhs}, {cast}_a, {stride_a}, _ipiv, {cast}_rhs_in, {stride_rhs});"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetrsMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a, cols_a), (desc_rhs, stride_rhs, rows_rhs,
                                             cols_rhs), desc_ipiv, desc_res = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(MKL_Complex8*)"
        elif lapack_dtype == 'z':
            cast = "(MKL_Complex16*)"
        if desc_a.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getrs(LAPACK_ROW_MAJOR, 'N', {rows_a}, {cols_rhs}, {cast}_a, {stride_a}, _ipiv, {cast}_rhs_in, {stride_rhs});"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetrsCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a, cols_a), (desc_rhs, stride_rhs, rows_rhs,
                                             cols_rhs), desc_ipiv, desc_res = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        veclen = desc_a.dtype.veclen

        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        func = func + 'getrs'

        n = n or node.n
        if veclen != 1:
            n /= veclen

        # NOTE: In the case where the RHS is only a single vector (1D array),
        # cuSOLVER still expects ldb to be the "number of rows"
        if len(desc_rhs.shape) == 1:
            stride_rhs = rows_rhs

        code = (environments.cusolverdn.cuSolverDn.handle_setup_code(node) + f"""
                cusolverDn{func}(
                    __dace_cusolverDn_handle, CUBLAS_OP_N, {rows_a}, {cols_rhs},
                    ({cuda_type}*)_a, {stride_a}, _ipiv, ({cuda_type}*)_rhs_in, {stride_rhs}, _res); 
                """)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dace.int32) if c == '_res' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn

        return tasklet


@dace.library.node
class Getrs(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandGetrsOpenBLAS, "MKL": ExpandGetrsMKL, "cuSolverDn": ExpandGetrsCuSolverDn}
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_a", "_rhs_in", "_ipiv"}, outputs={"_rhs_out", "_res"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A four-tuple (a, rhs, ipiv, res) of the data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 3:
            raise ValueError("Expected exactly three inputs to getrs")
        in_memlets = [None] * 2
        for _, _, _, conn, data in in_edges:
            if conn == '_a':
                in_memlets[0] = data
            elif conn == '_rhs_in':
                in_memlets[1] = data
        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError("Expected exactly two outputs from getrs")
        out_memlets = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        sqdims1 = squeezed1.squeeze()
        squeezed2 = copy.deepcopy(in_memlets[1].subset)
        sqdims2 = squeezed2.squeeze()

        desc_a, desc_rhs_in, desc_rhs_out, desc_ipiv, desc_res = None, None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_a":
                desc_a = sdfg.arrays[e.data.data]
            if e.dst_conn == "_rhs_in":
                desc_rhs_in = sdfg.arrays[e.data.data]
            if e.dst_conn == "_ipiv":
                desc_ipiv = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_res":
                desc_res = sdfg.arrays[e.data.data]
            if e.src_conn == "_rhs_out":
                desc_rhs_out = sdfg.arrays[e.data.data]

        if desc_rhs_in != desc_rhs_out:
            raise ValueError("Input and output must be equal!")
        desc_rhs = desc_rhs_in
        if desc_ipiv.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Pivot input must be an integer array!")

        stride_a = desc_a.strides[sqdims1[0]]
        shape_a = squeezed1.size()
        rows_a = shape_a[0]
        cols_a = shape_a[1]

        stride_rhs = desc_rhs.strides[sqdims2[0]]
        shape_rhs = squeezed2.size()
        rows_rhs = shape_rhs[0]
        if len(desc_rhs.shape) < 2:
            cols_rhs = 1
        else:
            cols_rhs = shape_rhs[1]

        if rows_a != cols_a:
            raise ValueError("Matrix A must be square")

        return (desc_a, stride_a, rows_a, cols_a), (desc_rhs, stride_rhs, rows_rhs, cols_rhs), desc_ipiv, desc_res
