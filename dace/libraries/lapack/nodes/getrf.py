# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers
from dace.ordered import OrderedSet


@dace.library.expansion
class ExpandGetrfPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GETRF.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandGetrfOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_ipiv, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(lapack_complex_float*)"
        elif lapack_dtype == 'z':
            cast = "(lapack_complex_double*)"
        if desc_x.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getrf(LAPACK_ROW_MAJOR, {rows_x}, {cols_x}, {cast}_xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetrfMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_ipiv, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(MKL_Complex8*)"
        elif lapack_dtype == 'z':
            cast = "(MKL_Complex16*)"
        if desc_x.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getrf(LAPACK_ROW_MAJOR, {rows_x}, {cols_x}, {cast}_xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetrfGPUSolver(ExpandTransformation):
    """LU factorization on a vendor GPU solver.

    Emitted per dialect rather than through a format string for the same reason as ``potrf``:
    cuSolverDn sizes and allocates a workspace first, rocSOLVER takes none.
    """

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_ipiv, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        func = func + 'getrf'

        n = n or node.n
        if veclen != 1:
            n /= veclen

        code = cls.environments[0].handle_setup_code(node) + cls.call(func, cuda_type, rows_x, cols_x, stride_x)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dace.int32) if c == '_res' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn

        return tasklet


@dace.library.expansion
class ExpandGetrfCuSolverDn(ExpandGetrfGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn]

    @classmethod
    def call(cls, func, ctype, rows, cols, stride) -> str:
        return f"""
                int __dace_workspace_size = 0;
                {ctype}* __dace_workspace;
                cusolverDn{func}_bufferSize(
                    __dace_cusolverDn_handle, {rows}, {cols}, ({ctype}*)_xin,
                    {stride}, &__dace_workspace_size);
                gpuMalloc<{ctype}>(
                    &__dace_workspace,
                    sizeof({ctype}) * __dace_workspace_size);
                cusolverDn{func}(
                    __dace_cusolverDn_handle, {rows}, {cols}, ({ctype}*)_xin,
                    {stride}, __dace_workspace, _ipiv, _res);
                gpuFree(__dace_workspace);
                """


@dace.library.expansion
class ExpandGetrfRocSolver(ExpandGetrfGPUSolver):
    environments = [environments.rocsolver.rocSOLVER]

    @classmethod
    def call(cls, func, ctype, rows, cols, stride) -> str:
        # rocSOLVER manages its own workspace, so there is nothing to size, allocate or free.
        return f"""
                dace::lapack::CheckRocsolverError(rocsolver_{func.lower()}(
                    __dace_rocblas_handle, {rows}, {cols}, ({ctype}*)_xin,
                    {stride}, _ipiv, _res));
                """


@dace.library.node
class Getrf(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "OpenBLAS": ExpandGetrfOpenBLAS,
        "MKL": ExpandGetrfMKL,
        "cuSolverDn": ExpandGetrfCuSolverDn,
        "rocSOLVER": ExpandGetrfRocSolver
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_xin"}, outputs=OrderedSet(('_xout', '_ipiv', '_res')), **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, ipiv, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to getrf")
        in_memlets = [in_edges[0].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 3:
            raise ValueError("Expected exactly three outputs from getrf product")
        out_memlet = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        sqdims1 = squeezed1.squeeze()

        desc_xin, desc_xout, desc_ipiv, desc_res = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_xin":
                desc_xin = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_xout":
                desc_xout = sdfg.arrays[e.data.data]
            if e.src_conn == "_ipiv":
                desc_ipiv = sdfg.arrays[e.data.data]
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_xin.dtype.base_type != desc_xout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")
        if desc_ipiv.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Pivot output must be an integer array!")

        stride_x = desc_xin.strides[sqdims1[0]]
        shape_x = squeezed1.size()
        rows_x = shape_x[0]
        cols_x = shape_x[1]

        if len(squeezed1.size()) != 2:
            print(str(squeezed1))
            raise ValueError("getrf only supported on 2-dimensional arrays")

        return (desc_xin, stride_x, rows_x, cols_x), desc_ipiv, desc_res
