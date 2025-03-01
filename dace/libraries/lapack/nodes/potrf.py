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


@dace.library.expansion
class ExpandPotrfPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK POTRF.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandPotrfOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        if desc_x.dtype.veclen > 1:
            raise (NotImplementedError)

        n = n or node.n
        uplo = "'L'" if node.lower else "'U'"
        code = f"_res = LAPACKE_{lapack_dtype}potrf(LAPACK_ROW_MAJOR, {uplo}, {rows_x}, _xin, {stride_x});"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandPotrfMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandPotrfOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandPotrfCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x), desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        func = func + 'potrf'

        n = n or node.n
        if veclen != 1:
            n /= veclen
        uplo = "CUBLAS_FILL_MODE_LOWER" if node.lower else "CUBLAS_FILL_MODE_UPPER"

        code = (environments.cusolverdn.cuSolverDn.handle_setup_code(node) + f"""
                int __dace_workspace_size = 0;
                {cuda_type}* __dace_workspace;
                cusolverDn{func}_bufferSize(
                    __dace_cusolverDn_handle, {uplo}, {rows_x}, _xin,
                    {stride_x}, &__dace_workspace_size);
                cudaMalloc<{cuda_type}>(
                    &__dace_workspace,
                    sizeof({cuda_type}) * __dace_workspace_size);
                cusolverDn{func}(
                    __dace_cusolverDn_handle, {uplo}, {rows_x}, _xin,
                    {stride_x}, __dace_workspace, __dace_workspace_size, _res);
                cudaFree(__dace_workspace);
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
class Potrf(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandPotrfOpenBLAS, "MKL": ExpandPotrfMKL, "cuSolverDn": ExpandPotrfCuSolverDn}
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)
    lower = dace.properties.Property(dtype=bool, default=True)

    def __init__(self, name, lower=True, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_xin"}, outputs={"_xout", "_res"}, **kwargs)
        self.lower = lower

    def validate(self, sdfg, state):
        """
        :return: A two-tuple (x, res) of the two data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to potrf")
        in_memlets = [in_edges[0].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError("Expected exactly two outputs from potrf product")
        out_memlet = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        sqdims1 = squeezed1.squeeze()

        desc_xin, desc_xout, desc_res = None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_xin":
                desc_xin = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_xout":
                desc_xout = sdfg.arrays[e.data.data]
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_xin.dtype.base_type != desc_xout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")

        stride_x = desc_xin.strides[sqdims1[0]]
        shape_x = squeezed1.size()
        rows_x = shape_x[0]
        cols_x = shape_x[1]

        if len(squeezed1.size()) != 2:
            print(str(squeezed1))
            raise ValueError("potrf only supported on 2-dimensional arrays")

        return (desc_xin, stride_x, rows_x, cols_x), desc_res
