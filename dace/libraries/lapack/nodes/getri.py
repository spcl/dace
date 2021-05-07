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

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x, rows_x, cols_x),  desc_ipiv, desc_result = node.validate(
            parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        lapack_dtype = "X"
        if dtype == dace.dtypes.float32:
            lapack_dtype = "s"
        elif dtype == dace.dtypes.float64:
            lapack_dtype = "d" 
        elif dtype == dace.dtypes.complex64:
            lapack_dtype = "c"
        elif dtype == dace.dtypes.complex128:
            lapack_dtype = "z"
        else:
            print("The datatype "+str(dtype)+" is not supported!")
            raise(NotImplementedError) 
        if desc_x.dtype.veclen > 1:
            raise(NotImplementedError)


        n = n or node.n
        code = f"_res = LAPACKE_{lapack_dtype}getri(LAPACK_ROW_MAJOR, {rows_x}, _xin, {stride_x}, _ipiv);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGetriMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGetriOpenBLAS.expansion(*args, **kwargs)


@dace.library.node
class Getri(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "OpenBLAS": ExpandGetriOpenBLAS,
        "MKL": ExpandGetriMKL,
    }
    default_implementation = ExpandGetriOpenBLAS

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
        in_memlets = [in_edges[0].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 2:
            raise ValueError("Expected exactly two outputs from getri")
        out_memlets = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        sqdims1 = squeezed1.squeeze()
        
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

        stride_x = desc_xin.strides[sqdims1[0]]
        shape_x = squeezed1.size()
        rows_x = shape_x[0]
        cols_x = shape_x[1]

        return (desc_xin, stride_x, rows_x, cols_x), desc_ipiv, desc_res

