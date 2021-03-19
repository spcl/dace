# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional
from dace.data import Array
from dace import dtypes, memlet as mm, properties
from dace.symbolic import symstr
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
import numpy as np
from numbers import Number



@dace.library.expansion
class ExpandPgemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        raise NotImplementedError


@dace.library.expansion
class ExpandPgemmMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKLScaLAPACK]

    @staticmethod
    def expansion(*args, **kwargs):
        (a_in, m, k), (b_in, k, n), (c_in, m, n), alpha, beta, desca, descb, descc = node.validate(parent_sdfg, parent_state)
        lapack_dtype_str = dace.libraries.lapack.utils.LAPACK_DTYPE_CHR(inA.dtype.base_type)
        
        if inbuffer.dtype.veclen > 1:
            raise(NotImplementedError)
 
        # mkl does not have an actual c interface for 
        code = f"p{lapack_dtype_str}gemm('N', 'N', _m, _n, _k, _alpha, _A, 1, 1, _desca, b, 1, 1, _descb, beta, _c, 1, 1, _descc);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Pgemm(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C.
    """

    # Global properties
    implementations = {
        "MKL": ExpandPgemmMKL,
    }
    default_implementation = "MKL"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_a_in", "_desca", "_b_in", "_descb" "_c_in", "_descc", "_alpha", "_beta", "_m", "_n", "_k"},
                         outputs={"_a", "_b", "_c"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple inbuffer, outbuffer of the data descriptors in the
                 parent SDFG.
        """
        a_in, b_in, c_in, alpha, beta, desca, descb, descc, m, n, k = None, None, None, None, None, None, None, None, None, None, None
   
        for e in state.in_edges(self):
            if e.dst_conn == "_a_in":
                a_in = sdfg.arrays[e.data.data]
            if e.dst_conn == "_b_in":
                b_in = sdfg.arrays[e.data.data]
            if e.dst_conn == "_c_in":
                c_in = sdfg.arrays[e.data.data]
            if e.dst_conn == "_desca":
                desca = sdfg.arrays[e.data.data]
            if e.dst_conn == "_descb":
                descb = sdfg.arrays[e.data.data]
            if e.dst_conn == "_descc":
                descc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_alpha":
                alpha = sdfg.arrays[e.data.data]
            if e.dst_conn == "_beta":
                beta = sdfg.arrays[e.data.data]
            if e.dst_conn == "_m":
                m = sdfg.arrays[e.data.data]
            if e.dst_conn == "_n":
                n = sdfg.arrays[e.data.data]
            if e.dst_conn == "_k":
                k = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_a_out":
                a_out = sdfg.arrays[e.data.data]
            if e.src_conn == "_b_out":
                b_out = sdfg.arrays[e.data.data]
            if e.src_conn == "_c_out":
                c_out = sdfg.arrays[e.data.data]

        if a_out != a_in:
            raise ValueError("A is modified in-place, thus A_in and A_out need to point to the same memory.")
        if b_out != b_in:
            raise ValueError("B is modified in-place, thus B_in and B_out need to point to the same memory.")
        if c_out != c_in:
            raise ValueError("C is modified in-place, thus C_in and C_out need to point to the same memory.")
        
        if a_in.dtype.base_type != b_in.dtype.base_type:
            raise ValueError("The type of A and B does not match!")
        if c_in.dtype.base_type != b_in.dtype.base_type:
            raise ValueError("The type of B and C does not match!")
        if c_in.dtype.base_type != alpha.dtype.base_type:
            raise ValueError("The type of C and alpha does not match!")
        if alpha.dtype.base_type != beta.dtype.base_type:
            raise ValueError("The type of alpha and beta does not match!")
        
        if desca.dtype.base_type != dace.dtypes.int32:
            raise ValueError("desca must be an integer array")
        if descb.dtype.base_type != dace.dtypes.int32:
            raise ValueError("descb must be an integer array")
        if descc.dtype.base_type != dace.dtypes.int32:
            raise ValueError("descc must be an integer array")

        if m.dtype.base_type != dace.dtypes.int32:
            raise ValueError("m must be an integer")
        if n.dtype.base_type != dace.dtypes.int32:
            raise ValueError("n must be an integer")
        if k.dtype.base_type != dace.dtypes.int32:
            raise ValueError("k must be an integer")

        return a_in, b_in, c_in, alpha, beta, desca, descb, descc, m, n, k



  