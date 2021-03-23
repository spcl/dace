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
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        from dace.libraries.lapack import utils
        lapack_dtype_str = utils.LAPACK_DTYPE_CHR(a.dtype.base_type)
        
        # if inbuffer.dtype.veclen > 1:
        #     raise(NotImplementedError)
 
        # mkl does not have an actual c interface for 
        code = (f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"
                f"MKL_INT grows = _desca[2];\n"
                f"MKL_INT gcols = _descb[3];\n"
                f"MKL_INT gld = _desca[2];\n"
                f"MKL_INT lld = _desca[4];\n"
                # f"MKL_INT gld = _descb[2];\n"
                # f"MKL_INT lld = _descb[4];\n"
                f"MKL_INT info;\n"
                f"descinit(_gdescc, &grows, &gcols, &grows, &gcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &gld, &info);\n"
                f"descinit(_ldescc, &grows, &gcols, &_desca[4], &_descb[5], &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &lld, &info);\n"
                f"MKL_INT _m = grows, _n = gcols, _k = _desca[3];\n"
                # f"MKL_INT _m = grows, _n = gcols, _k = _desca[2];\n"
                f"p{lapack_dtype_str}gemm(\n"
                # f"    &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _descb,\n"
                # f"    _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _desca, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescc);")
                f"    &trans, &trans, &_m, &_n, &_k, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _desca,\n"
                f"    _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _descb, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescc);")
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
                         inputs={"_a", "_desca", "_b", "_descb"},
                         outputs={"_c", "_gdescc", "_ldescc"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple inbuffer, outbuffer of the data descriptors in the
                 parent SDFG.
        """
        a, b, c, desca, descb, gdescc, ldesc = None, None, None, None, None, None, None
   
        for e in state.in_edges(self):
            if e.dst_conn == "_a":
                a = sdfg.arrays[e.data.data]
            if e.dst_conn == "_b":
                b = sdfg.arrays[e.data.data]
            if e.dst_conn == "_desca":
                desca = sdfg.arrays[e.data.data]
            if e.dst_conn == "_descb":
                descb = sdfg.arrays[e.data.data]
            # if e.dst_conn == "_alpha":
            #     alpha = sdfg.arrays[e.data.data]
            # if e.dst_conn == "_beta":
            #     beta = sdfg.arrays[e.data.data]

        for e in state.out_edges(self):
            if e.src_conn == "_gdescc":
                gdescc = sdfg.arrays[e.data.data]
            if e.src_conn == "_ldescc":
                ldescc = sdfg.arrays[e.data.data]
            if e.src_conn == "_c":
                c = sdfg.arrays[e.data.data]

        # if a_out != a_in:
        #     raise ValueError("A is modified in-place, thus A_in and A_out need to point to the same memory.")
        # if b_out != b_in:
        #     raise ValueError("B is modified in-place, thus B_in and B_out need to point to the same memory.")
        # if c_out != c_in:
        #     raise ValueError("C is modified in-place, thus C_in and C_out need to point to the same memory.")
        
        if a.dtype.base_type != b.dtype.base_type:
            raise ValueError("The type of A and B does not match!")
        if c.dtype.base_type != b.dtype.base_type:
            raise ValueError("The type of B and C does not match!")
        # if c.dtype.base_type != alpha.dtype.base_type:
        #     raise ValueError("The type of C and alpha does not match!")
        # if alpha.dtype.base_type != beta.dtype.base_type:
        #     raise ValueError("The type of alpha and beta does not match!")
        
        if desca.dtype.base_type != dace.dtypes.int32:
            raise ValueError("desca must be an integer array")
        if descb.dtype.base_type != dace.dtypes.int32:
            raise ValueError("descb must be an integer array")
        if gdescc.dtype.base_type != dace.dtypes.int32:
            raise ValueError("descc must be an integer array")
        if ldescc.dtype.base_type != dace.dtypes.int32:
            raise ValueError("descc must be an integer array")

        # if m.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("m must be an integer")
        # if n.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("n must be an integer")
        # if k.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("k must be an integer")

        # m = a.shape[0]
        # assert(m == c.shape[0])
        # k = a.shape[-1]
        # assert(k == b.shape[0])
        # n = b.shape[-1]
        # assert(n == c.shape[0])

        return a, b, c, desca, descb, gdescc, ldesc



  