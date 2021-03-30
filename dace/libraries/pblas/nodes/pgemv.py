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
class ExpandPgemvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        raise NotImplementedError


@dace.library.expansion
class ExpandPgemvMKL(ExpandTransformation):
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
                f"const char trans = '{node._transa}';\n"
                # f"MKL_INT grows = (trans == 'N' ? _desca[2] : _desca[3]);\n"
                # f"MKL_INT gcols = 1;\n"

                f"MKL_INT grows = (trans == 'N' ? {node._m} : {node._n});\n"
                f"MKL_INT gcols = 1;\n"
                f"MKL_INT a_rows = {node._m};\n"
                f"MKL_INT a_cols = {node._n};\n"
                f"MKL_INT b_rows = (trans == 'N' ? {node._n} : {node._m});\n"
                f"MKL_INT b_cols = 1;\n"
                f"MKL_INT brows = grows / __state->__mkl_scalapack_size;\n"
                f"MKL_INT bcols = 1;\n"
                f"MKL_INT a_brows = _a_block_sizes[0];\n"
                f"MKL_INT a_bcols = (a_cols > 1 ? _a_block_sizes[1]: 1);\n"
                f"MKL_INT b_brows = _b_block_sizes[0];\n"
                f"MKL_INT b_bcols = 1;\n"
                f"MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT a_mloc = numroc( &a_rows, &a_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT b_mloc = numroc( &b_rows, &b_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT info;\n"
                f"MKL_INT _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];\n"
                f"MKL_INT a_lld = a_mloc;\n"
                f"descinit(_a_ldesc, &a_rows, &a_cols, &a_brows, &a_bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);\n"
                f"MKL_INT b_lld = b_mloc;\n"
                f"descinit(_b_ldesc, &b_rows, &b_cols, &b_mloc, &b_bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);\n"
                f"MKL_INT c_lld = mloc;\n"
                f"descinit(_c_ldesc, &grows, &gcols, &mloc, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);\n"

                # f"MKL_INT brows = grows / __state->__mkl_scalapack_size;\n"
                # f"MKL_INT bcols = 1;\n"
                # f"MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                # f"MKL_INT nloc = numroc( &gcols, &bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                # f"MKL_INT gld = grows;\n"
                # f"MKL_INT lld = mloc;\n"
                # f"MKL_INT info;\n"
                # f"descinit(_ldescc, &grows, &gcols, &brows, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &lld, &info);\n"
                # f"MKL_INT _m = _desca[2], _n = _desca[3];\n"
                f"MKL_INT _m = a_rows, _n = a_cols;\n"
                f"p{lapack_dtype_str}gemv(\n"
                f"    &trans, &_m, &_n, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc,\n"
                f"    _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc, &__state->__mkl_int_one,\n"
                f"    &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc, &__state->__mkl_int_one);")
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Pgemv(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ x) + beta * y.
    """

    # Global properties
    implementations = {
        "MKL": ExpandPgemvMKL,
    }
    default_implementation = "MKL"

    def __init__(self, name, transa='N', m=None, n=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                        #  inputs={"_a", "_desca", "_b", "_descb"},
                        #  outputs={"_c", "_gdescc", "_ldescc"},
                         inputs={"_a", "_b", "_a_block_sizes", "_b_block_sizes"},
                         outputs={"_c"},
                         **kwargs)
        self._transa = transa
        self._m = m
        self._n = n

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
        
        # if desca.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("desca must be an integer array")
        # if descb.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("descb must be an integer array")
        # if gdescc.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("descc must be an integer array")
        # if ldescc.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("descc must be an integer array")

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



  