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
        code = (f"if (!__state->__mkl_scalapack_grid_init) {{\n"
                f"    __state->__mkl_scalapack_prows = Py;\n"
                f"    __state->__mkl_scalapack_pcols = Px;\n"
                f"    blacs_gridinit(&__state->__mkl_scalapack_context, \"C\", &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols);\n"
                f"    blacs_gridinfo(&__state->__mkl_scalapack_context, &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols, &__state->__mkl_scalapack_myprow, &__state->__mkl_scalapack_mypcol);\n"
                f"    __state->__mkl_scalapack_grid_init = true;\n"
                f"}}\n"
                f"const double  zero = 0.0E+0, one = 1.0E+0;\n"
                f"const char trans = 'N';\n"

                f"MKL_INT grows = {node._m};\n"
                f"MKL_INT gcols = {node._n};\n"
                f"MKL_INT a_cols = {node._k};\n"
                f"MKL_INT b_rows = {node._k};\n"
                f"MKL_INT brows = _a_block_sizes[0];\n"
                f"MKL_INT bcols = (gcols > 1 ? _b_block_sizes[1]: 1);\n"
                f"MKL_INT a_brows = _a_block_sizes[0];\n"
                f"MKL_INT a_bcols = (a_cols > 1 ? _a_block_sizes[1]: 1);\n"
                f"MKL_INT b_brows = _b_block_sizes[0];\n"
                f"MKL_INT b_bcols = _b_block_sizes[1];\n"
                f"MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT nloc = numroc( &gcols, &bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                # f"MKL_INT mloc = numroc( &grows, &bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                # f"MKL_INT nloc = numroc( &gcols, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                f"MKL_INT akloc = numroc( &a_cols, &a_bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                f"MKL_INT bkloc = numroc( &b_rows, &b_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                # f"MKL_INT akloc = numroc( &a_cols, &a_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);\n"
                # f"MKL_INT bkloc = numroc( &b_rows, &b_bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);\n"
                f"MKL_INT info;\n"
                f"MKL_INT _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];\n"
                # f"MKL_INT a_lld = mloc;\n"
                f"MKL_INT a_lld = max(akloc, a_bcols);\n"
                # f"printf(\"a_rows=%d, a_cols=%d, a_brows=%d, a_bcols=%d, a_lld=%d\\n\", grows, a_cols, brows, a_bcols, a_lld);\n"
                # f"descinit(_a_ldesc, &grows, &a_cols, &brows, &a_bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);\n"
                f"descinit(_a_ldesc, &a_cols, &grows, &a_bcols, &brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);\n"
                # f"MKL_INT b_lld = bkloc;\n"
                f"MKL_INT b_lld = max(nloc, bcols);\n"
                # f"descinit(_b_ldesc, &b_rows, &gcols, &b_brows, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);\n"
                f"descinit(_b_ldesc, &gcols, &b_rows, &bcols, &b_brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);\n"
                # f"MKL_INT c_lld = mloc;\n"
                # f"MKL_INT c_lld = _a_ldesc[4];\n"
                # f"MKL_INT c_lld = nloc;\n"
                f"MKL_INT c_lld = max(nloc, bcols);\n"
                # f"descinit(_c_ldesc, &grows, &gcols, &_a_ldesc[4], &_b_ldesc[5], &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);\n"
                f"descinit(_c_ldesc, &gcols, &grows, &bcols, &brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);\n"

                # f"MKL_INT grows = _desca[2];\n"
                # f"MKL_INT gcols = _descb[3];\n"
                # f"MKL_INT gld = _desca[2];\n"
                # f"MKL_INT lld = _desca[4];\n"
                # # f"MKL_INT gld = _descb[2];\n"
                # # f"MKL_INT lld = _descb[4];\n"
                # f"MKL_INT info;\n"
                # f"descinit(_gdescc, &grows, &gcols, &grows, &gcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &gld, &info);\n"
                # f"descinit(_ldescc, &grows, &gcols, &_desca[4], &_descb[5], &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &lld, &info);\n"
                # f"MKL_INT _m = grows, _n = gcols, _k = _desca[3];\n"
                # f"MKL_INT _m = grows, _n = gcols, _k = _desca[2];\n"
                f"MKL_INT _m = grows, _n = gcols, _k = a_cols;\n"
                f"p{lapack_dtype_str}gemm(\n"
                # f"    &trans, &trans, &_m, &_n, &_k, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _desca,\n"
                # f"    _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _descb, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _ldescc);")
                # f"    &trans, &trans, &_m, &_n, &_k, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc,\n"
                # f"    _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc);")
                f"    &trans, &trans, &_n, &_m, &_k, &one, _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc,\n"
                f"    _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc);")
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

    def __init__(self, name, m=None, n=None, k=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                        #  inputs={"_a", "_desca", "_b", "_descb"},
                        #  outputs={"_c", "_gdescc", "_ldescc"},
                         inputs={"_a", "_b", "_a_block_sizes", "_b_block_sizes"},
                         outputs={"_c"},
                         **kwargs)
        self._m = m
        self._n = n
        self._k = k

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



  