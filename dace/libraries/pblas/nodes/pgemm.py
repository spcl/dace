# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import blas_helpers
from dace.ordered import OrderedSet


@dace.library.expansion
class ExpandPgemmMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        code = f"""
            const {dtype.ctype} zero = 0.0E+0, one = 1.0E+0;
            const char trans = 'N';
            MKL_INT gc_rows = {node.n};
            MKL_INT gc_cols = {node.m};
            MKL_INT ga_rows = {node.k};
            MKL_INT ga_cols = {node.m};
            MKL_INT gb_rows = {node.n};
            MKL_INT gb_cols = {node.k};
            MKL_INT lc_rows = _b_block_sizes[1];
            MKL_INT lc_cols = _a_block_sizes[0];
            MKL_INT la_rows = _a_block_sizes[1];
            MKL_INT la_cols = _a_block_sizes[0];
            MKL_INT lb_rows = _b_block_sizes[1];
            MKL_INT lb_cols = _b_block_sizes[0];
            MKL_INT n_lc_rows = numroc_( &gc_rows, &lc_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            // MKL_INT n_lc_cols = numroc_( &gc_cols, &lc_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT c_lld = max(n_lc_rows, 1);
            MKL_INT n_la_rows = numroc_( &ga_rows, &la_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            // MKL_INT n_la_cols = numroc_( &ga_cols, &la_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT a_lld = max(n_la_rows, 1);
            MKL_INT n_lb_rows = numroc_( &gb_rows, &lb_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            // MKL_INT n_lb_cols = numroc_( &gb_cols, &lb_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT b_lld = max(n_lb_rows, 1);
            MKL_INT info;
            MKL_INT _c_ldesc[9], _a_ldesc[9],  _b_ldesc[9];
            descinit_(_c_ldesc, &gc_rows, &gc_cols, &lc_rows, &lc_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);
            descinit_(_b_ldesc, &gb_rows, &gb_cols, &lb_rows, &lb_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);
            MKL_INT _m = gc_rows, _n = gc_cols, _k = ga_rows;
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc,
                _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandPgemmMKLOpenMPI(ExpandTransformation):
    environments = [environments.intel_mkl_openmpi.IntelMKLScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return ExpandPgemmMKLMPICH.expansion(node, parent_state, parent_sdfg, **kwargs)


@dace.library.expansion
class ExpandPgemmReferenceMPICH(ExpandTransformation):
    environments = [environments.ref_mpich.ScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        code = f"""
            {dtype.ctype} zero = 0.0E+0, one = 1.0E+0;
            char trans = 'N';
            int gc_rows = {node.n};
            int gc_cols = {node.m};
            int ga_rows = {node.k};
            int ga_cols = {node.m};
            int gb_rows = {node.n};
            int gb_cols = {node.k};
            int lc_rows = _b_block_sizes[1];
            int lc_cols = _a_block_sizes[0];
            int la_rows = _a_block_sizes[1];
            int la_cols = _a_block_sizes[0];
            int lb_rows = _b_block_sizes[1];
            int lb_cols = _b_block_sizes[0];
            int n_lc_rows = numroc_( &gc_rows, &lc_rows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            // int n_lc_cols = numroc_( &gc_cols, &lc_cols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int c_lld = max(n_lc_rows, 1);
            int n_la_rows = numroc_( &ga_rows, &la_rows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            // int n_la_cols = numroc_( &ga_cols, &la_cols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int a_lld = max(n_la_rows, 1);
            int n_lb_rows = numroc_( &gb_rows, &lb_rows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            // int n_lb_cols = numroc_( &gb_cols, &lb_cols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int b_lld = max(n_lb_rows, 1);
            int info_c, info_a, info_b;
            int _c_ldesc[9], _a_ldesc[9],  _b_ldesc[9];
            descinit_(_c_ldesc, &gc_rows, &gc_cols, &lc_rows, &lc_cols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &c_lld, &info_c);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &a_lld, &info_a);
            descinit_(_b_ldesc, &gb_rows, &gb_cols, &lb_rows, &lb_cols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &b_lld, &info_b);
            int _m = gc_rows, _n = gc_cols, _k = ga_rows;
            {{
                double dbg_sa = 0.0, dbg_sb = 0.0;
                for (int dbg_i = 0; dbg_i < la_rows * la_cols; ++dbg_i) dbg_sa += _a[dbg_i];
                for (int dbg_i = 0; dbg_i < lb_rows * lb_cols; ++dbg_i) dbg_sb += _b[dbg_i];
                std::fprintf(stderr,
                    "[PGEMM-DBG] rank=%d size=%d ctx=%d prows=%d pcols=%d myprow=%d mypcol=%d "
                    "g(c=%dx%d a=%dx%d b=%dx%d) blk(c=%dx%d a=%dx%d b=%dx%d) "
                    "numroc(c=%d a=%d b=%d) lld(c=%d a=%d b=%d) info(c=%d a=%d b=%d) "
                    "mnk=(%d,%d,%d) bs_a=(%d,%d) bs_b=(%d,%d) sum_a=%.17g sum_b=%.17g\\n",
                    __state->__scalapack_rank, __state->__scalapack_size, __state->__scalapack_context,
                    __state->__scalapack_prows, __state->__scalapack_pcols,
                    __state->__scalapack_myprow, __state->__scalapack_mypcol,
                    (int)gc_rows, (int)gc_cols, (int)ga_rows, (int)ga_cols, (int)gb_rows, (int)gb_cols,
                    (int)lc_rows, (int)lc_cols, (int)la_rows, (int)la_cols, (int)lb_rows, (int)lb_cols,
                    (int)n_lc_rows, (int)n_la_rows, (int)n_lb_rows, (int)c_lld, (int)a_lld, (int)b_lld,
                    (int)info_c, (int)info_a, (int)info_b, (int)_m, (int)_n, (int)_k,
                    (int)_a_block_sizes[0], (int)_a_block_sizes[1],
                    (int)_b_block_sizes[0], (int)_b_block_sizes[1], dbg_sa, dbg_sb);
                std::fprintf(stderr,
                    "[PGEMM-DBG] rank=%d desc_c=[%d %d %d %d %d %d %d %d %d] "
                    "desc_a=[%d %d %d %d %d %d %d %d %d] desc_b=[%d %d %d %d %d %d %d %d %d]\\n",
                    __state->__scalapack_rank,
                    _c_ldesc[0], _c_ldesc[1], _c_ldesc[2], _c_ldesc[3], _c_ldesc[4], _c_ldesc[5], _c_ldesc[6], _c_ldesc[7], _c_ldesc[8],
                    _a_ldesc[0], _a_ldesc[1], _a_ldesc[2], _a_ldesc[3], _a_ldesc[4], _a_ldesc[5], _a_ldesc[6], _a_ldesc[7], _a_ldesc[8],
                    _b_ldesc[0], _b_ldesc[1], _b_ldesc[2], _b_ldesc[3], _b_ldesc[4], _b_ldesc[5], _b_ldesc[6], _b_ldesc[7], _b_ldesc[8]);
                {{
                    Dl_info dbg_dl;
                    if (dladdr((void *)p{lapack_dtype_str}gemm_, &dbg_dl) && dbg_dl.dli_fname)
                        std::fprintf(stderr, "[PGEMM-DBG] rank=%d p{lapack_dtype_str}gemm_ from %s\\n",
                                     __state->__scalapack_rank, dbg_dl.dli_fname);
                }}
                std::fflush(stderr);
            }}
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__int_one, &__state->__int_one, _b_ldesc,
                _a, &__state->__int_one, &__state->__int_one, _a_ldesc, &zero, _c, &__state->__int_one, &__state->__int_one, _c_ldesc);
            {{
                double dbg_sc = 0.0;
                for (int dbg_i = 0; dbg_i < lc_rows * lc_cols; ++dbg_i) dbg_sc += _c[dbg_i];
                std::fprintf(stderr, "[PGEMM-DBG] rank=%d sum_c=%.17g first_c=%.17g\\n",
                             __state->__scalapack_rank, dbg_sc, _c[0]);
                std::fflush(stderr);
            }}
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandPgemmReferenceOpenMPI(ExpandTransformation):
    environments = [environments.ref_openmpi.ScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return ExpandPgemmReferenceMPICH.expansion(node, parent_state, parent_sdfg, **kwargs)


@dace.library.node
class Pgemm(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C.
    """

    # Global properties
    implementations = {
        "MKLMPICH": ExpandPgemmMKLMPICH,
        "MKLOpenMPI": ExpandPgemmMKLOpenMPI,
        "ReferenceMPICH": ExpandPgemmReferenceMPICH,
        "ReferenceOpenMPI": ExpandPgemmReferenceOpenMPI
    }
    default_implementation = None

    m = dace.properties.SymbolicProperty(allow_none=True, default=None)
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)
    k = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, m=None, n=None, k=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs=OrderedSet(('_a', '_b', '_a_block_sizes', '_b_block_sizes')),
                         outputs={"_c"},
                         **kwargs)
        self.m = m
        self.n = n
        self.k = k

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

        for e in state.out_edges(self):
            if e.src_conn == "_gdescc":
                gdescc = sdfg.arrays[e.data.data]
            if e.src_conn == "_ldescc":
                ldescc = sdfg.arrays[e.data.data]
            if e.src_conn == "_c":
                c = sdfg.arrays[e.data.data]

        if a.dtype.base_type != b.dtype.base_type:
            raise ValueError("The types of A and B do not match!")
        if c.dtype.base_type != b.dtype.base_type:
            raise ValueError("The types of B and C do not match!")

        return a, b, c, desca, descb, gdescc, ldesc
