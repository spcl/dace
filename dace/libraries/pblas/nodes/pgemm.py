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
                    static const char *dbg_syms[] = {{"p{lapack_dtype_str}gemm_", "dgemm_", "Cblacs_gridinit",
                                                     "Cblacs_pinfo", "numroc_", "descinit_", "MPI_Init",
                                                     "blacs_gridinit_", "pdgemm", "openblas_get_config"}};
                    for (unsigned dbg_j = 0; dbg_j < sizeof(dbg_syms) / sizeof(dbg_syms[0]); ++dbg_j) {{
                        void *dbg_p = dlsym(RTLD_DEFAULT, dbg_syms[dbg_j]);
                        Dl_info dbg_dl;
                        const char *dbg_from = "(unresolved)";
                        if (dbg_p && dladdr(dbg_p, &dbg_dl) && dbg_dl.dli_fname) dbg_from = dbg_dl.dli_fname;
                        std::fprintf(stderr, "[PGEMM-DBG] rank=%d sym %s -> %s\\n",
                                     __state->__scalapack_rank, dbg_syms[dbg_j], dbg_from);
                    }}
                    if (__state->__scalapack_rank == 0) {{
                        std::FILE *dbg_maps = std::fopen("/proc/self/maps", "r");
                        if (dbg_maps) {{
                            char dbg_line[512], dbg_prev[512];
                            dbg_prev[0] = 0;
                            while (std::fgets(dbg_line, sizeof(dbg_line), dbg_maps)) {{
                                char *dbg_path = std::strrchr(dbg_line, ' ');
                                if (!dbg_path) continue;
                                ++dbg_path;
                                char *dbg_nl = std::strchr(dbg_path, '\\n');
                                if (dbg_nl) *dbg_nl = 0;
                                if (dbg_path[0] != '/' || !std::strstr(dbg_path, ".so")) continue;
                                if (std::strcmp(dbg_path, dbg_prev) == 0) continue;
                                std::snprintf(dbg_prev, sizeof(dbg_prev), "%s", dbg_path);
                                std::fprintf(stderr, "[PGEMM-DBG] lib %s\\n", dbg_path);
                            }}
                            std::fclose(dbg_maps);
                        }}
                    }}
                }}
                std::fprintf(stderr,
                    "[PGEMM-DBG] rank=%d a[0..3]=%a,%a,%a,%a b[0..3]=%a,%a,%a,%a ptr(a=%p b=%p c=%p)\\n",
                    __state->__scalapack_rank, _a[0], _a[1], _a[2], _a[3], _b[0], _b[1], _b[2], _b[3],
                    (const void *)_a, (const void *)_b, (const void *)_c);
                int dbg_mrank = -1, dbg_msize = -1, dbg_minit = 0, dbg_mthread = -1;
                MPI_Initialized(&dbg_minit);
                MPI_Query_thread(&dbg_mthread);
                MPI_Comm_rank(MPI_COMM_WORLD, &dbg_mrank);
                MPI_Comm_size(MPI_COMM_WORLD, &dbg_msize);
                double dbg_glob = -1.0;
                MPI_Allreduce(&dbg_sa, &dbg_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                const char *dbg_omp = std::getenv("OMP_NUM_THREADS");
                const char *dbg_obl = std::getenv("OPENBLAS_NUM_THREADS");
                std::fprintf(stderr,
                    "[PGEMM-DBG] rank=%d mpi(rank=%d size=%d init=%d thread=%d) allreduce_sum_a=%.17g "
                    "OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s\\n",
                    __state->__scalapack_rank, dbg_mrank, dbg_msize, dbg_minit, dbg_mthread, dbg_glob,
                    dbg_omp ? dbg_omp : "(unset)", dbg_obl ? dbg_obl : "(unset)");
                std::fflush(stderr);
            }}
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__int_one, &__state->__int_one, _b_ldesc,
                _a, &__state->__int_one, &__state->__int_one, _a_ldesc, &zero, _c, &__state->__int_one, &__state->__int_one, _c_ldesc);
            {{
                double dbg_sc = 0.0;
                for (int dbg_i = 0; dbg_i < lc_rows * lc_cols; ++dbg_i) dbg_sc += _c[dbg_i];
                std::fprintf(stderr, "[PGEMM-DBG] rank=%d sum_c=%.17g c[0..5]=%a,%a,%a,%a,%a,%a\\n",
                             __state->__scalapack_rank, dbg_sc, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5]);
                std::fflush(stderr);
                {dtype.ctype} *dbg_c2 = new {dtype.ctype}[(size_t)lc_rows * (size_t)lc_cols];
                p{lapack_dtype_str}gemm_(
                    &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__int_one, &__state->__int_one, _b_ldesc,
                    _a, &__state->__int_one, &__state->__int_one, _a_ldesc, &zero, dbg_c2, &__state->__int_one, &__state->__int_one, _c_ldesc);
                double dbg_sc2 = 0.0, dbg_maxdiff = 0.0;
                for (int dbg_i = 0; dbg_i < lc_rows * lc_cols; ++dbg_i) {{
                    dbg_sc2 += dbg_c2[dbg_i];
                    double dbg_d = dbg_c2[dbg_i] - _c[dbg_i];
                    if (dbg_d < 0) dbg_d = -dbg_d;
                    if (dbg_d > dbg_maxdiff) dbg_maxdiff = dbg_d;
                }}
                std::fprintf(stderr, "[PGEMM-DBG] rank=%d repeat sum_c2=%.17g maxdiff=%.17g\\n",
                             __state->__scalapack_rank, dbg_sc2, dbg_maxdiff);
                std::fflush(stderr);
                delete[] dbg_c2;
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
