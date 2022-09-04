# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandPgemvMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        # NOTE: MKL ScaLAPACK is using column-major order
        transa = 'T' if node.transa == 'N' else 'N'
        code = f"""
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = '{transa}';
            MKL_INT ga_rows = (trans == 'N' ? {node.n} : {node.n});
            MKL_INT ga_cols = (trans == 'N' ? {node.m} : {node.m});
            MKL_INT gy_rows = (trans == 'N' ? {node.n} : {node.m});
            MKL_INT gy_cols = 1;
            MKL_INT gx_rows = (trans == 'N' ? {node.m} : {node.n});
            MKL_INT gx_cols = 1;
            MKL_INT la_rows = (trans == 'N' ? _a_block_sizes[1] : _a_block_sizes[1]);
            MKL_INT la_cols = (trans == 'N' ? _a_block_sizes[0] : _a_block_sizes[0]);
            MKL_INT ly_rows = (trans == 'N' ? _a_block_sizes[1] : _a_block_sizes[0]);
            MKL_INT ly_cols = 1;
            MKL_INT lx_rows = (trans == 'N' ? _a_block_sizes[0] : _a_block_sizes[1]);
            MKL_INT lx_cols = 1;
            printf(\"(%d, %d)x(%d, %d)->(%d, %d)\\n\", la_rows, la_cols, lx_rows, lx_cols, ly_rows, ly_cols);
            MKL_INT n_ly_rows = numroc( &gy_rows, &ly_rows, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT n_ly_cols = numroc( &gy_cols, &ly_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            //MKL_INT n_ly_rows = numroc( &gy_rows, &ly_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            //MKL_INT n_ly_cols = numroc( &gy_cols, &ly_cols, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            printf(\"y: (%d, %d)\\n\", n_ly_rows, n_ly_cols);
            //assert(ly_rows == n_ly_rows);
            //assert(ly_cols == n_ly_cols);
            MKL_INT y_lld = max(n_ly_rows, 1);
            //MKL_INT n_la_rows = numroc( &ga_rows, &la_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            //MKL_INT n_la_cols = numroc( &ga_cols, &la_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT n_la_rows = numroc( &ga_rows, &la_rows, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT n_la_cols = numroc( &ga_cols, &la_cols, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            //assert(la_rows == n_la_rows);
            //assert(la_cols == n_la_cols);
            MKL_INT a_lld = max(la_rows, 1);
            MKL_INT n_lx_rows = numroc( &gx_rows, &lx_rows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT n_lx_cols = numroc( &gx_cols, &lx_cols, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            //MKL_INT n_lx_rows = numroc( &gx_rows, &lx_rows, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            //MKL_INT n_lx_cols = numroc( &gx_cols, &lx_cols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            printf(\"x: (%d, %d)\\n\", n_lx_rows, n_lx_cols);
            //assert(lx_rows == n_lx_rows);
            // assert(lx_cols == n_lx_cols);
            MKL_INT x_lld = max(n_lx_rows, 1);
            printf(\"Px: %d, Py: %d\\n\", __state->__mkl_scalapack_prows, __state->__mkl_scalapack_pcols);
            printf(\"a: %d, x: %d, y:%d\\n\", a_lld, x_lld, y_lld);
            MKL_INT info;
            MKL_INT _a_ldesc[9],  _x_ldesc[9], _y_ldesc[9];
            descinit(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);
            // MKL_INT b_lld = b_mloc;
            descinit(_x_ldesc, &gx_rows, &gx_cols, &lx_rows, &lx_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &x_lld, &info);
            // MKL_INT c_lld = mloc;
            descinit(_y_ldesc, &gy_rows, &gy_cols, &ly_rows, &ly_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &y_lld, &info);
            MKL_INT _m = ga_rows, _n = ga_cols;
            // MKL_INT _m = (trans == 'N' ? ga_cols : ga_rows);
            // MKL_INT _n = (trans == 'N' ? ga_rows : ga_cols);
            const char transa = 'T';
            p{lapack_dtype_str}gemv(
                &trans, &_m, &_n, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc,
                _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _x_ldesc, &__state->__mkl_int_one,
                &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _y_ldesc, &__state->__mkl_int_one);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandPgemvMKLOpenMPI(ExpandTransformation):
    environments = [environments.intel_mkl_openmpi.IntelMKLScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return ExpandPgemvMKLMPICH.expansion(node, parent_state, parent_sdfg, **kwargs)


@dace.library.expansion
class ExpandPgemvReferenceMPICH(ExpandTransformation):
    environments = [environments.ref_mpich.ScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        transa = 'N' if node._transa == 'T' else 'T'
        code = f"""
            double zero = 0.0E+0, one = 1.0E+0;
            char trans = '{transa}';
            int grows = (trans == 'T' ? {node.m} : {node.n});
            int gcols = 1;
            int a_rows = {node.n};
            int a_cols = {node.m};
            int b_rows = (trans == 'T' ? {node.n} : {node.m});
            int b_cols = 1;
            int brows = grows / __state->__scalapack_size;
            int bcols = 1;
            int a_brows = _a_block_sizes[1];
            int a_bcols = _a_block_sizes[0];
            int b_brows = _b_block_sizes[0];
            int b_bcols = 1;
            int mloc = numroc_( &grows, &brows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            int a_mloc = numroc_( &a_rows, &a_brows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            int a_nloc = numroc_( &a_cols, &a_bcols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int b_mloc = numroc_( &b_rows, &b_brows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            int info;
            int _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];
            int a_lld = a_mloc;
            descinit_(_a_ldesc, &a_rows, &a_cols, &a_brows, &a_bcols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &a_lld, &info);
            int b_lld = b_mloc;
            descinit_(_b_ldesc, &b_rows, &b_cols, &b_mloc, &b_bcols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &b_lld, &info);
            int c_lld = mloc;
            descinit_(_c_ldesc, &grows, &gcols, &mloc, &bcols, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &c_lld, &info);
            int _m = a_rows, _n = a_cols;
            p{lapack_dtype_str}gemv_(
                &trans, &_m, &_n, &one, _a, &__state->__int_one, &__state->__int_one, _a_ldesc,
                _b, &__state->__int_one, &__state->__int_one, _b_ldesc, &__state->__int_one,
                &zero, _c, &__state->__int_one, &__state->__int_one, _c_ldesc, &__state->__int_one);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandPgemvReferenceOpenMPI(ExpandTransformation):
    environments = [environments.ref_openmpi.ScaLAPACKOpenMPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return ExpandPgemvReferenceMPICH.expansion(node, parent_state, parent_sdfg, **kwargs)


@dace.library.node
class Pgemv(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ x) + beta * y.
    """

    # Global properties
    implementations = {
        "MKLMPICH": ExpandPgemvMKLMPICH,
        "MKLOpenMPI": ExpandPgemvMKLOpenMPI,
        "ReferenceMPICH": ExpandPgemvReferenceMPICH,
        "ReferenceOpenMPI": ExpandPgemvReferenceOpenMPI
    }
    default_implementation = None

    transa = dace.properties.Property(dtype=str, default='N')
    m = dace.properties.SymbolicProperty(allow_none=True, default=None)
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, transa='N', m=None, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_a", "_b", "_a_block_sizes", "_b_block_sizes"}, outputs={"_c"}, **kwargs)
        self.transa = transa
        self.m = m
        self.n = n

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
