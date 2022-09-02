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

        transa = 'N' if node._transa == 'T' else 'T'
        code = f"""
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = '{transa}';
            MKL_INT grows = (trans == 'T' ? {node.m} : {node.n});
            MKL_INT gcols = 1;
            MKL_INT a_rows = {node.n};
            MKL_INT a_cols = {node.m};
            MKL_INT b_rows = (trans == 'T' ? {node.n} : {node.m});
            MKL_INT b_cols = 1;
            MKL_INT brows = grows / __state->__mkl_scalapack_size;
            MKL_INT bcols = 1;
            MKL_INT a_brows = _a_block_sizes[1];
            MKL_INT a_bcols = _a_block_sizes[0];
            MKL_INT b_brows = _b_block_sizes[0];
            MKL_INT b_bcols = 1;
            MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT a_mloc = numroc( &a_rows, &a_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT a_nloc = numroc( &a_cols, &a_bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT b_mloc = numroc( &b_rows, &b_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT info;
            MKL_INT _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];
            MKL_INT a_lld = a_mloc;
            descinit(_a_ldesc, &a_rows, &a_cols, &a_brows, &a_bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);
            MKL_INT b_lld = b_mloc;
            descinit(_b_ldesc, &b_rows, &b_cols, &b_mloc, &b_bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);
            MKL_INT c_lld = mloc;
            descinit(_c_ldesc, &grows, &gcols, &mloc, &bcols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);
            MKL_INT _m = a_rows, _n = a_cols;
            p{lapack_dtype_str}gemv(
                &trans, &_m, &_n, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc,
                _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc, &__state->__mkl_int_one,
                &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc, &__state->__mkl_int_one);
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
