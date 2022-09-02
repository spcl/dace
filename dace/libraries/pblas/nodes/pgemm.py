# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandPgemmMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        code = f"""
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = 'N';
            MKL_INT grows = {node.m};
            MKL_INT gcols = {node.n};
            MKL_INT a_cols = {node.k};
            MKL_INT b_rows = {node.k};
            MKL_INT brows = _a_block_sizes[0];
            MKL_INT bcols = (gcols > 1 ? _b_block_sizes[1]: 1);
            MKL_INT a_brows = _a_block_sizes[0];
            MKL_INT a_bcols = (a_cols > 1 ? _a_block_sizes[1]: 1);
            MKL_INT b_brows = _b_block_sizes[0];
            MKL_INT b_bcols = _b_block_sizes[1];
            MKL_INT mloc = numroc( &grows, &brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT nloc = numroc( &gcols, &bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT akloc = numroc( &a_cols, &a_bcols, &__state->__mkl_scalapack_mypcol, &__state->__mkl_int_zero, &__state->__mkl_scalapack_pcols);
            MKL_INT bkloc = numroc( &b_rows, &b_brows, &__state->__mkl_scalapack_myprow, &__state->__mkl_int_zero, &__state->__mkl_scalapack_prows);
            MKL_INT info;
            MKL_INT _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];
            MKL_INT a_lld = max(akloc, a_bcols);
            descinit(_a_ldesc, &a_cols, &grows, &a_bcols, &brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &a_lld, &info);
            MKL_INT b_lld = max(nloc, bcols);
            descinit(_b_ldesc, &gcols, &b_rows, &bcols, &b_brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &b_lld, &info);
            MKL_INT c_lld = max(nloc, bcols);
            descinit(_c_ldesc, &gcols, &grows, &bcols, &brows, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context, &c_lld, &info);
            MKL_INT _m = grows, _n = gcols, _k = a_cols;
            p{lapack_dtype_str}gemm(
                &trans, &trans, &_n, &_m, &_k, &one, _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc,
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
            double  zero = 0.0E+0, one = 1.0E+0;
            char trans = 'N';
            int grows = {node.m};
            int gcols = {node.n};
            int a_cols = {node.k};
            int b_rows = {node.k};
            int brows = _a_block_sizes[0];
            int bcols = (gcols > 1 ? _b_block_sizes[1]: 1);
            int a_brows = _a_block_sizes[0];
            int a_bcols = (a_cols > 1 ? _a_block_sizes[1]: 1);
            int b_brows = _b_block_sizes[0];
            int b_bcols = _b_block_sizes[1];
            int mloc = numroc_( &grows, &brows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            int nloc = numroc_( &gcols, &bcols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int akloc = numroc_( &a_cols, &a_bcols, &__state->__scalapack_mypcol, &__state->__int_zero, &__state->__scalapack_pcols);
            int bkloc = numroc_( &b_rows, &b_brows, &__state->__scalapack_myprow, &__state->__int_zero, &__state->__scalapack_prows);
            int info;
            int _a_ldesc[9],  _b_ldesc[9], _c_ldesc[9];
            int a_lld = max(akloc, a_bcols);
            descinit_(_a_ldesc, &a_cols, &grows, &a_bcols, &brows, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &a_lld, &info);
            int b_lld = max(nloc, bcols);
            descinit_(_b_ldesc, &gcols, &b_rows, &bcols, &b_brows, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &b_lld, &info);
            int c_lld = max(nloc, bcols);
            descinit_(_c_ldesc, &gcols, &grows, &bcols, &brows, &__state->__int_zero, &__state->__int_zero, &__state->__scalapack_context, &c_lld, &info);
            int _m = grows, _n = gcols, _k = a_cols;
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_n, &_m, &_k, &one, _b, &__state->__int_one, &__state->__int_one, _b_ldesc,
                _a, &__state->__int_one, &__state->__int_one, _a_ldesc, &zero, _c, &__state->__int_one, &__state->__int_one, _c_ldesc);
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
        super().__init__(name, *args, inputs={"_a", "_b", "_a_block_sizes", "_b_block_sizes"}, outputs={"_c"}, **kwargs)
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
