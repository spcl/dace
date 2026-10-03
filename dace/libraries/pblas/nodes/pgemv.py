# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import blas_helpers
from dace.libraries.mpi.nodes.node import expanded_input_connectors
from dace.libraries.pblas.nodes.node import scalapack_grid_code
from dace.ordered import OrderedSet


@dace.library.expansion
class ExpandPgemvMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        # NOTE: MKL ScaLAPACK is using column-major order
        transa = 'N' if node.transa == 'T' else 'T'
        code = f"""
        {scalapack_grid_code(node, parent_state, True)}
            const {dtype.ctype} zero = 0.0E+0, one = 1.0E+0;
            const char trans = '{transa}';
            MKL_INT ga_rows = (trans == 'N' ? {node.n} : {node.n});
            MKL_INT ga_cols = (trans == 'N' ? {node.m} : {node.m});
            MKL_INT gy_rows = (trans == 'N' ? {node.n} : {node.m});
            MKL_INT gy_cols = 1;
            MKL_INT gx_rows = (trans == 'N' ? {node.m} : {node.n});
            MKL_INT gx_cols = 1;
            MKL_INT la_rows = (trans == 'N' ? _a_block_sizes[1] : _a_block_sizes[1]);
            MKL_INT la_cols = (trans == 'N' ? _a_block_sizes[0] : _a_block_sizes[0]);
            MKL_INT ly_rows = gy_rows;
            MKL_INT ly_cols = 1;
            MKL_INT lx_rows = gx_rows;
            MKL_INT lx_cols = 1;
            MKL_INT n_ly_rows = numroc_( &gy_rows, &ly_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            // MKL_INT n_ly_cols = numroc_( &gy_cols, &ly_cols, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT y_lld = max(n_ly_rows, 1);
            MKL_INT n_la_rows = numroc_( &ga_rows, &la_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            // MKL_INT n_la_cols = numroc_( &ga_cols, &la_cols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT a_lld = max(la_rows, 1);
            MKL_INT n_lx_rows = numroc_( &gx_rows, &lx_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            // MKL_INT n_lx_cols = numroc_( &gx_cols, &lx_cols, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT x_lld = max(n_lx_rows, 1);
            MKL_INT info_y, info_a, info_x;
            MKL_INT _y_ldesc[9], _a_ldesc[9],  _x_ldesc[9];
            descinit_(_y_ldesc, &gy_rows, &gy_cols, &ly_rows, &ly_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &y_lld, &info_y);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &a_lld, &info_a);
            descinit_(_x_ldesc, &gx_rows, &gx_cols, &lx_rows, &lx_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &x_lld, &info_x);
            if (info_y != 0 || info_a != 0 || info_x != 0) {{
                fprintf(stderr, "descinit_ refused a PGEMV descriptor: info(y=%d a=%d x=%d). A negative value is "
                                "minus the index of the argument it refused; the call would otherwise "
                                "return wrong numbers rather than fail.\\n", info_y, info_a, info_x);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }}
            MKL_INT _m = ga_rows, _n = ga_cols;
            p{lapack_dtype_str}gemv_(
                &trans, &_m, &_n, &one, _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc,
                _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _x_ldesc, &__state->__mkl_int_one,
                &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _y_ldesc, &__state->__mkl_int_one);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
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

        # NOTE: MKL ScaLAPACK is using column-major order
        transa = 'N' if node.transa == 'T' else 'T'
        code = f"""
        {scalapack_grid_code(node, parent_state, False)}
            {dtype.ctype} zero = 0.0E+0, one = 1.0E+0;
            char trans = '{transa}';
            int ga_rows = (trans == 'N' ? {node.n} : {node.n});
            int ga_cols = (trans == 'N' ? {node.m} : {node.m});
            int gy_rows = (trans == 'N' ? {node.n} : {node.m});
            int gy_cols = 1;
            int gx_rows = (trans == 'N' ? {node.m} : {node.n});
            int gx_cols = 1;
            int la_rows = (trans == 'N' ? _a_block_sizes[1] : _a_block_sizes[1]);
            int la_cols = (trans == 'N' ? _a_block_sizes[0] : _a_block_sizes[0]);
            int ly_rows = gy_rows;
            int ly_cols = 1;
            int lx_rows = gx_rows;
            int lx_cols = 1;
            int n_ly_rows = numroc_( &gy_rows, &ly_rows, &__myprow, &__state->__int_zero, &__nprow);
            // int n_ly_cols = numroc_( &gy_cols, &ly_cols, &__myprow, &__state->__int_zero, &__nprow);
            int y_lld = max(n_ly_rows, 1);
            int n_la_rows = numroc_( &ga_rows, &la_rows, &__myprow, &__state->__int_zero, &__nprow);
            // int n_la_cols = numroc_( &ga_cols, &la_cols, &__mypcol, &__state->__int_zero, &__npcol);
            int a_lld = max(la_rows, 1);
            int n_lx_rows = numroc_( &gx_rows, &lx_rows, &__myprow, &__state->__int_zero, &__nprow);
            // int n_lx_cols = numroc_( &gx_cols, &lx_cols, &__myprow, &__state->__int_zero, &__nprow);
            int x_lld = max(n_lx_rows, 1);
            int info_y, info_a, info_x;
            int _y_ldesc[9], _a_ldesc[9],  _x_ldesc[9];
            descinit_(_y_ldesc, &gy_rows, &gy_cols, &ly_rows, &ly_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &y_lld, &info_y);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &a_lld, &info_a);
            descinit_(_x_ldesc, &gx_rows, &gx_cols, &lx_rows, &lx_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &x_lld, &info_x);
            if (info_y != 0 || info_a != 0 || info_x != 0) {{
                fprintf(stderr, "descinit_ refused a PGEMV descriptor: info(y=%d a=%d x=%d). A negative value is "
                                "minus the index of the argument it refused; the call would otherwise "
                                "return wrong numbers rather than fail.\\n", info_y, info_a, info_x);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }}
            int _m = ga_rows, _n = ga_cols;
            p{lapack_dtype_str}gemv_(
                &trans, &_m, &_n, &one, _a, &__state->__int_one, &__state->__int_one, _a_ldesc,
                _b, &__state->__int_one, &__state->__int_one, _x_ldesc, &__state->__int_one,
                &zero, _c, &__state->__int_one, &__state->__int_one, _y_ldesc, &__state->__int_one);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
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
        super().__init__(name,
                         *args,
                         inputs=OrderedSet(('_a', '_b', '_a_block_sizes', '_b_block_sizes')),
                         outputs={"_c"},
                         **kwargs)
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
