# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandPgemvMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKLScaLAPACK]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(
            parent_sdfg, parent_state)
        from dace.libraries.lapack import utils
        lapack_dtype_str = utils.LAPACK_DTYPE_CHR(a.dtype.base_type)

        transa = 'N' if node._transa == 'T' else 'T'
        code = f"""
            const double  zero = 0.0E+0, one = 1.0E+0;
            const char trans = '{transa}';
            MKL_INT grows = (trans == 'T' ? {node._m} : {node._n});
            MKL_INT gcols = 1;
            MKL_INT a_rows = {node._n};
            MKL_INT a_cols = {node._m};
            MKL_INT b_rows = (trans == 'T' ? {node._n} : {node._m});
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
                &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc, &__state->__mkl_int_one);"
        """
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
        super().__init__(
            name,
            *args,
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
