# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import blas_helpers
from dace.libraries.mpi.nodes.node import expanded_input_connectors
from dace.libraries.pblas.nodes.node import scalapack_grid_code
from dace.ordered import OrderedSet
from dace.symbolic import symstr


def local_extent_guard(a, b, c) -> str:
    """C++ checking that the block sizes handed to ``descinit_`` really describe the local arrays.

    ``descinit_`` only validates a descriptor against itself: it never sees how big the buffers
    actually are. When the block sizes disagree with them every argument is still individually
    legal, so the call runs and PBLAS touches the block IT believes this process owns. A block
    smaller than the local array leaves the rest of ``_c`` exactly as allocated, and the caller
    gets that memory back as its answer -- plausible-looking numbers that change from run to run
    with whatever the allocator handed out, which is worse than a crash.

    The operands are stored row-major and read by PBLAS column-major, so the block this process
    owns is ``shape[1]`` rows by ``shape[0]`` columns.

    :param a: Local descriptor of the A operand.
    :param b: Local descriptor of the B operand.
    :param c: Local descriptor of the C operand.
    :returns: C++ that aborts, naming each operand and both extents, when they disagree.
    """
    operands = (('c', c), ('a', a), ('b', b))
    checks = ' || '.join(f'n_l{n}_rows != ({symstr(d.shape[1])}) || n_l{n}_cols != ({symstr(d.shape[0])})'
                         for n, d in operands)
    # Everything is widened to ``long long`` for the report: MKL_INT is 64-bit under ILP64, so a
    # ``%d`` against it would print the wrong half of the number in the one build that most needs
    # the diagnostic.
    reported = ', '.join(f'(long long)n_l{n}_rows, (long long)({symstr(d.shape[1])}), '
                         f'(long long)n_l{n}_cols, (long long)({symstr(d.shape[0])})' for n, d in operands)
    fields = ' '.join(f'{n}=%lldx%lld(want %lldx%lld)' for n, _ in operands)
    return f"""            if ({checks}) {{
                fprintf(stderr, "PGEMM block sizes do not describe the local arrays: {fields}. "
                                "The descriptors are individually legal, so the call would "
                                "return wrong numbers rather than fail.\\n", {reported});
                MPI_Abort(MPI_COMM_WORLD, 1);
            }}"""


@dace.library.expansion
class ExpandPgemmMKLMPICH(ExpandTransformation):
    environments = [environments.intel_mkl_mpich.IntelMKLScaLAPACKMPICH]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        a, b, c, desca, descb, gdescc, ldesc = node.validate(parent_sdfg, parent_state)
        dtype = a.dtype.base_type
        lapack_dtype_str = blas_helpers.to_blastype(dtype.type).lower()

        code = f"""
        {scalapack_grid_code(node, parent_state, True)}
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
            MKL_INT n_lc_rows = numroc_( &gc_rows, &lc_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT n_lc_cols = numroc_( &gc_cols, &lc_cols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT c_lld = max(n_lc_rows, 1);
            MKL_INT n_la_rows = numroc_( &ga_rows, &la_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT n_la_cols = numroc_( &ga_cols, &la_cols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT a_lld = max(n_la_rows, 1);
            MKL_INT n_lb_rows = numroc_( &gb_rows, &lb_rows, &__myprow, &__state->__mkl_int_zero, &__nprow);
            MKL_INT n_lb_cols = numroc_( &gb_cols, &lb_cols, &__mypcol, &__state->__mkl_int_zero, &__npcol);
            MKL_INT b_lld = max(n_lb_rows, 1);
{local_extent_guard(a, b, c)}
            MKL_INT info_c, info_a, info_b;
            MKL_INT _c_ldesc[9], _a_ldesc[9],  _b_ldesc[9];
            descinit_(_c_ldesc, &gc_rows, &gc_cols, &lc_rows, &lc_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &c_lld, &info_c);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &a_lld, &info_a);
            descinit_(_b_ldesc, &gb_rows, &gb_cols, &lb_rows, &lb_cols, &__state->__mkl_int_zero, &__state->__mkl_int_zero, &__ctxt, &b_lld, &info_b);
            if (info_c != 0 || info_a != 0 || info_b != 0) {{
                fprintf(stderr, "descinit_ refused a PGEMM descriptor: info(c=%d a=%d b=%d). A negative value is "
                                "minus the index of the argument it refused; the call would otherwise "
                                "return wrong numbers rather than fail.\\n", info_c, info_a, info_b);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }}
            MKL_INT _m = gc_rows, _n = gc_cols, _k = ga_rows;
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__mkl_int_one, &__state->__mkl_int_one, _b_ldesc,
                _a, &__state->__mkl_int_one, &__state->__mkl_int_one, _a_ldesc, &zero, _c, &__state->__mkl_int_one, &__state->__mkl_int_one, _c_ldesc);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
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
        {scalapack_grid_code(node, parent_state, False)}
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
            int n_lc_rows = numroc_( &gc_rows, &lc_rows, &__myprow, &__state->__int_zero, &__nprow);
            int n_lc_cols = numroc_( &gc_cols, &lc_cols, &__mypcol, &__state->__int_zero, &__npcol);
            int c_lld = max(n_lc_rows, 1);
            int n_la_rows = numroc_( &ga_rows, &la_rows, &__myprow, &__state->__int_zero, &__nprow);
            int n_la_cols = numroc_( &ga_cols, &la_cols, &__mypcol, &__state->__int_zero, &__npcol);
            int a_lld = max(n_la_rows, 1);
            int n_lb_rows = numroc_( &gb_rows, &lb_rows, &__myprow, &__state->__int_zero, &__nprow);
            int n_lb_cols = numroc_( &gb_cols, &lb_cols, &__mypcol, &__state->__int_zero, &__npcol);
            int b_lld = max(n_lb_rows, 1);
{local_extent_guard(a, b, c)}
            int info_c, info_a, info_b;
            int _c_ldesc[9], _a_ldesc[9],  _b_ldesc[9];
            descinit_(_c_ldesc, &gc_rows, &gc_cols, &lc_rows, &lc_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &c_lld, &info_c);
            descinit_(_a_ldesc, &ga_rows, &ga_cols, &la_rows, &la_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &a_lld, &info_a);
            descinit_(_b_ldesc, &gb_rows, &gb_cols, &lb_rows, &lb_cols, &__state->__int_zero, &__state->__int_zero, &__ctxt, &b_lld, &info_b);
            if (info_c != 0 || info_a != 0 || info_b != 0) {{
                fprintf(stderr, "descinit_ refused a PGEMM descriptor: info(c=%d a=%d b=%d). A negative value is "
                                "minus the index of the argument it refused; the call would otherwise "
                                "return wrong numbers rather than fail.\\n", info_c, info_a, info_b);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }}
            int _m = gc_rows, _n = gc_cols, _k = ga_rows;
            p{lapack_dtype_str}gemm_(
                &trans, &trans, &_m, &_n, &_k, &one, _b, &__state->__int_one, &__state->__int_one, _b_ldesc,
                _a, &__state->__int_one, &__state->__int_one, _a_ldesc, &zero, _c, &__state->__int_one, &__state->__int_one, _c_ldesc);
        """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
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
