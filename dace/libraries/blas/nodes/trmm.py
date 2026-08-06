# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo


def _cblas_flags(node):
    return ('CblasLeft' if not node.side else 'CblasRight', 'CblasUpper' if node.uplo else 'CblasLower',
            'CblasTrans' if node.transA else 'CblasNoTrans', 'CblasUnit' if node.unit_diag else 'CblasNonUnit')


def _cublas_flags(node):
    return ('CUBLAS_SIDE_LEFT' if not node.side else 'CUBLAS_SIDE_RIGHT',
            'CUBLAS_FILL_MODE_UPPER' if node.uplo else 'CUBLAS_FILL_MODE_LOWER',
            'CUBLAS_OP_T' if node.transA else 'CUBLAS_OP_N',
            'CUBLAS_DIAG_UNIT' if node.unit_diag else 'CUBLAS_DIAG_NON_UNIT')


@dace.library.expansion
class ExpandTrmmOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_Bin, ldb_in), ldb_out, m, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        prefix = func.lower()
        side, uplo, trans, diag = _cblas_flags(node)
        a = node.alpha
        code = f"""
        std::memcpy(_Bout, _Bin, sizeof({dt.ctype}) * ({m}) * ({ldb_in}));
        cblas_{prefix}trmm(CblasRowMajor, {side}, {uplo}, {trans}, {diag}, {m}, {n}, ({dt.ctype})({a}), _A, {lda}, _Bout, {ldb_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandTrmmMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandTrmmOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandTrmmCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_Bin, ldb_in), ldb_out, m, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        side, uplo, trans, diag = _cublas_flags(node)
        a = node.alpha
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dt.ctype} __alpha = ({dt.ctype})({a});
        cublas{func}trmm(__dace_cublas_handle, {side}, {uplo}, {trans}, {diag}, {m}, {n}, &__alpha, _A, {lda}, _Bin, {ldb_in}, _Bout, {ldb_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Trmm(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandTrmmOpenBLAS, "MKL": ExpandTrmmMKL, "cuBLAS": ExpandTrmmCuBLAS}
    default_implementation = None

    # Object fields
    side = dace.properties.Property(dtype=bool,
                                    default=False,
                                    desc="False: B := alpha op(A) B; True: B := alpha B op(A)")
    uplo = dace.properties.Property(dtype=bool, default=False, desc="True for upper triangular A")
    transA = dace.properties.Property(dtype=bool, default=False, desc="True to use A^T")
    unit_diag = dace.properties.Property(dtype=bool, default=False, desc="True for implicit unit diagonal")
    alpha = dace.properties.SymbolicProperty(allow_none=False, default=1)

    def __init__(self, name, side=False, uplo=False, transA=False, unit_diag=False, alpha=1, **kwargs):
        super().__init__(name, inputs={"_A", "_Bin"}, outputs={"_Bout"}, **kwargs)
        self.side, self.uplo, self.transA, self.unit_diag, self.alpha = side, uplo, transA, unit_diag, alpha

    def validate(self, sdfg, state):
        """
        :return: A five-tuple ((A, lda), (Bin, ldb_in), ldb_out, m, n).
        """
        desc_A = desc_B = lda = ldb_in = ldb_out = m = n = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_A":
                desc_A, lda = desc, desc.strides[dims[0]]
            elif e.dst_conn == "_Bin":
                desc_B, ldb_in = desc, desc.strides[dims[0]]
                m, n = sq.size()[0], sq.size()[1]
        for e in state.out_edges(self):
            if e.src_conn == "_Bout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                ldb_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if desc_A is None or desc_B is None:
            raise ValueError("TRMM needs _A and _Bin inputs and _Bout output")
        return (desc_A, lda), (desc_B, ldb_in), ldb_out, m, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.trmm')
@oprepo.replaces('dace.libraries.blas.Trmm')
def trmm_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 B,
                 result=None,
                 side=False,
                 uplo=False,
                 transA=False,
                 unit_diag=False,
                 alpha=1):
    result = result if result is not None else B
    A_in, B_in = state.add_read(A), state.add_read(B)
    B_out = state.add_write(result)

    libnode = Trmm('trmm', side=side, uplo=uplo, transA=transA, unit_diag=unit_diag, alpha=alpha)
    state.add_node(libnode)

    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_Bin', mm.Memlet(B))
    state.add_edge(libnode, '_Bout', B_out, None, mm.Memlet(result))

    return []
