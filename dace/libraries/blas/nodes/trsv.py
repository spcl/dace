# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-2 ``TRSV`` library node — triangular solve ``op(A) x = b``.

Modeled with separate ``_xin`` (RHS) and ``_xout`` (solution) connectors
following the :class:`Potrf` pattern so DaCe codegen doesn't generate
two declarations for the same name. The OpenBLAS / cuBLAS expansions
first copy ``_xin`` into ``_xout``, then call the in-place cBLAS /
cuBLAS triangular solve on ``_xout``.
"""
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
    return ('CblasUpper' if node.uplo else 'CblasLower', 'CblasTrans' if node.transA else 'CblasNoTrans',
            'CblasUnit' if node.unit_diag else 'CblasNonUnit')


def _cublas_flags(node):
    return ('CUBLAS_FILL_MODE_UPPER' if node.uplo else 'CUBLAS_FILL_MODE_LOWER',
            'CUBLAS_OP_T' if node.transA else 'CUBLAS_OP_N',
            'CUBLAS_DIAG_UNIT' if node.unit_diag else 'CUBLAS_DIAG_NON_UNIT')


@dace.library.expansion
class ExpandTrsvOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_x, sx_in), sx_out, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        prefix = func.lower()
        uplo, trans, diag = _cblas_flags(node)
        code = f"""
        cblas_{prefix}copy({n}, _xin, {sx_in}, _xout, {sx_out});
        cblas_{prefix}trsv(CblasColMajor, {uplo}, {trans}, {diag}, {n}, _A, {lda}, _xout, {sx_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandTrsvMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandTrsvOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandTrsvCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_x, sx_in), sx_out, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        uplo, trans, diag = _cublas_flags(node)
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        cublas{func}copy(__dace_cublas_handle, {n}, _xin, {sx_in}, _xout, {sx_out});
        cublas{func}trsv(__dace_cublas_handle, {uplo}, {trans}, {diag}, {n}, _A, {lda}, _xout, {sx_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Trsv(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?TRSV``: solve ``op(A) x = b`` where ``A`` is triangular.

    Inputs: ``_A`` (triangular matrix), ``_xin`` (right-hand-side).
    Outputs: ``_xout`` (solution).
    """

    implementations = {"OpenBLAS": ExpandTrsvOpenBLAS, "MKL": ExpandTrsvMKL, "cuBLAS": ExpandTrsvCuBLAS}
    default_implementation = None

    uplo = dace.properties.Property(dtype=bool, default=False, desc="True for upper triangular A")
    transA = dace.properties.Property(dtype=bool, default=False, desc="True to solve op(A)=A^T")
    unit_diag = dace.properties.Property(dtype=bool, default=False, desc="True if A has implicit unit diagonal")

    def __init__(self, name, uplo=False, transA=False, unit_diag=False, **kwargs):
        super().__init__(name, inputs={"_A", "_xin"}, outputs={"_xout"}, **kwargs)
        self.uplo, self.transA, self.unit_diag = uplo, transA, unit_diag

    def validate(self, sdfg, state):
        """:return: ``((desc_A, lda), (desc_x, sx_in), sx_out, n)``."""
        desc_A = desc_x = lda = sx_in = sx_out = n = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_A":
                desc_A, lda = desc, desc.strides[dims[0]]
                n = sq.size()[0]
            elif e.dst_conn == "_xin":
                desc_x, sx_in = desc, desc.strides[dims[0]]
        for e in state.out_edges(self):
            if e.src_conn == "_xout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                sx_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if desc_A is None or desc_x is None:
            raise ValueError("TRSV needs _A and _xin inputs and _xout output")
        return (desc_A, lda), (desc_x, sx_in), sx_out, n


@oprepo.replaces('dace.libraries.blas.trsv')
@oprepo.replaces('dace.libraries.blas.Trsv')
def trsv_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 x,
                 result=None,
                 uplo=False,
                 transA=False,
                 unit_diag=False):
    """Build a :class:`Trsv` node. ``result`` defaults to ``x`` for in-place semantics."""
    result = result if result is not None else x
    A_in, x_in = state.add_read(A), state.add_read(x)
    x_out = state.add_write(result)
    libnode = Trsv('trsv', uplo=uplo, transA=transA, unit_diag=unit_diag)
    state.add_node(libnode)
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_xin', mm.Memlet(x))
    state.add_edge(libnode, '_xout', x_out, None, mm.Memlet(result))
    return []
