# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-2 ``TRMV`` library node — ``y := op(A) * x`` with triangular ``A``.

Modeled with ``_xin`` input and ``_xout`` output following the :class:`Trsv` pattern.
The expansion copies ``_xin`` into ``_xout`` then calls cBLAS / cuBLAS triangular MV
in place on ``_xout``.
"""
import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace.libraries.blas import gpu_dialect
from dace import memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
from dace.ordered import OrderedSet


def _cblas_flags(node):
    return ('CblasUpper' if node.uplo else 'CblasLower', 'CblasTrans' if node.transA else 'CblasNoTrans',
            'CblasUnit' if node.unit_diag else 'CblasNonUnit')


def _gpu_flags(node, dialect):
    """(uplo, trans, diag): a column-major library reads the row-major A as A^T, so both flip."""
    return dialect.fill(not node.uplo), dialect.op('N' if node.transA else 'T'), dialect.diag(node.unit_diag)


@dace.library.expansion
class ExpandTrmvOpenBLAS(ExpandTransformation):

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
        cblas_{prefix}trmv(CblasRowMajor, {uplo}, {trans}, {diag}, {n}, _A, {lda}, _xout, {sx_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandTrmvMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandTrmvOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandTrmvGPUBLAS(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_x, sx_in), sx_out, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        uplo, trans, diag = _gpu_flags(node, cls.dialect)
        code = cls.environments[0].handle_setup_code(node)
        code += f"""
        {cls.dialect.func(func, 'copy')}({cls.dialect.handle}, {n}, _xin, {sx_in}, _xout, {sx_out});
        {cls.dialect.check_error}({cls.dialect.func(func, 'trmv')}({cls.dialect.handle}, {uplo}, {trans}, {diag}, {n}, _A, {lda}, _xout, {sx_out}));
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandTrmvCuBLAS(ExpandTrmvGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    dialect = gpu_dialect.CUBLAS


@dace.library.expansion
class ExpandTrmvRocBLAS(ExpandTrmvGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    dialect = gpu_dialect.ROCBLAS


@dace.library.node
class Trmv(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?TRMV``: triangular matrix-vector multiply, ``_xout := op(A) * _xin``."""

    implementations = {
        "OpenBLAS": ExpandTrmvOpenBLAS,
        "MKL": ExpandTrmvMKL,
        "cuBLAS": ExpandTrmvCuBLAS,
        "rocBLAS": ExpandTrmvRocBLAS
    }
    default_implementation = None

    uplo = dace.properties.Property(dtype=bool, default=False, desc="True for upper triangular A")
    transA = dace.properties.Property(dtype=bool, default=False, desc="True to use A^T")
    unit_diag = dace.properties.Property(dtype=bool, default=False, desc="True if implicit unit diagonal")

    def __init__(self, name, uplo=False, transA=False, unit_diag=False, **kwargs):
        super().__init__(name, inputs=OrderedSet(('_A', '_xin')), outputs={"_xout"}, **kwargs)
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
            raise ValueError("TRMV needs _A and _xin inputs and _xout output")
        return (desc_A, lda), (desc_x, sx_in), sx_out, n


@oprepo.replaces('dace.libraries.blas.trmv')
@oprepo.replaces('dace.libraries.blas.Trmv')
def trmv_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 x,
                 result=None,
                 uplo=False,
                 transA=False,
                 unit_diag=False):
    """Build a :class:`Trmv` node. ``result`` defaults to ``x`` for in-place semantics."""
    result = result if result is not None else x
    A_in, x_in = state.add_read(A), state.add_read(x)
    x_out = state.add_write(result)
    libnode = Trmv('trmv', uplo=uplo, transA=transA, unit_diag=unit_diag)
    state.add_node(libnode)
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_xin', mm.Memlet(x))
    state.add_edge(libnode, '_xout', x_out, None, mm.Memlet(result))
    return []
