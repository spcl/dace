# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-3 ``SYMM`` library node — ``C := alpha A B + beta C`` (or right-side) with symmetric ``A``.

Uses separate ``_Cin`` and ``_Cout`` connectors (see :class:`Trsm`).
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


@dace.library.expansion
class ExpandSymmOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_B, ldb), (desc_Cin, ldc_in), ldc_out, m, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        prefix = func.lower()
        side = 'CblasLeft' if not node.side else 'CblasRight'
        uplo = 'CblasUpper' if node.uplo else 'CblasLower'
        a, b = node.alpha, node.beta
        code = f"""
        std::memcpy(_Cout, _Cin, sizeof({dt.ctype}) * ({m}) * ({ldc_in}));
        cblas_{prefix}symm(CblasRowMajor, {side}, {uplo}, {m}, {n}, ({dt.ctype})({a}), _A, {lda}, _B, {ldb}, ({dt.ctype})({b}), _Cout, {ldc_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandSymmMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandSymmOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandSymmCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_B, ldb), (desc_Cin, ldc_in), ldc_out, m, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        side = 'CUBLAS_SIDE_LEFT' if not node.side else 'CUBLAS_SIDE_RIGHT'
        uplo = 'CUBLAS_FILL_MODE_UPPER' if node.uplo else 'CUBLAS_FILL_MODE_LOWER'
        a, b = node.alpha, node.beta
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dt.ctype} __alpha = ({dt.ctype})({a}); {dt.ctype} __beta = ({dt.ctype})({b});
        cudaMemcpyAsync(_Cout, _Cin, sizeof({dt.ctype}) * ({m}) * ({ldc_in}),
                        cudaMemcpyDeviceToDevice, __dace_current_stream);
        cublas{func}symm(__dace_cublas_handle, {side}, {uplo}, {m}, {n}, &__alpha, _A, {lda}, _B, {ldb}, &__beta, _Cout, {ldc_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Symm(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?SYMM``: symmetric matrix-matrix multiply."""

    implementations = {"OpenBLAS": ExpandSymmOpenBLAS, "MKL": ExpandSymmMKL, "cuBLAS": ExpandSymmCuBLAS}
    default_implementation = None

    side = dace.properties.Property(dtype=bool,
                                    default=False,
                                    desc="False: C = alpha A B + beta C; True: C = alpha B A + beta C")
    uplo = dace.properties.Property(dtype=bool, default=False, desc="True if upper triangle of A is referenced")
    alpha = dace.properties.SymbolicProperty(allow_none=False, default=1)
    beta = dace.properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, side=False, uplo=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs={"_A", "_B", "_Cin"}, outputs={"_Cout"}, **kwargs)
        self.side, self.uplo, self.alpha, self.beta = side, uplo, alpha, beta

    def validate(self, sdfg, state):
        """:return: ``((desc_A, lda), (desc_B, ldb), (desc_C, ldc_in), ldc_out, m, n)``."""
        descs, strides = {}, {}
        m = n = ldc_out = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            descs[e.dst_conn] = desc
            strides[e.dst_conn] = desc.strides[dims[0]]
            if e.dst_conn == "_Cin":
                m, n = sq.size()[0], sq.size()[1]
        for e in state.out_edges(self):
            if e.src_conn == "_Cout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                ldc_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if not all(k in descs for k in ("_A", "_B", "_Cin")):
            raise ValueError("SYMM needs _A, _B, _Cin inputs and _Cout output")
        return ((descs["_A"], strides["_A"]), (descs["_B"], strides["_B"]), (descs["_Cin"], strides["_Cin"]), ldc_out,
                m, n)


@oprepo.replaces('dace.libraries.blas.symm')
@oprepo.replaces('dace.libraries.blas.Symm')
def symm_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 B,
                 C,
                 result=None,
                 side=False,
                 uplo=False,
                 alpha=1,
                 beta=0):
    """Build a :class:`Symm` node. ``result`` defaults to ``C`` for in-place semantics."""
    result = result if result is not None else C
    A_in, B_in, C_in = (state.add_read(name) for name in (A, B, C))
    C_out = state.add_write(result)
    libnode = Symm('symm', side=side, uplo=uplo, alpha=alpha, beta=beta)
    state.add_node(libnode)
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_B', mm.Memlet(B))
    state.add_edge(C_in, None, libnode, '_Cin', mm.Memlet(C))
    state.add_edge(libnode, '_Cout', C_out, None, mm.Memlet(result))
    return []
