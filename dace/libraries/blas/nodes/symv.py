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


@dace.library.expansion
class ExpandSymvOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (_, sx), (_, syi), syo, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        prefix = func.lower()
        uplo = 'CblasUpper' if node.uplo else 'CblasLower'
        a, b = node.alpha, node.beta
        code = f"""
        cblas_{prefix}copy({n}, _yin, {syi}, _yout, {syo});
        cblas_{prefix}symv(CblasColMajor, {uplo}, {n}, ({dt.ctype})({a}), _A, {lda}, _x, {sx}, ({dt.ctype})({b}), _yout, {syo});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandSymvMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandSymvOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandSymvCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (_, sx), (_, syi), syo, n = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        uplo = 'CUBLAS_FILL_MODE_UPPER' if node.uplo else 'CUBLAS_FILL_MODE_LOWER'
        a, b = node.alpha, node.beta
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dt.ctype} __alpha = ({dt.ctype})({a}); {dt.ctype} __beta = ({dt.ctype})({b});
        cublas{func}copy(__dace_cublas_handle, {n}, _yin, {syi}, _yout, {syo});
        cublas{func}symv(__dace_cublas_handle, {uplo}, {n}, &__alpha, _A, {lda}, _x, {sx}, &__beta, _yout, {syo});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Symv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandSymvOpenBLAS, "MKL": ExpandSymvMKL, "cuBLAS": ExpandSymvCuBLAS}
    default_implementation = None

    # Object fields
    uplo = dace.properties.Property(dtype=bool, default=False, desc="True if upper triangle of A is referenced")
    alpha = dace.properties.SymbolicProperty(allow_none=False, default=1)
    beta = dace.properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, uplo=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs={"_A", "_x", "_yin"}, outputs={"_yout"}, **kwargs)
        self.uplo, self.alpha, self.beta = uplo, alpha, beta

    def validate(self, sdfg, state):
        """
        :return: A five-tuple ((A, lda), (x, sx), (y, syi), syo, n).
        """
        descs, strides = {}, {}
        n = syo = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            descs[e.dst_conn] = desc
            strides[e.dst_conn] = desc.strides[dims[0]]
            if e.dst_conn == "_A":
                n = sq.size()[0]
        for e in state.out_edges(self):
            if e.src_conn == "_yout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                syo = sdfg.arrays[e.data.data].strides[dims[0]]
        for k in ("_A", "_x", "_yin"):
            if k not in descs:
                raise ValueError(f"SYMV needs _A, _x and _yin inputs (missing {k})")
        return ((descs["_A"], strides["_A"]), (descs["_x"], strides["_x"]), (descs["_yin"], strides["_yin"]), syo, n)


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.symv')
@oprepo.replaces('dace.libraries.blas.Symv')
def symv_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, A, x, y, result=None, alpha=1, beta=0, uplo=False):
    result = result if result is not None else y
    A_in, x_in, y_in = (state.add_read(name) for name in (A, x, y))
    y_out = state.add_write(result)

    libnode = Symv('symv', uplo=uplo, alpha=alpha, beta=beta)
    state.add_node(libnode)

    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_yin', mm.Memlet(y))
    state.add_edge(libnode, '_yout', y_out, None, mm.Memlet(result))

    return []
