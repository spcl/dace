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
class ExpandSyrkOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_Cin, ldc_in), ldc_out, n, k = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        prefix = func.lower()
        uplo = 'CblasUpper' if node.uplo else 'CblasLower'
        trans = 'CblasTrans' if node.transA else 'CblasNoTrans'
        a, b = node.alpha, node.beta
        code = f"""
        std::memcpy(_Cout, _Cin, sizeof({dt.ctype}) * ({n}) * ({ldc_in}));
        cblas_{prefix}syrk(CblasRowMajor, {uplo}, {trans}, {n}, {k}, ({dt.ctype})({a}), _A, {lda}, ({dt.ctype})({b}), _Cout, {ldc_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandSyrkMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandSyrkOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandSyrkCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda), (desc_Cin, ldc_in), ldc_out, n, k = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        uplo = 'CUBLAS_FILL_MODE_UPPER' if node.uplo else 'CUBLAS_FILL_MODE_LOWER'
        trans = 'CUBLAS_OP_T' if node.transA else 'CUBLAS_OP_N'
        a, b = node.alpha, node.beta
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dt.ctype} __alpha = ({dt.ctype})({a}); {dt.ctype} __beta = ({dt.ctype})({b});
        cudaMemcpyAsync(_Cout, _Cin, sizeof({dt.ctype}) * ({n}) * ({ldc_in}),
                        cudaMemcpyDeviceToDevice, __dace_current_stream);
        cublas{func}syrk(__dace_cublas_handle, {uplo}, {trans}, {n}, {k}, &__alpha, _A, {lda}, &__beta, _Cout, {ldc_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Syrk(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandSyrkOpenBLAS, "MKL": ExpandSyrkMKL, "cuBLAS": ExpandSyrkCuBLAS}
    default_implementation = None

    # Object fields
    uplo = dace.properties.Property(dtype=bool, default=False, desc="True if upper triangle of C is written")
    transA = dace.properties.Property(dtype=bool, default=False, desc="False: C += alpha A A^T; True: C += alpha A^T A")
    alpha = dace.properties.SymbolicProperty(allow_none=False, default=1)
    beta = dace.properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, uplo=False, transA=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs={"_A", "_Cin"}, outputs={"_Cout"}, **kwargs)
        self.uplo, self.transA, self.alpha, self.beta = uplo, transA, alpha, beta

    def validate(self, sdfg, state):
        """
        :return: A five-tuple ((A, lda), (C, ldc_in), ldc_out, n, k).
        """
        desc_A = desc_C = lda = ldc_in = ldc_out = None
        n = k = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_A":
                desc_A, lda = desc, desc.strides[dims[0]]
                if self.transA:
                    k = sq.size()[0]
                else:
                    k = sq.size()[1]
            elif e.dst_conn == "_Cin":
                desc_C, ldc_in = desc, desc.strides[dims[0]]
                n = sq.size()[0]
        for e in state.out_edges(self):
            if e.src_conn == "_Cout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                ldc_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if desc_A is None or desc_C is None:
            raise ValueError("SYRK needs _A and _Cin inputs and _Cout output")
        return (desc_A, lda), (desc_C, ldc_in), ldc_out, n, k


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.syrk')
@oprepo.replaces('dace.libraries.blas.Syrk')
def syrk_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 C,
                 result=None,
                 uplo=False,
                 transA=False,
                 alpha=1,
                 beta=0):
    result = result if result is not None else C
    A_in, C_in = state.add_read(A), state.add_read(C)
    C_out = state.add_write(result)

    libnode = Syrk('syrk', uplo=uplo, transA=transA, alpha=alpha, beta=beta)
    state.add_node(libnode)

    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(C_in, None, libnode, '_Cin', mm.Memlet(C))
    state.add_edge(libnode, '_Cout', C_out, None, mm.Memlet(result))

    return []
