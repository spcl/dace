# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandPotrsOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda, n_A), (desc_B, ldb_in, ldb_out, nrhs) = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        if desc_A.dtype.veclen > 1:
            raise NotImplementedError
        lap = blas_helpers.to_blastype(dt.type).lower()
        uplo = "'L'" if node.lower else "'U'"
        code = f"""
        std::memcpy(_bout, _bin, sizeof({dt.ctype}) * ({n_A}) * ({ldb_in}));
        _res = LAPACKE_{lap}potrs(LAPACK_ROW_MAJOR, {uplo}, {n_A}, {nrhs}, _a, {lda}, _bout, {ldb_out});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandPotrsMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandPotrsOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandPotrsCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda, n_A), (desc_B, ldb_in, ldb_out, nrhs) = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, _, _ = blas_helpers.cublas_type_metadata(dt)
        func = func + 'potrs'
        uplo = "CUBLAS_FILL_MODE_LOWER" if node.lower else "CUBLAS_FILL_MODE_UPPER"
        code = environments.cusolverdn.cuSolverDn.handle_setup_code(node) + f"""
            cudaMemcpyAsync(_bout, _bin, sizeof({dt.ctype}) * ({n_A}) * ({ldb_in}),
                            cudaMemcpyDeviceToDevice, __dace_current_stream);
            cusolverDn{func}(
                __dace_cusolverDn_handle, {uplo}, {n_A}, {nrhs},
                _a, {lda}, _bout, {ldb_out}, _res);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        tasklet.out_connectors = {c: (dtypes.pointer(dtypes.int32) if c == '_res' else t) for c, t in conn.items()}
        return tasklet


@dace.library.node
class Potrs(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandPotrsOpenBLAS, "MKL": ExpandPotrsMKL, "cuSolverDn": ExpandPotrsCuSolverDn}
    default_implementation = None

    # Object fields
    lower = dace.properties.Property(dtype=bool, default=True, desc="True if the factor in _a is lower triangular")

    def __init__(self, name, lower=True, **kwargs):
        super().__init__(name, inputs={"_a", "_bin"}, outputs={"_bout", "_res"}, **kwargs)
        self.lower = lower

    def validate(self, sdfg, state):
        """
        :return: A two-tuple ((A, lda, n), (Bin, ldb_in, ldb_out, nrhs)).
        """
        desc_A = desc_B = lda = ldb_in = ldb_out = None
        n_A = nrhs = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_a":
                if len(sq.size()) != 2:
                    raise ValueError("POTRS: _a must be 2-D (the Cholesky factor)")
                desc_A, lda, n_A = desc, desc.strides[dims[0]], sq.size()[0]
            elif e.dst_conn == "_bin":
                if len(sq.size()) not in (1, 2):
                    raise ValueError("POTRS: _bin must be 1-D or 2-D")
                desc_B, ldb_in = desc, desc.strides[dims[0]]
                nrhs = sq.size()[1] if len(sq.size()) == 2 else 1
        for e in state.out_edges(self):
            if e.src_conn == "_bout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                ldb_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if desc_A is None or desc_B is None:
            raise ValueError("POTRS needs _a and _bin inputs and _bout output")
        return (desc_A, lda, n_A), (desc_B, ldb_in, ldb_out, nrhs)
