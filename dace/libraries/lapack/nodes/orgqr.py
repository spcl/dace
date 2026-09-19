# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LAPACK ``ORGQR`` library node — materialise explicit ``Q`` from the compact GEQRF output.

Uses ``_ain`` / ``_aout`` for the matrix so input and output connectors
stay distinct in codegen.
"""
import copy

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers
from dace.ordered import OrderedSet


@dace.library.expansion
class ExpandOrgqrOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda_in, lda_out, m, n), (desc_tau, k) = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        if desc_A.dtype.veclen > 1:
            raise NotImplementedError
        lap = blas_helpers.to_blastype(dt.type).lower()
        code = f"""
        std::memcpy(_aout, _ain, sizeof({dt.ctype}) * ({m}) * ({lda_in}));
        _res = LAPACKE_{lap}orgqr(LAPACK_ROW_MAJOR, {m}, {n}, {k}, _aout, {lda_out}, _tau);
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandOrgqrMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandOrgqrOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandOrgqrGPUSolver(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda_in, lda_out, m, n), (desc_tau, k) = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dt)
        func = func + 'orgqr'
        code = cls.environments[0].handle_setup_code(node) + f"""
            gpuMemcpyAsync(_aout, _ain, sizeof({dt.ctype}) * ({m}) * ({lda_in}),
                            gpuMemcpyDeviceToDevice, __dace_current_stream);
            """ + cls.call(func, cuda_type, m, n, k, lda_out, dt)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        tasklet.out_connectors = {c: (dtypes.pointer(dtypes.int32) if c == '_res' else t) for c, t in conn.items()}
        return tasklet


@dace.library.expansion
class ExpandOrgqrCuSolverDn(ExpandOrgqrGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn]

    @classmethod
    def call(cls, func, ctype, m, n, k, lda, dt) -> str:
        return f"""
            int __dace_workspace_size = 0;
            {ctype}* __dace_workspace;
            cusolverDn{func}_bufferSize(
                __dace_cusolverDn_handle, {m}, {n}, {k}, _aout, {lda}, _tau, &__dace_workspace_size);
            gpuMalloc<{ctype}>(&__dace_workspace, sizeof({ctype}) * __dace_workspace_size);
            cusolverDn{func}(
                __dace_cusolverDn_handle, {m}, {n}, {k}, _aout, {lda}, _tau,
                __dace_workspace, __dace_workspace_size, _res);
            gpuFree(__dace_workspace);
            """


@dace.library.expansion
class ExpandOrgqrRocSolver(ExpandOrgqrGPUSolver):
    environments = [environments.rocsolver.rocSOLVER]

    @classmethod
    def call(cls, func, ctype, m, n, k, lda, dt) -> str:
        # LAPACK names this ORGQR for real operands and UNGQR for complex ones, and rocSOLVER
        # follows LAPACK: there is no `rocsolver_corgqr`. cuSolverDn papers over the split with one
        # camel-case name, so the letter alone does not determine the routine here.
        letter = func[0].lower()
        routine = "ungqr" if letter in ("c", "z") else "orgqr"
        return f"""
            dace::lapack::CheckRocsolverError(rocsolver_{letter}{routine}(
                __dace_rocblas_handle, {m}, {n}, {k}, _aout, {lda}, _tau));
            DACE_GPU_CHECK(gpuMemsetAsync(_res, 0, sizeof(int), __dace_current_stream));
            """


@dace.library.node
class Orgqr(dace.sdfg.nodes.LibraryNode):
    """LAPACK ``?ORGQR``: materialise explicit ``Q`` matrix from GEQRF output."""

    implementations = {
        "OpenBLAS": ExpandOrgqrOpenBLAS,
        "MKL": ExpandOrgqrMKL,
        "cuSolverDn": ExpandOrgqrCuSolverDn,
        "rocSOLVER": ExpandOrgqrRocSolver
    }
    default_implementation = None

    def __init__(self, name, **kwargs):
        super().__init__(name, inputs=OrderedSet(('_ain', '_tau')), outputs=OrderedSet(('_aout', '_res')), **kwargs)

    def validate(self, sdfg, state):
        """:return: ``((desc_A, lda_in, lda_out, m, n), (desc_tau, k))``."""
        desc_A = lda_in = lda_out = m = n = desc_tau = k = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            desc = sdfg.arrays[e.data.data]
            if e.dst_conn == "_ain":
                if len(sq.size()) != 2:
                    raise ValueError("ORGQR: _ain must be 2-D")
                desc_A, lda_in = desc, desc.strides[dims[0]]
                m, n = sq.size()[0], sq.size()[1]
            elif e.dst_conn == "_tau":
                if len(sq.size()) != 1:
                    raise ValueError("ORGQR: _tau must be 1-D")
                desc_tau, k = desc, sq.size()[0]
        for e in state.out_edges(self):
            if e.src_conn == "_aout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                lda_out = sdfg.arrays[e.data.data].strides[dims[0]]
        if desc_A is None or desc_tau is None:
            raise ValueError("ORGQR needs _ain and _tau inputs and _aout output")
        return (desc_A, lda_in, lda_out, m, n), (desc_tau, k)
