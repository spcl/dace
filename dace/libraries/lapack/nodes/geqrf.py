# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LAPACK ``GEQRF`` library node — QR factorization of a general matrix.

Produces ``A := QR`` with ``Q`` stored implicitly as Householder
reflectors in the lower triangle of ``A`` plus a ``tau`` vector of
length ``min(m, n)``. Use :class:`Orgqr` to materialise ``Q``.

Uses ``_ain`` / ``_aout`` for the matrix to keep input and output
connectors distinct in the codegen.
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
class ExpandGeqrfOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda_in, lda_out, m, n), _ = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        if desc_A.dtype.veclen > 1:
            raise NotImplementedError
        lap = blas_helpers.to_blastype(dt.type).lower()
        code = f"""
        std::memcpy(_aout, _ain, sizeof({dt.ctype}) * ({m}) * ({lda_in}));
        _res = LAPACKE_{lap}geqrf(LAPACK_ROW_MAJOR, {m}, {n}, _aout, {lda_out}, _tau);
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandGeqrfMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGeqrfOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGeqrfGPUSolver(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda_in, lda_out, m, n), _ = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dt)
        func = func + 'geqrf'
        code = cls.environments[0].handle_setup_code(node) + f"""
            gpuMemcpyAsync(_aout, _ain, sizeof({dt.ctype}) * ({m}) * ({lda_in}),
                            gpuMemcpyDeviceToDevice, __dace_current_stream);
            """ + cls.call(func, cuda_type, m, n, lda_out)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        tasklet.out_connectors = {c: (dtypes.pointer(dtypes.int32) if c == '_res' else t) for c, t in conn.items()}
        return tasklet


@dace.library.expansion
class ExpandGeqrfCuSolverDn(ExpandGeqrfGPUSolver):
    environments = [environments.cusolverdn.cuSolverDn]

    @classmethod
    def call(cls, func, ctype, m, n, lda) -> str:
        return f"""
            int __dace_workspace_size = 0;
            {ctype}* __dace_workspace;
            cusolverDn{func}_bufferSize(
                __dace_cusolverDn_handle, {m}, {n}, _aout, {lda}, &__dace_workspace_size);
            gpuMalloc<{ctype}>(&__dace_workspace, sizeof({ctype}) * __dace_workspace_size);
            cusolverDn{func}(
                __dace_cusolverDn_handle, {m}, {n}, _aout, {lda}, _tau,
                __dace_workspace, __dace_workspace_size, _res);
            gpuFree(__dace_workspace);
            """


@dace.library.expansion
class ExpandGeqrfRocSolver(ExpandGeqrfGPUSolver):
    environments = [environments.rocsolver.rocSOLVER]

    @classmethod
    def call(cls, func, ctype, m, n, lda) -> str:
        # rocSOLVER manages its own workspace and reports no info code, so the status the caller
        # reads is written here rather than by the solver.
        return f"""
            dace::lapack::CheckRocsolverError(rocsolver_{func.lower()}(
                __dace_rocblas_handle, {m}, {n}, _aout, {lda}, _tau));
            DACE_GPU_CHECK(gpuMemsetAsync(_res, 0, sizeof(int), __dace_current_stream));
            """


@dace.library.node
class Geqrf(dace.sdfg.nodes.LibraryNode):
    """LAPACK ``?GEQRF``: QR factorization, ``A := QR``.

    Inputs: ``_ain``. Outputs: ``_aout`` (R + Householder reflectors),
    ``_tau`` (scalar vector of length ``min(m, n)``), ``_res`` (info).
    """

    implementations = {
        "OpenBLAS": ExpandGeqrfOpenBLAS,
        "MKL": ExpandGeqrfMKL,
        "cuSolverDn": ExpandGeqrfCuSolverDn,
        "rocSOLVER": ExpandGeqrfRocSolver
    }
    default_implementation = None

    def __init__(self, name, **kwargs):
        super().__init__(name, inputs={"_ain"}, outputs=OrderedSet(('_aout', '_tau', '_res')), **kwargs)

    def validate(self, sdfg, state):
        """:return: ``((desc_A, lda_in, lda_out, m, n), desc_tau)``."""
        desc_A = lda_in = lda_out = m = n = desc_tau = None
        for e in state.in_edges(self):
            if e.dst_conn != "_ain":
                continue
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            if len(sq.size()) != 2:
                raise ValueError("GEQRF: _ain must be 2-D")
            desc_A = sdfg.arrays[e.data.data]
            lda_in = desc_A.strides[dims[0]]
            m, n = sq.size()[0], sq.size()[1]
        for e in state.out_edges(self):
            if e.src_conn == "_aout":
                sq = copy.deepcopy(e.data.subset)
                dims = sq.squeeze()
                lda_out = sdfg.arrays[e.data.data].strides[dims[0]]
            elif e.src_conn == "_tau":
                desc_tau = sdfg.arrays[e.data.data]
        if desc_A is None:
            raise ValueError("GEQRF needs _ain input and _aout output")
        return (desc_A, lda_in, lda_out, m, n), desc_tau
