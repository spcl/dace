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
class ExpandOrgqrCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_A, lda_in, lda_out, m, n), (desc_tau, k) = node.validate(parent_sdfg, parent_state)
        dt = desc_A.dtype.base_type
        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dt)
        func = func + 'orgqr'
        code = environments.cusolverdn.cuSolverDn.handle_setup_code(node) + f"""
            cudaMemcpyAsync(_aout, _ain, sizeof({dt.ctype}) * ({m}) * ({lda_in}),
                            cudaMemcpyDeviceToDevice, __dace_current_stream);
            int __dace_workspace_size = 0;
            {cuda_type}* __dace_workspace;
            cusolverDn{func}_bufferSize(
                __dace_cusolverDn_handle, {m}, {n}, {k}, _aout, {lda_out}, _tau, &__dace_workspace_size);
            cudaMalloc<{cuda_type}>(&__dace_workspace, sizeof({cuda_type}) * __dace_workspace_size);
            cusolverDn{func}(
                __dace_cusolverDn_handle, {m}, {n}, {k}, _aout, {lda_out}, _tau,
                __dace_workspace, __dace_workspace_size, _res);
            cudaFree(__dace_workspace);
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
class Orgqr(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {"OpenBLAS": ExpandOrgqrOpenBLAS, "MKL": ExpandOrgqrMKL, "cuSolverDn": ExpandOrgqrCuSolverDn}
    default_implementation = None

    def __init__(self, name, **kwargs):
        super().__init__(name, inputs={"_ain", "_tau"}, outputs={"_aout", "_res"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A two-tuple ((A, lda_in, lda_out, m, n), (tau, k)).
        """
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
