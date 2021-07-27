# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.dtypes import ScheduleType, StorageType
from dace.sdfg import nodes, SDFG, SDFGState
from dace.data import Scalar

import dace.libraries.blas

# Define symbolic sizes for arbitrary inputs
M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')
L = dace.symbol('L')
O = dace.symbol('O')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

#####################################################################


@dace.program
def matmul_lib(A: dtype[M, K], B: dtype[K, N]):
    return A @ B


@dace.program
def three_matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L], D: dtype[L,
                                                                          O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)
    return matmul_lib(M1, M2)


@dace.program
def three_matmul_debug(A: dtype[M, K], B: dtype[K, N], C: dtype[N, L],
                       D: dtype[L, O]):
    M1 = matmul_lib(A, B)
    M2 = matmul_lib(C, D)
    return matmul_lib(M1, M2)


@pytest.mark.multigpu
def test_three_matmul():
    gpuHelper = 1
    gpuMain = 0
    dace.libraries.blas.default_implementation = 'cuBLAS'

    sdfg: dace.SDFG = three_matmul.to_sdfg()
    sdfg.name = 'gpu_p2p'
    sdfg.expand_library_nodes()

    state = sdfg.start_state
    output = state.sink_nodes()[0]
    mM1M2sdfg = state.predecessors(output)[0]
    m1 = state.predecessors(mM1M2sdfg)[0]
    m2 = state.predecessors(mM1M2sdfg)[1]
    mABsdfg = state.predecessors(m1)[0]
    mCDsdfg = state.predecessors(m2)[0]

    mABsdfg.location = {'gpu': gpuMain}
    mABsdfg.schedule = ScheduleType.GPU_Device

    mCDsdfg.location = {'gpu': gpuHelper}
    mCDsdfg.schedule = ScheduleType.GPU_Device

    mM1M2sdfg.location = {'gpu': gpuMain}
    mM1M2sdfg.schedule = ScheduleType.GPU_Device

    sdfg.arrays['M1'].location = {'gpu': gpuMain}
    sdfg.arrays['M1'].storage = StorageType.GPU_Global
    sdfg.arrays['M2'].location = {'gpu': gpuMain}
    sdfg.arrays['M2'].storage = StorageType.GPU_Global

    sdfg.apply_strict_transformations()

    np.random.seed(0)
    m = 1024
    k = 2000
    n = 3000
    l = 900
    o = 7777
    A = np.ndarray(shape=[m, k], dtype=np_dtype)
    B = np.ndarray(shape=[k, n], dtype=np_dtype)
    C = np.ndarray(shape=[n, l], dtype=np_dtype)
    D = np.ndarray(shape=[l, o], dtype=np_dtype)
    A[:] = np.random.rand(m, k)[:]
    B[:] = np.random.rand(k, n)[:]
    C[:] = np.random.rand(n, l)[:]
    D[:] = np.random.rand(l, o)[:]

    E = sdfg(A=A, B=B, C=C, D=D, M=m, K=k, N=n, L=l, O=o)
    res = (A @ B) @ (C @ D)
    idx = list(zip(*np.where(~np.isclose(E, res, atol=0, rtol=1e-7))))
    numErrors = len(idx)
    if numErrors > 0:
        print("number of errors:", numErrors)
    if numErrors < 100:
        for i in idx:
            print(i, E[i], res[i])
    assert np.allclose(E, res)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/cudastreams/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_three_matmul()
