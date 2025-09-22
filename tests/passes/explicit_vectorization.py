# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import copy
import pytest
import cupy
from dace.transformation.passes.explicit_vectorization import ExplicitVectorizationPipelineGPU

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global,
         B: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j] + 2.0


@pytest.mark.gpu
def test_simple():
    # Allocate 64x64 GPU arrays using CuPy
    A_gpu = cupy.random.random((64, 64), dtype=cupy.float64)
    B_gpu = cupy.random.random((64, 64), dtype=cupy.float64)

    # Create copies for comparison
    A_orig = cupy.copy(A_gpu)
    B_orig = cupy.copy(B_gpu)
    A_vec = cupy.copy(A_gpu)
    B_vec = cupy.copy(B_gpu)

    # Original SDFG
    sdfg = vadd.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineGPU(vector_width=4).apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()
    copy_sdfg.save("a.sdfg")

    c_sdfg(A=A_orig, B=B_orig, N=64)
    c_copy_sdfg(A=A_vec, B=B_vec, N=64)

    # Compare results
    assert cupy.allclose(A_orig, A_vec)
    assert cupy.allclose(B_orig, B_vec)


if __name__ == "__main__":
    test_simple()
