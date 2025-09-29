# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import copy
import pytest
import numpy
from dace.transformation.passes.explicit_vectorization_cpu import ExplicitVectorizationPipelineCPU
from dace.transformation.passes.explicit_vectorization_gpu import ExplicitVectorizationPipelineGPU

N = dace.symbol('N')


@dace.program
def vadds_gpu(A: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vadds_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def tets_no_map_tasklet():
    pass


S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")


@dace.program
def tasklet_in_nested_sdfg(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    offset1: dace.int64,
    offset2: dace.int64,
):
    for i, j in dace.map[S1:S2:1, S1:S2:1] @ dace.dtypes.ScheduleType.Sequential:
        a[i, j] = ((1.5 * b[i + offset1, j + offset2]) + (2.0 * a[i + offset1, j + offset2])) / 3.5


@dace.program
def test_tasklets_in_if(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[S1:S2:1]:
        for j in dace.map[S1:S2:1]:
            if a[i, j] > c:
                b[i, j] = b[i, j] + d[i, j]
            else:
                b[i, j] = b[i, j] - d[i, j]
            b[i, j] = (1 - a[i, j]) * c[i, j]


@pytest.mark.gpu
def test_simple_gpu():
    import cupy

    # Allocate 64x64 GPU arrays using CuPy
    A_gpu = cupy.random.random((64, 64), dtype=cupy.float64)
    B_gpu = cupy.random.random((64, 64), dtype=cupy.float64)

    # Create copies for comparison
    A_orig = cupy.copy(A_gpu)
    B_orig = cupy.copy(B_gpu)
    A_vec = cupy.copy(A_gpu)
    B_vec = cupy.copy(B_gpu)

    # Original SDFG
    sdfg = vadds_gpu.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineGPU(vector_width=4).apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(A=A_orig, B=B_orig, N=64)
    c_copy_sdfg(A=A_vec, B=B_vec, N=64)

    # Compare results
    assert cupy.allclose(A_orig, A_vec)
    assert cupy.allclose(B_orig, B_vec)


def test_simple_cpu():
    import numpy

    # Allocate 64x64 CPU arrays using NumPy
    A_cpu = numpy.random.random((64, 64))
    B_cpu = numpy.random.random((64, 64))

    # Create copies for comparison
    A_orig = A_cpu.copy()
    B_orig = B_cpu.copy()
    A_vec = A_cpu.copy()
    B_vec = B_cpu.copy()

    # Original SDFG
    sdfg = vadds_cpu.to_sdfg()
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineCPU(vector_width=4).apply_pass(copy_sdfg, {})
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(A=A_orig, B=B_orig, N=64)
    c_copy_sdfg(A=A_vec, B=B_vec, N=64)

    # Compare results
    assert numpy.allclose(A_orig, A_vec)
    assert numpy.allclose(B_orig, B_vec)


def test_nested_sdfg():
    _S1 = 1
    _S2 = 65
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    # Create copies for comparison
    A_orig = copy.deepcopy(A)
    B_orig = copy.deepcopy(B)
    A_vec = copy.deepcopy(A)
    B_vec = copy.deepcopy(B)

    # Original SDFG
    sdfg = tasklet_in_nested_sdfg.to_sdfg()

    # Vectorized SDFG
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=A_orig, b=B_orig, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)
    c_copy_sdfg(a=A_vec, b=B_vec, S=_S, S1=_S1, S2=_S2, offset1=-1, offset2=-1)

    # Compare results
    assert numpy.allclose(A_orig, A_vec)
    assert numpy.allclose(B_orig, B_vec)


n = dace.symbol('n')  # number of rows
m = dace.symbol('m')  # number of columns
nnz = dace.symbol('nnz')  # number of nonzeros


@dace.program
def spmv_csr(indptr: dace.int32[n + 1], indices: dace.int32[nnz], data: dace.float64[nnz], x: dace.float64[m],
             y: dace.float64[n]):
    n_rows = len(indptr) - 1

    for i in dace.map[0:n_rows:1]:
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for idx in dace.map[row_start:row_end:1]:
            j = indices[idx]
            y[i] = y[i] + data[idx] * x[j]


def test_spmv():
    sdfg = spmv_csr.to_sdfg()
    copy_sdfg = copy.deepcopy(sdfg)
    ExplicitVectorizationPipelineCPU(vector_width=8).apply_pass(copy_sdfg, {})

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()
    sdfg.save("a.sdfg")


if __name__ == "__main__":
    # test_simple()
    # test_simple_cpu()
    test_spmv()
    #test_nested_sdfg()
