# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the Memory Buffer transformation. """
import copy
import dace
import dace.libraries.blas
import networkx as nx
import numpy as np

from dace.transformation.dataflow import memory_buffering as mb
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.fpga_testing import xilinx_test
from dace.transformation.dataflow import Vectorization

N = 64
M = 64
K = 64


@dace.program
def vecadd_1_streaming(A: dace.float32[N], B: dace.float32[N]):
    B[:] = A + 1.0


@dace.program
def vecadd_streaming(A: dace.float32[N], B: dace.float32[N],
                     C: dace.float32[N]):
    C[:] = A + B


@dace.program
def matadd_streaming(A: dace.float32[M, N], B: dace.float32[M, N],
                     C: dace.float32[M, N]):
    C[:] = A + B

@dace.program
def matmul_streaming(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


# @dace.program
# def matmul_streaming(A: dace.float32[M, K], B: dace.float32[K, N],
#                      C: dace.float32[M, N]):
#     C[:] = A @ B


@xilinx_test()
def test_mem_buffer_vec_add_1():
    # Make SDFG
    sdfg: dace.SDFG = vecadd_1_streaming.to_sdfg()
    # Transform

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    # sdfg.apply_transformations_repeated(mb.MemoryBuffering)

    # assert sdfg.apply_transformations_repeated(
    #     mb.MemoryBuffering, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    sdfg(A=A, B=B)

    assert all(B == A + 1)

    return sdfg


@xilinx_test()
def test_mem_buffer_vec_add():
    # Make SDFG
    sdfg: dace.SDFG = vecadd_streaming.to_sdfg()
    # Transform

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    sdfg.apply_transformations_repeated(mb.MemoryBuffering)

    # assert sdfg.apply_transformations_repeated(
    #     mb.MemoryBuffering, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)
    C = np.random.rand(N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_mem_buffer_mat_add():
    # Make SDFG
    sdfg: dace.SDFG = matadd_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])


    # sdfg.apply_transformations_repeated(mb.MemoryBuffering)


    # assert sdfg.apply_transformations_repeated(
    #     mb.MemoryBuffering, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))

    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_mem_buffer_mat_mul():
    # Make SDFG
    sdfg: dace.SDFG = matmul_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    # sdfg.apply_transformations([mb.MemoryBuffering])

    # assert sdfg.apply_transformations_repeated(
    #     mb.MemoryBuffering, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A @ B))
    assert diff <= 1e-5

    return sdfg


if __name__ == "__main__":
    # test_mem_buffer_vec_add_1(None)
    # test_mem_buffer_vec_add(None)
    # test_mem_buffer_mat_add(None)
    test_mem_buffer_mat_mul(None)