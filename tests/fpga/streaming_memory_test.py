# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the StreamingMemory transformation. """
import copy

from numpy.core.numeric import allclose
import pytest
import dace
import dace.libraries.blas
import networkx as nx
import numpy as np

from dace.transformation.dataflow import streaming_memory as sm, MapExpansion
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.fpga_testing import xilinx_test
from dace.transformation.auto.fpga import fpga_rr_interleave_containers_to_banks

M, N, K = 64, 64, 64

M_s = dace.symbol('M_s')
N_s = dace.symbol('N_s')
K_s = dace.symbol('K_s')


@dace.program
def two_maps_kernel_legal(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N],
                          E: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_a << A[i]
        in_b << B[i]
        out >> D[i]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N]):
        in_b << B[i]
        in_c << C[i]
        out >> E[i]
        out = in_b + in_c


@dace.program
def two_maps_kernel_illegal(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N],
                            E: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_a << A[i]
        in_b << B[i]
        out >> D[i]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N:2]):
        in_b << B[i]
        in_c << C[i]
        out >> E[i]
        out = in_b + in_c


@dace.program
def bicg(A: dace.float32[N, M], p: dace.float32[M], r: dace.float32[N]):
    return r @ A, A @ p


@dace.program
def atax(A: dace.float32[M, N], x: dace.float32[N]):
    return (A @ x) @ A


@dace.program
def vecadd_1_streaming(A: dace.float32[N], B: dace.float32[N]):
    B[:] = A + 1.0


@dace.program
def vecadd_1_streaming_non_appl_0(A: dace.float32[N], B: dace.float32[N]):
    for i in dace.map[0:61]:
        with dace.tasklet:
            in_A << A[i]
            out >> B[i]
            out = in_A + 1.0


@dace.program
def vecadd_1_streaming_non_appl_1(A: dace.float32[N], B: dace.float32[N]):
    for i in dace.map[0:N:2]:
        with dace.tasklet:
            in_A << A[i]
            out >> B[i]
            out = in_A + 1.0


@dace.program
def vecadd_1_streaming_symbol(A: dace.float32[N_s], B: dace.float32[N_s]):
    B[:] = A + 1.0


@dace.program
def vecadd_streaming(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N]):
    C[:] = A + B


def vecadd_streaming_type(type0, type1, type2):
    @dace.program
    def vecadd_streaming_type_kernel(A: type0[N], B: type1[N], C: type2[N]):
        C[:] = A + B

    return vecadd_streaming_type_kernel


@dace.program
def matadd_streaming(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N]):
    C[:] = A + B


@dace.program
def matadd_streaming_symbol(A: dace.float32[M_s, N_s], B: dace.float32[M_s, N_s], C: dace.float32[M_s, N_s]):
    C[:] = A + B


@dace.program
def matadd_streaming_bad_stride(A: dace.float32[M + 1, N + 1], B: dace.float32[M + 1, N + 1], C: dace.float32[M + 1,
                                                                                                              N + 1]):
    C[:] = A + B


@dace.program
def tensoradd_streaming(A: dace.float32[M, N, K], B: dace.float32[M, N, K], C: dace.float32[M, N, K]):
    C[:] = A + B


@dace.program
def maporder_streaming(A: dace.float32[N, N, N], B: dace.float32[N, N, N], C: dace.float32[N, N, N],
                       D: dace.float32[N, N, N], E: dace.float32[N, N, N], F: dace.float32[N, N,
                                                                                           N], G: dace.float32[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            in_A << A[i, j, 0]  # No
            in_B << B[i, 0, j]  # Yes
            in_C << C[0, i, j]  # Yes
            in_D << D[j, i, 0]  # No
            in_E << E[j, 0, i]  # No
            in_F << F[0, j, i]  # No
            out >> G[i, j]  # Yes

            out = in_A + in_B + in_C + in_D + in_E + in_F


@dace.program
def matadd_multistream(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N], D: dace.float32[M, N]):
    C[:] = A + B
    D[:] = A - B


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


@dace.program
def streamingcomp(A: dace.float32[M, N], B: dace.float32[M, N]):
    # Slightly tricky situation
    tmp = np.ndarray((M, N), dtype=A.dtype)
    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b << B[i, j]
            t >> tmp[i, j]
            t = a + b

    return tmp * B


@dace.program
def streaming_not_composable(A: dace.float32[M, N], B: dace.float32[M, N]):
    for i, j in dace.map[0:M, 0:N - 1]:
        with dace.tasklet:
            a1 << A[i, j + 1]
            a2 << A[i, j]
            b >> B[i, j]
            b = (a1 + a2) / 2
    for i, j in dace.map[0:M, 0:N - 1]:
        with dace.tasklet:
            a1 << B[i, j + 1]
            a2 << B[i, j]
            b >> A[i, j]
            b = (a1 + a2) / 2


@xilinx_test()
def test_streaming_mem():
    # Make SDFG
    sdfg: dace.SDFG = matadd_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    assert sdfg.apply_transformations_repeated(
        sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_multistream():
    # Make SDFG
    sdfg: dace.SDFG = matadd_multistream.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    assert sdfg.apply_transformations_repeated(
        sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 4

    # Ensure only 4 connected components exist
    mainstate = next(s for s in sdfg.nodes() if 'copy' not in s.label)
    assert len(list(nx.weakly_connected_components(mainstate.nx))) == 6

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    D = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C, D=D)

    diff1 = np.linalg.norm(C - (A + B))
    diff2 = np.linalg.norm(D - (A - B))
    assert diff1 <= 1e-5 and diff2 <= 1e-5

    return sdfg


@xilinx_test()
def test_multistream_with_deps():
    # Make SDFG
    sdfg: dace.SDFG = streamingcomp.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    assert sdfg.apply_transformations_repeated(
        sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Ensure only 4 connected components exist
    mainstate = next(s for s in sdfg.nodes() if 'copy' not in s.label)
    assert len(list(nx.weakly_connected_components(mainstate.nx))) == 4

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_streaming_mem_mapnests():
    # Make SDFG
    sdfg: dace.SDFG = matadd_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG, MapExpansion])
    assert sdfg.apply_transformations_repeated(
        sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_streaming_composition_matching():
    sdfg: dace.SDFG = streaming_not_composable.to_sdfg()
    assert sdfg.apply_transformations_repeated(sm.StreamingComposition) == 0
    return []  # SDFG was not compiled, so we can't run HLS on it


@xilinx_test()
def test_streaming_composition():
    # Make SDFG
    sdfg: dace.SDFG = streamingcomp.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    assert sdfg.apply_transformations_repeated(
        sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_streaming_composition_mapnests():
    # Make SDFG
    sdfg: dace.SDFG = streamingcomp.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    # Test 1 - both maps expanded
    test1 = copy.deepcopy(sdfg)
    assert test1.apply_transformations_repeated(MapExpansion) == 2
    assert test1.apply_transformations_repeated(
        sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Test 2 - one only one map expanded
    sdfg.apply_transformations(MapExpansion)
    assert sdfg.apply_transformations_repeated(
        sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_streaming_and_composition():
    # Make SDFG
    sdfg: dace.SDFG = streamingcomp.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(
        sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3
    assert sdfg.apply_transformations_repeated(
        sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_1():
    # Make SDFG
    sdfg: dace.SDFG = vecadd_1_streaming.to_sdfg()
    # Transform

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 2

    # Run verification
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    sdfg(A=A, B=B)

    assert all(B == A + 1)

    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_1_symbolic():
    # Make SDFG
    sdfg: dace.SDFG = vecadd_1_streaming_symbol.to_sdfg()
    # Transform

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 2

    # Run verification
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    sdfg(A=A, B=B, N_s=256)

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

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)
    C = np.random.rand(N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


def mem_buffer_vec_add_types(dace_type0, dace_type1, dace_type2, np_type0, np_type1, np_type2):

    sdfg: dace.SDFG = vecadd_streaming_type(
        dace_type0, dace_type1, dace_type2).to_sdfg()

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = (np.random.rand(N) * 100).astype(np_type0)
    B = (np.random.rand(N) * 100).astype(np_type1)
    C = (np.random.rand(N) * 100).astype(np_type2)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


@pytest.mark.skip(reason="Save time")
# def test_mem_buffer_vec_add_float16():
#     return mem_buffer_vec_add_types(dace.float16, dace.float16, dace.float16, np.float16, np.float16, np.float16)
@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_float32():
    return mem_buffer_vec_add_types(dace.float32, dace.float32, dace.float32, np.float32, np.float32, np.float32)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_float64():
    return mem_buffer_vec_add_types(dace.float64, dace.float64, dace.float64, np.float64, np.float64, np.float64)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_int8():
    return mem_buffer_vec_add_types(dace.int8, dace.int8, dace.int8, np.int8, np.int8, np.int8)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_int16():
    return mem_buffer_vec_add_types(dace.int16, dace.int16, dace.int16, np.int16, np.int16, np.int16)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_int32():
    return mem_buffer_vec_add_types(dace.int32, dace.int32, dace.int32, np.int32, np.int32, np.int32)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_int64():
    return mem_buffer_vec_add_types(dace.int64, dace.int64, dace.int64, np.int64, np.int64, np.int64)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_complex64():
    return mem_buffer_vec_add_types(dace.complex64, dace.complex64, dace.complex64, np.complex64, np.complex64,
                                    np.complex64)


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_complex128():
    return mem_buffer_vec_add_types(dace.complex128, dace.complex128, dace.complex128, np.complex128, np.complex128,
                                    np.complex128)


@pytest.mark.skip(reason="Save time")
# def test_mem_buffer_vec_add_mixed_float():
#     return mem_buffer_vec_add_types(dace.float16, dace.float32, dace.float64, np.float16, np.float32, np.float64)
@pytest.mark.skip(reason="Save time")
def test_mem_buffer_vec_add_mixed_int():
    return mem_buffer_vec_add_types(dace.int16, dace.int32, dace.int64, np.int16, np.int32, np.int64)


@xilinx_test()
def test_mem_buffer_mat_add():
    # Make SDFG
    sdfg: dace.SDFG = matadd_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))

    assert diff <= 1e-5

    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_mat_add_symbol():
    # Make SDFG
    sdfg: dace.SDFG = matadd_streaming_symbol.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C, M_s=256, N_s=512)

    diff = np.linalg.norm(C - (A + B))

    assert diff <= 1e-5

    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_tensor_add():
    # Make SDFG
    sdfg: dace.SDFG = tensoradd_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = np.random.rand(M, N, K).astype(np.float32)
    B = np.random.rand(M, N, K).astype(np.float32)
    C = np.random.rand(M, N, K).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))

    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_mem_buffer_multistream():
    # Make SDFG
    sdfg: dace.SDFG = matadd_multistream.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 4

    mainstate = next(s for s in sdfg.nodes() if 'copy' not in s.label)
    assert len(list(nx.weakly_connected_components(mainstate.nx))) == 12

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    D = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C, D=D)

    diff1 = np.linalg.norm(C - (A + B))
    diff2 = np.linalg.norm(D - (A - B))
    assert diff1 <= 1e-5 and diff2 <= 1e-5

    return sdfg


@xilinx_test()
def test_mem_buffer_multistream_with_deps():
    # Make SDFG
    sdfg: dace.SDFG = streamingcomp.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    mainstate = next(s for s in sdfg.nodes() if 'copy' not in s.label)
    assert len(list(nx.weakly_connected_components(mainstate.nx))) == 8

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_mat_mul():
    # Make SDFG
    sdfg: dace.SDFG = matmul_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 1

    # Run verification
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A @ B))
    assert diff <= 1e-5

    return sdfg


@xilinx_test()
def test_mem_buffer_map_order():
    # Make SDFG
    sdfg: dace.SDFG = maporder_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 3

    # Run verification
    A = np.random.rand(N, N, N).astype(np.float32)
    B = np.random.rand(N, N, N).astype(np.float32)
    C = np.random.rand(N, N, N).astype(np.float32)
    D = np.random.rand(N, N, N).astype(np.float32)
    E = np.random.rand(N, N, N).astype(np.float32)
    F = np.random.rand(N, N, N).astype(np.float32)
    G = np.random.rand(N, N).astype(np.float32)
    G_sol = np.random.rand(N, N).astype(np.float32)

    for i in range(N):
        for j in range(N):
            G_sol[i][j] = A[i, j, 0] + B[i, 0, j] + \
                C[0, i, j] + D[j, i, 0] + E[j, 0, i] + F[0, j, i]

    sdfg(A=A, B=B, C=C, D=D, E=E, F=F, G=G)

    assert allclose(G_sol, G)

    return sdfg


@xilinx_test()
def test_mem_buffer_not_applicable():

    sdfg: dace.SDFG = vecadd_1_streaming.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local,
                                                   "memory_buffering_target_bytes": 65
                                               }]) == 0

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local,
                                                   "memory_buffering_target_bytes": 0
                                               }]) == 0

    sdfg2: dace.SDFG = matadd_streaming_bad_stride.to_sdfg()
    sdfg2.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg2.apply_transformations_repeated(sm.StreamingMemory,
                                                options=[{
                                                    'use_memory_buffering': True,
                                                    "storage": dace.StorageType.FPGA_Local,
                                                }]) == 0

    sdfg3: dace.SDFG = vecadd_1_streaming_non_appl_0.to_sdfg()
    sdfg3.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg3.apply_transformations_repeated(sm.StreamingMemory,
                                                options=[{
                                                    'use_memory_buffering': True,
                                                    "storage": dace.StorageType.FPGA_Local,
                                                }]) == 0

    sdfg4: dace.SDFG = vecadd_1_streaming_non_appl_1.to_sdfg()
    sdfg4.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg4.apply_transformations_repeated(sm.StreamingMemory,
                                                options=[{
                                                    'use_memory_buffering': True,
                                                    "storage": dace.StorageType.FPGA_Local,
                                                }]) == 0

    return []


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_atax():

    A = np.random.rand(M, N).astype(np.float32)
    x = np.random.rand(N).astype(np.float32)

    # Parse SDFG and apply FPGA friendly optimization
    sdfg = atax.to_sdfg(strict=True)
    applied = sdfg.apply_transformations([FPGATransformSDFG])
    assert applied == 1

    fpga_rr_interleave_containers_to_banks(sdfg, num_banks=4)

    # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
    from dace.libraries.blas import Gemv
    Gemv.default_implementation = "FPGA_Accumulate"
    sdfg.expand_library_nodes()
    sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingMemory], [{}, {
        'storage': dace.StorageType.FPGA_Local,
        'use_memory_buffering': True
    }],
        print_report=False)
    assert sm_applied == 5  # 3 inlines and 2 Streaming memories

    sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingMemory], [{}, {
        'storage': dace.StorageType.FPGA_Local,
        'use_memory_buffering': False
    }],
        print_report=False)

    assert sm_applied == 1  # 1 Streaming memories

    # specialize the SDFG (needed by the GEMV expansion)
    sdfg.specialize(dict(M=M, N=N))

    y = sdfg(A=A, x=x)

    # Compute ground truth and Validate result
    y_ref = atax.f(A, x)

    assert np.allclose(y, y_ref)
    return sdfg


@pytest.mark.skip(reason="Save time")
def test_mem_buffer_bicg():

    A = np.random.rand(N, M).astype(np.float32)
    p = np.random.rand(M).astype(np.float32)
    r = np.random.rand(M).astype(np.float32)

    # Parse SDFG and apply FPGA friendly optimization
    sdfg = bicg.to_sdfg(strict=True)
    applied = sdfg.apply_transformations([FPGATransformSDFG])
    assert applied == 1

    fpga_rr_interleave_containers_to_banks(sdfg, num_banks=4)

    # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
    from dace.libraries.blas import Gemv
    Gemv.default_implementation = "FPGA_Accumulate"
    sdfg.expand_library_nodes()
    sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingMemory], [{}, {
        'storage': dace.StorageType.FPGA_Local,
        'use_memory_buffering': True
    }],
        print_report=True)
    assert sm_applied == 7  # 3 inlines and 4 Streaming memories

    sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingMemory], [{}, {
        'storage': dace.StorageType.FPGA_Local,
        'use_memory_buffering': False
    }],
        print_report=True)

    assert sm_applied == 1  # 1 Streaming memories

    # specialize the SDFG (needed by the GEMV expansion)
    sdfg.specialize(dict(M=M, N=N))

    res0, res1 = sdfg(A=A, p=p, r=r)

    # Compute ground truth and Validate result
    res0_ref, res1_ref = bicg.f(A, p, r)

    assert np.allclose(res0_ref, res0)
    assert np.allclose(res1, res1_ref)

    return sdfg


@xilinx_test()
def test_two_maps_legal():

    A = np.random.rand(N).astype(dace.float32.type)
    B = np.random.rand(N).astype(dace.float32.type)
    C = np.random.rand(N).astype(dace.float32.type)
    D = np.random.rand(N).astype(dace.float32.type)
    E = np.random.rand(N).astype(dace.float32.type)

    D_exp = A + B
    E_exp = B + C

    sdfg: dace.SDFG = two_maps_kernel_legal.to_sdfg()

    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                               options=[{
                                                   'use_memory_buffering': True,
                                                   "storage": dace.StorageType.FPGA_Local
                                               }]) == 5

    sdfg(A=A, B=B, C=C, D=D, E=E)

    assert np.allclose(D, D_exp)
    assert np.allclose(E, E_exp)

    return sdfg


@xilinx_test()
def test_two_maps_illegal():

    A = np.random.rand(N).astype(dace.float32.type)
    B = np.random.rand(N).astype(dace.float32.type)
    C = np.random.rand(N).astype(dace.float32.type)
    D = np.random.rand(N).astype(dace.float32.type)
    E = np.random.rand(N).astype(dace.float32.type)
    E_exp = np.copy(E)

    D_exp = A + B
    for i in range(0, 64, 2):
        E_exp[i] = B[i] + C[i]

    sdfg = two_maps_kernel_illegal.to_sdfg()

    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    sdfg.apply_transformations_repeated(sm.StreamingMemory,
                                        options=[{
                                            'use_memory_buffering': True,
                                            "storage": dace.StorageType.FPGA_Local
                                        }]) == 2

    sdfg(A=A, B=B, C=C, D=D, E=E)

    assert np.allclose(D, D_exp)
    assert np.allclose(E, E_exp)

    return sdfg


if __name__ == "__main__":
    test_streaming_mem(None)
    test_streaming_mem_mapnests(None)
    test_multistream(None)
    test_multistream_with_deps(None)
    test_streaming_composition_matching(None)
    test_streaming_composition(None)
    test_streaming_composition_mapnests(None)
    test_streaming_and_composition(None)

    test_mem_buffer_vec_add_1(None)
    test_mem_buffer_vec_add_1_symbolic(None)
    test_mem_buffer_vec_add(None)
    test_mem_buffer_mat_add(None)
    test_mem_buffer_mat_add_symbol(None)
    test_mem_buffer_tensor_add(None)
    test_mem_buffer_multistream(None)
    test_mem_buffer_multistream_with_deps(None)
    test_mem_buffer_mat_mul(None)
    test_mem_buffer_not_applicable(None)
    test_mem_buffer_map_order(None)

    # test_mem_buffer_vec_add_float16(None)
    test_mem_buffer_vec_add_float32(None)
    test_mem_buffer_vec_add_float64(None)
    test_mem_buffer_vec_add_int8(None)
    test_mem_buffer_vec_add_int16(None)
    test_mem_buffer_vec_add_int32(None)
    test_mem_buffer_vec_add_int64(None)
    # test_mem_buffer_vec_add_mixed_float(None)
    test_mem_buffer_vec_add_mixed_int(None)
    test_mem_buffer_vec_add_complex64(None)
    test_mem_buffer_vec_add_complex128(None)

    test_mem_buffer_atax(None)
    test_mem_buffer_bicg(None)

    test_two_maps_legal(None)
    test_two_maps_illegal(None)
