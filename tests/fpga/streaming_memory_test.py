# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the StreamingMemory transformation. """
import copy
import dace
import dace.libraries.blas
import networkx as nx
import numpy as np

from dace.transformation.dataflow import streaming_memory as sm, MapExpansion
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.fpga_testing import xilinx_test

M, N, K = 64, 64, 64


@dace.program
def matadd_streaming(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N]):
    C[:] = A + B


@dace.program
def matadd_multistream(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N], D: dace.float32[M, N]):
    C[:] = A + B
    D[:] = A - B


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
    assert sdfg.apply_transformations_repeated(sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

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
    assert sdfg.apply_transformations_repeated(sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 4

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
    assert sdfg.apply_transformations_repeated(sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

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
    assert sdfg.apply_transformations_repeated(sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3

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
    assert sdfg.apply_transformations_repeated(sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

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
    assert test1.apply_transformations_repeated(sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Test 2 - one only one map expanded
    sdfg.apply_transformations(MapExpansion)
    assert sdfg.apply_transformations_repeated(sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

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

    assert sdfg.apply_transformations_repeated(sm.StreamingMemory, dict(storage=dace.StorageType.FPGA_Local)) == 3
    assert sdfg.apply_transformations_repeated(sm.StreamingComposition, dict(storage=dace.StorageType.FPGA_Local)) == 1

    # Run verification
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)

    C = sdfg(A=A, B=B)

    diff = np.linalg.norm(C - ((A + B) * B)) / (M * N)
    assert diff <= 1e-5

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
