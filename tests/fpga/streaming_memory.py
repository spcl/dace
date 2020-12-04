# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the StreamingMemory transformation. """
import dace
import dace.libraries.blas
import numpy as np

from dace.transformation.dataflow import streaming_memory as sm, MapExpansion
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

M, N, K = 64, 64, 64


@dace.program
def matmul_streaming(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            c = a * b
            c >> C(1, lambda x, y: x + y)[i, j]


def test_matmul():
    # Make SDFG
    sdfg: dace.SDFG = matmul_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.apply_transformations_repeated(sm.StreamingMemory)

    # Run verification
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - A@B) / (M*N)
    print('Difference:', diff)
    assert diff <= 1e-5


def test_matmul_mapnests():
    # Make SDFG
    sdfg: dace.SDFG = matmul_streaming.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG, MapExpansion])
    sdfg.apply_transformations_repeated(sm.StreamingMemory)

    # Run verification
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - A @ B) / (M * N)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test_matmul()
    test_matmul_mapnests()