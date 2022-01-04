# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.subsets import Range

M, N, K = 24, 25, 26


@dace.program
def add(A: dace.float64[M, N]):
    return A + A


@dace.program
def addunk(A):
    return A + A


@dace.program
def gemm(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N], alpha: dace.float32, beta: dace.float32):
    C[:] = alpha * A @ B + beta * C


def test_add():
    A = np.random.rand(M, N)
    sdfg = add.to_sdfg()
    result = sdfg(A=A)

    # Check validity of result
    assert np.allclose(result, A + A)

    # Check map sequence
    me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    assert me.map.range == Range([(0, 23, 1), (0, 24, 1)])


def test_add_11dim():
    A = np.random.rand(*(2 if i < 9 else 3 for i in range(11)))
    sdfg = addunk.to_sdfg(A)
    result = sdfg(A=A)

    # Check validity of result
    assert np.allclose(result, A + A)

    # Check map sequence
    me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    assert me.map.range == Range([(0, 1, 1) if i < 9 else (0, 2, 1) for i in range(11)])


def test_gemm():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    origC = np.zeros([M, N], dtype=np.float32)
    origC[:] = C
    gemm(A, B, C, 1.0, 1.0)

    realC = 1.0 * (A @ B) + 1.0 * origC
    diff = np.linalg.norm(C - realC) / (M * N)
    print('Difference:', diff)
    assert diff < 1e-5


if __name__ == '__main__':
    test_add()
    test_add_11dim()
    test_gemm()
