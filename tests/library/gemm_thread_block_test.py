# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Matrix products that a GPU thread block computes inside a kernel (the ThreadBlock implementations of Gemm)."""

import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm
from dace.transformation.dataflow import GatherIntoMatrixProduct, TransposeIntoMatrixProduct

BM, N, K, T, R = 64, 128, 256, 3, 500  # Rows per tile, columns, inner dimension, tiles, rows of the gathered matrix
GPU_Global, GPU_Shared = dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared
TB = dace.ScheduleType.GPU_ThreadBlock


@dace.program
def tiles(
    x: dace.bfloat16[T * BM, K] @ GPU_Global,
    w: dace.bfloat16[N, K] @ GPU_Global,
    y: dace.float32[T * BM, N] @ GPU_Global,
):
    for t in dace.map[0:T] @ dace.ScheduleType.GPU_Device:
        tile = dace.ndarray([BM, N], dtype=dace.bfloat16, storage=GPU_Shared)
        tile[:] = x[t * BM : t * BM + BM, :] @ w.T
        for j in dace.map[0:N] @ TB:  # A thread per column
            for r in range(BM):
                y[t * BM + r, j] = dace.float32(tile[r, j])


@dace.program
def gathered(
    x: dace.bfloat16[R, K] @ GPU_Global,
    idx: dace.int32[T, BM] @ GPU_Global,
    w: dace.bfloat16[N, K] @ GPU_Global,
    y: dace.float32[T * BM, N] @ GPU_Global,
):
    for t in dace.map[0:T] @ dace.ScheduleType.GPU_Device:
        tile = dace.ndarray([BM, N], dtype=dace.bfloat16, storage=GPU_Shared)
        tile[:] = x[idx[t]] @ w.T
        for j in dace.map[0:N] @ TB:
            for r in range(BM):
                y[t * BM + r, j] = dace.float32(tile[r, j])


def products(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (Gemm, dace.libraries.blas.MatMul))]


def run(program, implementation: str, **arrays):
    import cupy

    sdfg = program.to_sdfg()
    sdfg.apply_transformations_repeated([GatherIntoMatrixProduct, TransposeIntoMatrixProduct])
    sdfg.expand_library_nodes(recursive=False)  # MatMul -> Gemm, which selects a thread-block implementation
    (gemm,) = [n for n in products(sdfg) if isinstance(n, Gemm)]
    assert gemm.implementation == "ThreadBlockMMA"
    gemm.implementation = implementation
    args = {name: cupy.asarray(value) for name, value in arrays.items()}
    sdfg(**args)
    return cupy.asnumpy(args["y"])


def random_bf16(*shape):
    """Random multiples of 1/2 that bfloat16 represents exactly: their products, summed in float32, are exact, and the
    tile rounds them to bfloat16 as the reference does."""
    return (np.random.default_rng(len(shape)).integers(-2, 3, shape) / 2).astype(np.float32)


def as_bf16(values: np.ndarray):
    import ml_dtypes

    return values.astype(ml_dtypes.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["ThreadBlockMMA", "ThreadBlock"])
def test_tiles(implementation):
    x, w = random_bf16(T * BM, K), random_bf16(N, K)
    y = run(tiles, implementation, x=as_bf16(x), w=as_bf16(w), y=np.zeros((T * BM, N), np.float32))
    assert np.array_equal(y, as_bf16(x @ w.T).astype(np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["ThreadBlockMMA", "ThreadBlock"])
def test_gathered_rows(implementation):
    x, w = random_bf16(R, K), random_bf16(N, K)
    idx = np.random.default_rng(1).integers(0, R, (T, BM)).astype(np.int32)
    y = run(gathered, implementation, x=as_bf16(x), idx=idx, w=as_bf16(w), y=np.zeros((T * BM, N), np.float32))
    assert np.array_equal(y, as_bf16(x[idx.reshape(-1)] @ w.T).astype(np.float32))


if __name__ == "__main__":
    for implementation in ("ThreadBlockMMA", "ThreadBlock"):
        test_tiles(implementation)
        test_gathered_rows(implementation)
