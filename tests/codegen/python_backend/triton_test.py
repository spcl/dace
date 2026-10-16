# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU map over BLOCK tile nodes runs as a Triton kernel: one program per map iteration, one ``tl`` call per node."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.triton,
    # Triton's own code generator builds an ast.AnnAssign Python 3.14 deprecates; not this backend's code
    pytest.mark.filterwarnings("ignore:AnnAssign.__init__ missing:DeprecationWarning"),
]

M, N, K, L = (dace.symbol(name) for name in "MNKL")


@contextlib.contextmanager
def triton_backend() -> Iterator[None]:
    with (
        dace.config.set_temporary("compiler", "backend", value="python"),
        dace.config.set_temporary("compiler", "python", "tile_dialect", value="triton"),
    ):
        yield


def kernel_source(compiled) -> str:
    """The Triton kernels of a compiled program, without the prelude."""
    source = compiled.path.read_text()
    return source[source.index("@triton.jit\ndef kernel_") :]


@dace.program
def scale_add(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    for i in dace.map[0:N]:
        z[i] = x[i] * 2.0 + y[i]


def test_a_vectorized_map_with_a_ragged_tail_runs_as_one_masked_kernel():
    import cupy

    sdfg = scale_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,))).apply_pass(sdfg, {})
    x, y, z = (cupy.arange(1000, dtype=np.float32), cupy.ones(1000, np.float32), cupy.zeros(1000, np.float32))

    with triton_backend():
        compiled = sdfg.compile()
        compiled(x=x, y=y, z=z, N=1000)

    assert np.array_equal(z.get(), 2 * np.arange(1000, dtype=np.float32) + 1)
    source = kernel_source(compiled)
    assert "tl.program_id(0) * 512" in source and "& _tile_iter_mask" in source
    assert "[(int_ceil(N, 512),)]" in compiled.path.read_text()


BM, BN, BK = 16, 16, 16


@dace.program
def blocked_gemm(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    for i, j in dace.map[0:M:BM, 0:N:BN]:
        acc = dace.define_local([BM, BN], dace.float32, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, BK):
            dace.tile.mma(A[i : i + BM, k : k + BK], B[k : k + BK, j : j + BN], acc)
        dace.tile.store(C[i : i + BM, j : j + BN], acc)


def test_a_blocked_gemm_accumulates_tl_dot_in_a_loop():
    sdfg = blocked_gemm.to_sdfg()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(0)
    a, b, c = rng.random((64, 48), np.float32), rng.random((48, 32), np.float32), np.zeros((64, 32), np.float32)

    with triton_backend():
        compiled = sdfg.compile()
        compiled(A=a, B=b, C=c, M=64, N=32, K=48)

    np.testing.assert_allclose(c, a @ b, rtol=1e-3, atol=1e-3)
    source = kernel_source(compiled)
    assert "tl.full([16, 16], (0.0), tl.float32)" in source and "tl.dot(A_tile, B_tile)" in source


@dace.program
def fused_rows(A: dace.float64[L, 64], B: dace.float64[L, 64], D: dace.float64[L, 64], S: dace.float64[L]):
    for r in dace.map[0:L]:
        row = dace.tile.fma(A[r, :], B[r, :], dace.tile.exp(A[r, :]))
        dace.tile.store(D[r, :], row)
        S[r : r + 1] = dace.tile.sum(row)


def test_elementwise_tiles_and_a_row_sum_store_one_element():
    sdfg = fused_rows.to_sdfg()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(1)
    a, b, d, s = rng.random((4, 64)), rng.random((4, 64)), np.zeros((4, 64)), np.zeros(4)

    with triton_backend():
        compiled = sdfg.compile()
        compiled(A=a, B=b, D=d, S=s, L=4)

    expected = a * b + np.exp(a)
    np.testing.assert_allclose(d, expected, rtol=1e-12)
    np.testing.assert_allclose(s, expected.sum(axis=1), rtol=1e-12)
    assert "tl.store(S_gpu + (r + 0) * 1, sum)" in kernel_source(compiled)


W = 256


@dace.program
def row_reductions(A: dace.float64[L, W], R: dace.float64[L, 4], C: dace.float64[L, W], X: dace.float64[L, W]):
    for r in dace.map[0:L]:
        R[r, 0:1] = dace.tile.sum(A[r, :])
        R[r, 1:2] = dace.tile.prod(A[r, :])
        R[r, 2:3] = dace.tile.max(A[r, :])
        R[r, 3:4] = dace.tile.min(A[r, :])
        dace.tile.store(C[r, :], dace.tile.cumsum(A[r, :]))
        dace.tile.store(X[r, :], dace.tile.cummax(A[r, :]))


def test_reductions_and_scans_match_numpy():
    sdfg = row_reductions.to_sdfg()
    sdfg.apply_gpu_transformations()
    a = 0.5 + np.random.default_rng(2).random((4, W))
    r, c, x = np.zeros((4, 4)), np.zeros((4, W)), np.zeros((4, W))

    with triton_backend():
        sdfg(A=a, R=r, C=c, X=x, L=4)

    np.testing.assert_allclose(r[:, 0], a.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(r[:, 1], a.prod(axis=1), rtol=1e-12)
    np.testing.assert_array_equal(r[:, 2], a.max(axis=1))
    np.testing.assert_array_equal(r[:, 3], a.min(axis=1))
    np.testing.assert_allclose(c, np.cumsum(a, axis=1), rtol=1e-12)
    np.testing.assert_array_equal(x, np.maximum.accumulate(a, axis=1))


@dace.program
def total(A: dace.float64[L], S: dace.float64[1]):
    for i in dace.map[0:L:64]:
        S[0:1] += dace.tile.sum(A[i : i + 64])


def test_a_sum_accumulated_by_every_program_adds_atomically():
    sdfg = total.to_sdfg()
    sdfg.apply_gpu_transformations()
    a, s = np.random.default_rng(3).random(64 * 6), np.zeros(1)

    with triton_backend():
        compiled = sdfg.compile()
        compiled(A=a, S=s, L=a.size)

    np.testing.assert_allclose(s, [a.sum()], rtol=1e-12)
    assert "tl.atomic_add(S_gpu + " in kernel_source(compiled)
