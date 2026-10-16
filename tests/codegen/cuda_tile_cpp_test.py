# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.cuda.tile_backend: cuda_tile`` writes a GPU map over BLOCK tile nodes as a CUDA Tile C++
``__tile_global__`` kernel, one tile block per map iteration; the build takes ``--enable-tile`` only when it holds
one."""

import contextlib
import pathlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CodegenError
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

pytestmark = [pytest.mark.gpu, pytest.mark.cuda_tile, pytest.mark.tile]

M, N, K, L = (dace.symbol(name) for name in "MNKL")


@contextlib.contextmanager
def cuda_tile_backend() -> Iterator[None]:
    with (
        dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"),
        dace.config.set_temporary("compiler", "cuda", "tile_backend", value="cuda_tile"),
    ):
        yield


def device_source(sdfg: dace.SDFG) -> str:
    return next(pathlib.Path(sdfg.build_folder, "src", "experimental_cuda").glob("*.cu")).read_text()


def builds_with_tile_flags(sdfg: dace.SDFG) -> bool:
    return "CudaTileCppFlags" in pathlib.Path(sdfg.build_folder, "dace_environments.csv").read_text()


@dace.program
def scale_add(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    for i in dace.map[0:N]:
        z[i] = x[i] * 2.0 + y[i]


def test_a_vectorized_map_with_a_ragged_tail_is_one_tile_kernel():
    import cupy

    sdfg = scale_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,))).apply_pass(sdfg, {})
    sdfg.name = "cuda_tile_scale_add"
    x, y, z = cupy.arange(1000, dtype=np.float32), cupy.ones(1000, np.float32), cupy.zeros(1000, np.float32)

    with cuda_tile_backend():
        sdfg(x=x, y=y, z=z, N=1000)

    assert np.array_equal(z.get(), 2 * np.arange(1000, dtype=np.float32) + 1)
    source = device_source(sdfg)
    assert "__tile_global__ void" in source and "dim3(int_ceil(N, 512), 1, 1), dim3(1, 1, 1)" in source
    assert "ct::load_masked(" in source and builds_with_tile_flags(sdfg)


@dace.program
def blocked_gemm(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    for i, j in dace.map[0:M:16, 0:N:16]:
        acc = dace.define_local([16, 16], dace.float32, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, 16):
            dace.tile.mma(A[i : i + 16, k : k + 16], B[k : k + 16, j : j + 16], acc)
        dace.tile.store(C[i : i + 16, j : j + 16], acc)


def test_a_blocked_gemm_accumulates_tile_products():
    sdfg = blocked_gemm.to_sdfg()
    sdfg.name = "cuda_tile_gemm"
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(0)
    a, b, c = rng.random((64, 48), np.float32), rng.random((48, 32), np.float32), np.zeros((64, 32), np.float32)

    with cuda_tile_backend():
        sdfg(A=a, B=b, C=c, M=64, N=32, K=48)

    np.testing.assert_allclose(c, a @ b, rtol=1e-3, atol=1e-3)
    assert "ct::matmul(A_tile, B_tile)" in device_source(sdfg)


@dace.program
def fused_rows(A: dace.float64[L, 64], B: dace.float64[L, 64], D: dace.float64[L, 64], S: dace.float64[L]):
    for r in dace.map[0:L]:
        row = dace.tile.fma(A[r, :], B[r, :], dace.tile.exp(A[r, :]))
        dace.tile.store(D[r, :], row)
        S[r : r + 1] = dace.tile.sum(row)


def test_a_row_sum_lands_as_a_one_lane_tile_store():
    sdfg = fused_rows.to_sdfg()
    sdfg.name = "cuda_tile_rows"
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(1)
    a, b, d, s = rng.random((4, 64)), rng.random((4, 64)), np.zeros((4, 64)), np.zeros(4)

    with cuda_tile_backend():
        sdfg(A=a, B=b, D=d, S=s, L=4)

    expected = a * b + np.exp(a)
    np.testing.assert_allclose(d, expected, rtol=1e-12)
    np.testing.assert_allclose(s, expected.sum(axis=1), rtol=1e-12)
    assert "ct::broadcast<ct::shape<1>>(sum)" in device_source(sdfg)


@dace.program
def total(A: dace.float64[L], S: dace.float64[1]):
    for i in dace.map[0:L:64]:
        S[0:1] += dace.tile.sum(A[i : i + 64])


def test_a_sum_every_block_accumulates_adds_atomically():
    sdfg = total.to_sdfg()
    sdfg.name = "cuda_tile_total"
    sdfg.apply_gpu_transformations()
    a, s = np.random.default_rng(3).random(64 * 6), np.zeros(1)

    with cuda_tile_backend():
        sdfg(A=a, S=s, L=a.size)

    np.testing.assert_allclose(s, [a.sum()], rtol=1e-12)
    assert "ct::atomic_add<ct::memory_order::relaxed>(" in device_source(sdfg)


@dace.program
def plain_scale(x: dace.float64[N]):
    for i in dace.map[0:N]:
        x[i] = 2 * x[i]


# The GPU stream wiring warns about any plain offloaded kernel, on the default tile backend too; not tile code's warning
@pytest.mark.filterwarnings('ignore:WARNING. Use of uninitialized transient "gpu_streams":UserWarning')
def test_a_build_without_tile_kernels_takes_no_tile_flags():
    sdfg = plain_scale.to_sdfg()
    sdfg.name = "cuda_tile_plain"
    offload_to_gpu(sdfg)
    import cupy

    x = cupy.arange(8.0)

    with cuda_tile_backend():
        sdfg(x=x, N=8)

    assert np.array_equal(x.get(), 2 * np.arange(8.0))
    assert "__tile_global__" not in device_source(sdfg) and not builds_with_tile_flags(sdfg)


def test_a_cpp_tasklet_in_a_tile_kernel_is_refused():
    sdfg = scale_add.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(4,))).apply_pass(sdfg, {})
    entry, state = next(
        (n, s)
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device
    )
    tasklet = state.add_tasklet("cpp_lanes", {}, {}, "constexpr int lanes = 4;", language=dace.Language.CPP)
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_nedge(tasklet, state.exit_node(entry), dace.Memlet())

    with cuda_tile_backend(), pytest.raises(CodegenError, match="tasklet 'cpp_lanes' is CPP"):
        sdfg.generate_code()
