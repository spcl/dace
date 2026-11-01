# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU maps that run in strides of the thread block or of the grid (``Map.gpu_strided``)."""

import numpy as np
import pytest

import dace

N = dace.symbol("N")
TB = dace.ScheduleType.GPU_ThreadBlock
WIDTH = 1000  # Wider than any thread block, and not a multiple of one


def _to_gpu(sdfg: dace.SDFG, kernel_params: list[str], block_size: list[int] | None = None):
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.params == kernel_params:
            node.map.schedule = dace.ScheduleType.GPU_Device
            node.map.gpu_block_size = block_size


def _mark_strided(sdfg: dace.SDFG, params: list[str]):
    marked = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.params == params:
            node.map.gpu_strided = True
            marked += 1
    assert marked > 0


def _code(sdfg: dace.SDFG) -> str:
    return next(c for c in sdfg.generate_code() if c.language == "cu").clean_code


@dace.program
def row_norms(A: dace.float64[N, WIDTH], scaled: dace.float64[N, WIDTH]):
    for i in dace.map[0:N]:
        total = dace.ndarray([1], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        total[0] = 0
        for j in dace.map[0:WIDTH] @ TB:
            total[0] += A[i, j] * A[i, j]
        for k in dace.map[0:WIDTH] @ TB:
            scaled[i, k] = A[i, k] / total[0]


@pytest.mark.gpu
def test_block_strided_maps():
    """Thread-block maps wider than the thread block, which the kernel sets."""
    sdfg = row_norms.to_sdfg(simplify=True)
    _to_gpu(sdfg, ["i"], [64, 1, 1])
    _mark_strided(sdfg, ["j"])
    _mark_strided(sdfg, ["k"])
    code = _code(sdfg)
    assert "dim3(64, 1, 1)" in code  # The thread block of the kernel, not the width of the maps
    assert "total_acc[" in code  # Each thread accumulates in a register...
    assert "warp_combine(" in code  # ...and the threads of a warp combine their updates of the total

    import cupy as cp  # Only GPU tests use CuPy

    A = np.random.default_rng(0).random((20, WIDTH))
    scaled = cp.zeros((20, WIDTH))
    sdfg(A=cp.asarray(A), scaled=scaled, N=20)
    assert np.allclose(cp.asnumpy(scaled), A / (A * A).sum(axis=1, keepdims=True))


@dace.program
def prefix_rows(A: dace.float64[N, WIDTH], counts: dace.int32[N], out: dace.float64[N, WIDTH]):
    for i in dace.map[0:N]:
        for t in dace.map[0:32] @ TB:  # Sets the thread block
            out[i, t] = -1
        n = counts[i]
        for j in dace.map[0:n] @ TB:  # A range that depends on data
            out[i, j] = A[i, j] * 2


@pytest.mark.gpu
def test_block_strided_dynamic_range():
    """A thread-block map whose range is read from data, strided by the thread block of another map."""
    sdfg = prefix_rows.to_sdfg(simplify=True)
    _to_gpu(sdfg, ["i"])
    _mark_strided(sdfg, ["j"])
    assert "dim3(32, 1, 1)" in _code(sdfg)

    import cupy as cp  # Only GPU tests use CuPy

    rows = 16
    A = np.random.default_rng(1).random((rows, WIDTH))
    counts = np.random.default_rng(2).integers(0, WIDTH, rows).astype(np.int32)
    out = cp.zeros((rows, WIDTH))
    sdfg(A=cp.asarray(A), counts=cp.asarray(counts), out=out, N=rows)
    expected = np.zeros((rows, WIDTH))
    for r in range(rows):
        expected[r, :32] = -1
        expected[r, : counts[r]] = A[r, : counts[r]] * 2
    assert np.allclose(cp.asnumpy(out), expected)


@dace.program
def two_norms(A: dace.float64[N, WIDTH], B: dace.float64[N, WIDTH], norms: dace.float64[N, 2]):
    for i in dace.map[0:N]:
        sums = dace.ndarray([2], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        sums[0] = 0
        sums[1] = 0
        for j in dace.map[0:WIDTH] @ TB:
            a = A[i, j]
            b = B[i, j]
            sums[0] += a * a
            sums[1] += b * b
        norms[i, 0] = sums[0]
        norms[i, 1] = sums[1]


@pytest.mark.gpu
def test_block_strided_accumulates_per_thread():
    """Updates of the same locations in every iteration accumulate in registers, and update memory once per thread:
    both locations in one branch of warp-combined updates."""
    sdfg = two_norms.to_sdfg(simplify=True)
    _to_gpu(sdfg, ["i"], [64, 1, 1])
    _mark_strided(sdfg, ["j"])
    code = _code(sdfg)
    assert "sums_acc[" in code
    assert ") & dace::warp_same_location(" in code
    assert code.count("reduce_atomic(") == 4  # Both locations, combined or not: none in the loop

    import cupy as cp  # Only GPU tests use CuPy

    rng = np.random.default_rng(5)
    A, B = rng.random((20, WIDTH)), rng.random((20, WIDTH))
    norms = cp.zeros((20, 2))
    sdfg(A=cp.asarray(A), B=cp.asarray(B), norms=norms, N=20)
    assert np.allclose(cp.asnumpy(norms), np.stack([(A * A).sum(axis=1), (B * B).sum(axis=1)], axis=1))


@dace.program
def strided_bins(A: dace.float64[N, WIDTH], bins: dace.float64[N, 4]):
    for i in dace.map[0:N]:
        for j in dace.map[0:WIDTH] @ TB:
            bins[i, j % 4] += A[i, j]


@pytest.mark.gpu
def test_block_strided_moving_updates():
    """Updates of a location that changes with the iteration stay updates of memory."""
    sdfg = strided_bins.to_sdfg(simplify=True)
    _to_gpu(sdfg, ["i"], [64, 1, 1])
    _mark_strided(sdfg, ["j"])
    assert "_acc[" not in _code(sdfg)

    import cupy as cp  # Only GPU tests use CuPy

    A = np.random.default_rng(6).random((20, WIDTH))
    bins = cp.zeros((20, 4))
    sdfg(A=cp.asarray(A), bins=bins, N=20)
    assert np.allclose(cp.asnumpy(bins), A.reshape(20, WIDTH // 4, 4).sum(axis=1))


@dace.program
def axpy(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = 2 * x[i] + y[i]


@pytest.mark.gpu
def test_grid_strided_map():
    """A device map over a grid of as many blocks as fit on the device, launched as any kernel (not cooperatively:
    it does not synchronize its grid)."""
    sdfg = axpy.to_sdfg(simplify=True)
    _to_gpu(sdfg, ["i"], [128, 1, 1])
    _mark_strided(sdfg, ["i"])
    code = _code(sdfg)
    assert "OccupancyMaxActiveBlocksPerMultiprocessor" in code
    assert "cudaLaunchKernel(" in code and "LaunchCooperativeKernel" not in code

    import cupy as cp  # Only GPU tests use CuPy

    n = 1 << 22  # Many more iterations than threads
    x = np.random.default_rng(3).random(n)
    y = np.random.default_rng(4).random(n)
    y_gpu = cp.asarray(y)
    sdfg(x=cp.asarray(x), y=y_gpu, N=n)
    assert np.allclose(cp.asnumpy(y_gpu), 2 * x + y)


def test_strided_map_outside_gpu_raises():
    sdfg = axpy.to_sdfg(simplify=True)
    _mark_strided(sdfg, ["i"])  # A CPU map
    with pytest.raises(ValueError, match="neither a thread-block nor a device map"):
        dace.transformation.passes.gpu_strided_maps.LowerStridedGPUMaps().apply_pass(sdfg, {})


if __name__ == "__main__":
    test_block_strided_maps()
    test_block_strided_dynamic_range()
    test_block_strided_accumulates_per_thread()
    test_block_strided_moving_updates()
    test_grid_strided_map()
    test_strided_map_outside_gpu_raises()
