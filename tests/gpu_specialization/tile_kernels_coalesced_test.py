# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TileKernelsCoalesced: an explicit thread block whose lanes walk the coalesced axis, each thread several points of
an outer axis. The structural checks need no GPU; the numeric one runs on the device at sizes nothing divides."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.tile_kernels_coalesced import TileKernelsCoalesced

M, N = (dace.symbol(s) for s in "MN")


@dace.program
def scaled_rows(w: dace.float64[N], A: dace.float64[M, N], B: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = w[j] * A[i, j]


@dace.program
def scaled_columns_transposed(w: dace.float64[M], A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[0:M, 0:N]:
        B[j, i] = w[i] * A[j, i]


@dace.program
def scaled(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = 2.0 * A[i]


def offloaded(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    return sdfg


def schedules(sdfg: dace.SDFG) -> dict:
    entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    return {entry.map.schedule: entry for entry in entries}


@pytest.mark.parametrize("program, coalesced, outer", [(scaled_rows, "j", "i"), (scaled_columns_transposed, "i", "j")])
def test_the_coalesced_axis_is_the_lanes_and_the_outer_axis_the_thread_loop(program, coalesced, outer):
    """The parameter indexing the unit-stride dimension becomes threadIdx.x, whatever its position in the map."""
    sdfg = offloaded(program, f"tile_coalesced_{program.name}")
    (label, result), *_ = TileKernelsCoalesced(factor=4).apply_pass(sdfg, {}).items()
    assert result == (coalesced, outer, 4), result
    maps = schedules(sdfg)
    grid, block, thread = (
        maps[dtypes.ScheduleType.GPU_Device],
        maps[dtypes.ScheduleType.GPU_ThreadBlock],
        maps[dtypes.ScheduleType.Sequential],
    )
    # The domain-matched block the offload chose for a 2-D map: 256 threads, a 64-lane wavefront on the coalesced axis.
    assert list(grid.map.gpu_block_size) == [64, 4, 1]
    assert block.map.params[-1].endswith(coalesced) and block.map.params[-2].endswith(outer), block.map.params
    assert thread.map.params == [outer, coalesced] and str(thread.map.range.size()[0]) != "1"
    sdfg.validate()


def test_a_one_dimensional_kernel_keeps_its_selected_block():
    """Re-blocking after the selection would conflict with what was sized from it (a tree reduction's
    block-reduce, a warp tile promoted later)."""
    sdfg = offloaded(scaled, "tile_coalesced_1d")
    (kernel,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    block = list(kernel.map.gpu_block_size)
    assert TileKernelsCoalesced(factor=4).apply_pass(sdfg, {}) is None
    assert list(kernel.map.gpu_block_size) == block


@dace.program
def last_row_zeroed(B: dace.float64[N + 1, M, M]):
    for i, j, k in dace.map[N : N + 1, 0:M, 0:M]:
        B[i, j, k] = 0.0


def test_an_axis_narrower_than_its_tile_is_tiled_by_its_extent():
    """One row tiled by rows of block rows reaches past the array in the memlets around the grid (cp2k trs4's sink)."""
    # Not canonicalized, which would drop the one-iteration axis: a pass after it narrows a map to one row
    sdfg = last_row_zeroed.to_sdfg(simplify=True)
    sdfg.name = "tile_coalesced_one_row"
    offload_to_gpu(sdfg)
    TileKernelsCoalesced(factor=4).apply_pass(sdfg, {})
    sdfg.validate()
    # One row: no block rows over it, no thread loop along it
    grid = schedules(sdfg)[dtypes.ScheduleType.GPU_Device]
    assert list(grid.map.gpu_block_size)[1:] == [1, 1]


@pytest.mark.gpu
@pytest.mark.parametrize("program", [scaled_rows, scaled_columns_transposed])
def test_tiled_kernels_compute_what_numpy_does(program):
    import cupy

    sdfg = offloaded(program, f"tile_coalesced_run_{program.name}")
    TileKernelsCoalesced(factor=3).apply_pass(sdfg, {})
    finalize_for_target(sdfg, "gpu")
    m, n = 37, 101
    rows = program is scaled_rows
    w = np.random.rand(n if rows else m)
    A = np.random.rand(m, n) if rows else np.random.rand(n, m)
    # The offload moved the arguments to the device.
    B = cupy.zeros(A.shape)
    sdfg(w=cupy.asarray(w), A=cupy.asarray(A), B=B, M=m, N=n)
    assert np.allclose(B.get(), w[None, :] * A)


if __name__ == "__main__":
    test_the_coalesced_axis_is_the_lanes_and_the_outer_axis_the_thread_loop(scaled_rows, "j", "i")
    test_the_coalesced_axis_is_the_lanes_and_the_outer_axis_the_thread_loop(scaled_columns_transposed, "i", "j")
    test_a_one_dimensional_kernel_keeps_its_selected_block()
    test_tiled_kernels_compute_what_numpy_does(scaled_rows)
    test_tiled_kernels_compute_what_numpy_does(scaled_columns_transposed)
