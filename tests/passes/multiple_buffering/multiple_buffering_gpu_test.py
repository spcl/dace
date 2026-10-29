# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Multiple buffering of shared-memory tiles in GPU kernels."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")
BLOCK, TILE = 64, 16
BM, BN, BK = 16, 16, 8


@dace.program
def tiled_add(A: dace.float64[N] @ dace.StorageType.GPU_Global, B: dace.float64[N] @ dace.StorageType.GPU_Global):
    for blk in dace.map[0:N:BLOCK] @ dace.ScheduleType.GPU_Device:
        buf = dace.define_local([TILE], dace.float64, storage=dace.StorageType.GPU_Shared)
        for t in range(0, BLOCK, TILE):
            buf[:] = A[blk + t : blk + t + TILE]
            for j in dace.map[0:TILE] @ dace.ScheduleType.GPU_ThreadBlock:
                B[blk + t + j] = 2 * buf[j] + 1


@dace.program
def tiled_gemm(
    A: dace.float64[M, K] @ dace.StorageType.GPU_Global,
    B: dace.float64[K, N] @ dace.StorageType.GPU_Global,
    C: dace.float64[M, N] @ dace.StorageType.GPU_Global,
):
    for bi, bj in dace.map[0:M:BM, 0:N:BN] @ dace.ScheduleType.GPU_Device:
        a_tile = dace.define_local([BM, BK], dace.float64, storage=dace.StorageType.GPU_Shared)
        b_tile = dace.define_local([BK, BN], dace.float64, storage=dace.StorageType.GPU_Shared)
        for kk in range(0, K, BK):
            a_tile[:] = A[bi : bi + BM, kk : kk + BK]
            b_tile[:] = B[kk : kk + BK, bj : bj + BN]
            for i, j in dace.map[0:BM, 0:BN] @ dace.ScheduleType.GPU_ThreadBlock:
                for l in range(BK):
                    C[bi + i, bj + j] += a_tile[i, l] * b_tile[l, j]


@dace.program
def blocked_gemm(
    A: dace.float64[M, K] @ dace.StorageType.GPU_Global,
    B: dace.float64[K, N] @ dace.StorageType.GPU_Global,
    C: dace.float64[M, N] @ dace.StorageType.GPU_Global,
    D: dace.float64[M, N] @ dace.StorageType.GPU_Global,
):
    """``C = A @ B + D`` tiled in M, N and K: a kernel over M tiles, a loop over N tiles that stages a tile of ``D``,
    and an innermost loop over K tiles."""
    for bi in dace.map[0:M:BM] @ dace.ScheduleType.GPU_Device:
        a_tile = dace.define_local([BM, BK], dace.float64, storage=dace.StorageType.GPU_Shared)
        b_tile = dace.define_local([BK, BN], dace.float64, storage=dace.StorageType.GPU_Shared)
        d_tile = dace.define_local([BM, BN], dace.float64, storage=dace.StorageType.GPU_Shared)
        for bj in range(0, N, BN):
            d_tile[:] = D[bi : bi + BM, bj : bj + BN]
            for kk in range(0, K, BK):
                a_tile[:] = A[bi : bi + BM, kk : kk + BK]
                b_tile[:] = B[kk : kk + BK, bj : bj + BN]
                for i, j in dace.map[0:BM, 0:BN] @ dace.ScheduleType.GPU_ThreadBlock:
                    for l in range(BK):
                        C[bi + i, bj + j] += a_tile[i, l] * b_tile[l, j]
            for i, j in dace.map[0:BM, 0:BN] @ dace.ScheduleType.GPU_ThreadBlock:
                C[bi + i, bj + j] += d_tile[i, j]


def pipelined(program, depth: int, n_depth: int = 0) -> dace.SDFG:
    """``program`` with ``pipeline_depth`` set on its loop over tiles (``t`` or ``kk``), and ``n_depth`` on its loop
    over N tiles (``bj``) if given."""
    sdfg = program.to_sdfg(simplify=True)
    loops = {
        region.loop_variable: region
        for region in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(region, LoopRegion)
    }
    tile_loops = [loops[v] for v in ("t", "kk") if v in loops]
    assert len(tile_loops) == 1
    tile_loops[0].pipeline_depth = depth
    if n_depth:
        loops["bj"].pipeline_depth = n_depth
    return sdfg


def test_shared_tiles_code_structure():
    """The kernel holds a ring of 3 tiles in shared memory, and the main loop consumes before it produces, so the
    barrier after each copy also guards the slot the next copy overwrites. Generating the code needs no GPU."""
    sdfg = pipelined(tiled_add, 3)
    (code,) = [obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu"]
    assert f"__shared__ double buf[{3 * TILE}];" in code
    main = code[code.index("for (t = 0;") :]
    main = main[: main.index("for (t = ", 1)]  # up to the epilogue
    assert main.index("% 3)") < main.index("GlobalToShared")


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
def test_tiled_add(depth: int):
    import cupy as cp

    sdfg = pipelined(tiled_add, depth)
    n = 4 * BLOCK
    A = cp.random.rand(n)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    assert np.allclose(B.get(), 2 * A.get() + 1)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("k", [BK, 2 * BK, 7 * BK])
def test_tiled_gemm(depth: int, k: int):
    import cupy as cp

    sdfg = pipelined(tiled_gemm, depth)
    m, n = 2 * BM, 3 * BN
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    assert np.allclose(C.get(), A.get() @ B.get())


@pytest.mark.parametrize("n_depth", [2, 3])
def test_blocked_gemm_code_structure(n_depth: int):
    """Both the loop over K tiles and the loop over N tiles get rings of shared tiles. Needs no GPU."""
    sdfg = pipelined(blocked_gemm, 2, n_depth)
    (code,) = [obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu"]
    assert f"__shared__ double a_tile[{2 * BM * BK}];" in code
    assert f"__shared__ double b_tile[{2 * BK * BN}];" in code
    assert f"__shared__ double d_tile[{n_depth * BM * BN}];" in code


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("k", [BK, 2 * BK, 7 * BK])
def test_blocked_gemm_k_tiles(depth: int, k: int):
    """GEMM tiled in M, N and K with only the loop over K tiles pipelined; the N tiles are a sequential loop."""
    import cupy as cp

    sdfg = pipelined(blocked_gemm, depth)
    m, n = 2 * BM, 3 * BN
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    D = cp.random.rand(m, n)
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n, K=k)
    assert np.allclose(C.get(), A.get() @ B.get() + D.get())


@pytest.mark.gpu
@pytest.mark.parametrize("n_depth", [2, 3])
@pytest.mark.parametrize("n", [BN, 2 * BN, 5 * BN])
def test_blocked_gemm_pipelined_n_tiles(n_depth: int, n: int):
    """Both the loop over K tiles and the loop over N tiles (staging a tile of ``D``) are pipelined."""
    import cupy as cp

    sdfg = pipelined(blocked_gemm, 2, n_depth)
    m, k = 2 * BM, 4 * BK
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    D = cp.random.rand(m, n)
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n, K=k)
    assert np.allclose(C.get(), A.get() @ B.get() + D.get())


@pytest.mark.parametrize("depth", [4, 5, 8])
def test_shared_tiles_depth_beyond_trip_count_code_structure(depth: int):
    """Depths up to and beyond the BLOCK / TILE = 4 iterations of the loop still allocate a ring of ``depth`` tiles."""
    sdfg = pipelined(tiled_add, depth)
    (code,) = [obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu"]
    assert f"__shared__ double buf[{depth * TILE}];" in code
    assert f"% {depth})" in code


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 5, 8])
def test_tiled_add_depth_beyond_trip_count(depth: int):
    """The prologue wants more tiles than the loop has, so it is clamped and the main loop never runs."""
    import cupy as cp

    sdfg = pipelined(tiled_add, depth)
    n = 3 * BLOCK
    A = cp.random.rand(n)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    assert np.allclose(B.get(), 2 * A.get() + 1)


@pytest.mark.gpu
def test_tiled_add_single_block():
    import cupy as cp

    sdfg = pipelined(tiled_add, 2)
    A = cp.random.rand(BLOCK)
    B = cp.zeros(BLOCK)
    sdfg(A=A, B=B, N=BLOCK)
    assert np.allclose(B.get(), 2 * A.get() + 1)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 5])
def test_tiled_gemm_depth_beyond_trip_count(depth: int):
    """A single K tile and depths beyond the number of K tiles."""
    import cupy as cp

    sdfg = pipelined(tiled_gemm, depth)
    m, n, k = 2 * BM, 2 * BN, 2 * BK
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    assert np.allclose(C.get(), A.get() @ B.get())


@pytest.mark.gpu
@pytest.mark.parametrize("n_depth", [4, 6])
def test_blocked_gemm_n_depth_beyond_trip_count(n_depth: int):
    """The pipelined N loop has 2 iterations but a depth of 4 or 6."""
    import cupy as cp

    sdfg = pipelined(blocked_gemm, 3, n_depth)
    m, n, k = 2 * BM, 2 * BN, 3 * BK
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    D = cp.random.rand(m, n)
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n, K=k)
    assert np.allclose(C.get(), A.get() @ B.get() + D.get())


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
def test_large_tiled_add(depth: int):
    """2**26 doubles (512 MiB per array) over 2**20 thread blocks."""
    import cupy as cp

    sdfg = pipelined(tiled_add, depth)
    n = 2**26
    A = cp.random.rand(n)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    assert cp.allclose(B, 2 * A + 1)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
def test_large_blocked_gemm(depth: int):
    """A 1024 x 1024 x 2048 GEMM with 256 iterations in the pipelined K loop and N loop staging ``D``."""
    import cupy as cp

    sdfg = pipelined(blocked_gemm, depth, 2)
    m, n, k = 1024, 1024, 2048
    A = cp.random.rand(m, k)
    B = cp.random.rand(k, n)
    C = cp.zeros((m, n))
    D = cp.random.rand(m, n)
    sdfg(A=A, B=B, C=C, D=D, M=m, N=n, K=k)
    assert cp.allclose(C, A @ B + D, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    test_shared_tiles_code_structure()
    for d in (2, 3):
        test_tiled_add(d)
        for kk in (BK, 2 * BK, 7 * BK):
            test_tiled_gemm(d, kk)
            test_blocked_gemm_k_tiles(d, kk)
    for d in (2, 3):
        test_blocked_gemm_code_structure(d)
        for nn in (BN, 2 * BN, 5 * BN):
            test_blocked_gemm_pipelined_n_tiles(d, nn)
    for d in (4, 5, 8):
        test_shared_tiles_depth_beyond_trip_count_code_structure(d)
        test_tiled_add_depth_beyond_trip_count(d)
    test_tiled_add_single_block()
    for d in (4, 5):
        test_tiled_gemm_depth_beyond_trip_count(d)
    for d in (4, 6):
        test_blocked_gemm_n_depth_beyond_trip_count(d)
    for d in (2, 3):
        test_large_tiled_add(d)
        test_large_blocked_gemm(d)
