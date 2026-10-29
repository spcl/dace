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
