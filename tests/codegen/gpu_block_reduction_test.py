# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Communication between the threads of a block: thread-block reductions into shared memory across the states of a
kernel (one thread initializes the result, all threads add to it, then the threads read it), and global memory that
one thread writes and the others read after synchronizing."""

import numpy as np
import pytest

import dace

N = dace.symbol("N")
TB = dace.ScheduleType.GPU_ThreadBlock
THREADS = 64
WIDTH = 200  # Not a multiple of the threads: some threads add more elements than others


@dace.program
def row_sums(A: dace.float64[N, WIDTH], offsets: dace.float64[N], out: dace.float64[N, WIDTH]):
    # A kernel with a copy of one element at the device level (which code generation treats as a collaborative copy,
    # and which must not affect the next kernel)
    for i in dace.map[0:N]:
        o = offsets[i]
        for j in dace.map[0:WIDTH] @ TB:
            out[i, j] = o
    # A thread block per row
    for i in dace.map[0:N]:
        total = dace.ndarray([1], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for _ in dace.map[0:1] @ TB:
            with dace.tasklet:
                t >> total[0]
                t = 0
        for th in dace.map[0:THREADS] @ TB:
            with dace.tasklet:
                row << A[i, :]
                t >> total(1, lambda a, b: a + b)[0]
                part = dace.float64(0)
                for j in range(th, WIDTH, THREADS):
                    part = part + row[j]
                t = part
        for th in dace.map[0:THREADS] @ TB:
            with dace.tasklet:
                t << total[0]
                o >> out(-1)[i, :]
                for j in range(th, WIDTH, THREADS):
                    o[j] = o[j] + t


def _sdfg() -> dace.SDFG:
    sdfg = row_sums.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.params == ["i"]:
            node.map.schedule = dace.ScheduleType.GPU_Device
    return sdfg


@pytest.mark.gpu
def test_block_reduction_across_states():
    sdfg = _sdfg()
    code = next(c for c in sdfg.generate_code() if c.language == "cu").clean_code
    reduction = code[code.index("total[1]") :]
    # Between the initialization, the additions and the reads
    assert reduction.count("__syncthreads();") >= 2

    import cupy as cp  # Only GPU tests use CuPy

    rows = 37
    A = np.random.rand(rows, WIDTH)
    offsets = np.random.rand(rows)
    out = cp.zeros((rows, WIDTH))
    sdfg(A=cp.asarray(A), offsets=cp.asarray(offsets), out=out, N=rows)
    expected = offsets[:, None] + A.sum(axis=1, keepdims=True)
    assert np.allclose(cp.asnumpy(out), np.broadcast_to(expected, (rows, WIDTH)))


@dace.program
def warp_aggregation(A: dace.float64[N, 64], out: dace.float64[N, 5], grand_total: dace.float64[1]):
    for i in dace.map[0:N]:
        acc = dace.ndarray([5], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for _ in dace.map[0:1] @ TB:
            with dace.tasklet:
                a >> acc[:]
                a[0] = 0
                a[1] = 1e30
                a[2] = 0
                a[3] = 0
                a[4] = 0
        for th in dace.map[0:64] @ TB:
            with dace.tasklet:
                v << A[i, th]
                total >> acc(1, lambda a, b: a + b)[0]  # Every lane of every warp: combined per warp
                everything >> grand_total(1, lambda a, b: a + b)[0]  # Also in global memory
                least >> acc(1, lambda a, b: min(a, b))[1]
                parity >> acc(1, lambda a, b: a + b)[:]  # Two locations per warp: not combined
                total = v
                everything = v
                least = v
                parity[2 + th % 2] = v
        for th in dace.map[0:40] @ TB:  # The second warp is partial: not combined
            with dace.tasklet:
                v << A[i, th]
                some >> acc(1, lambda a, b: a + b)[4]
                some = v
        for th in dace.map[0:5] @ TB:
            with dace.tasklet:
                a << acc[th]
                o >> out[i, th]
                o = a


@pytest.mark.gpu
def test_warp_aggregated_shared_atomics():
    sdfg = warp_aggregation.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.params == ["i"]:
            node.map.schedule = dace.ScheduleType.GPU_Device
    code = next(c for c in sdfg.generate_code() if c.language == "cu").clean_code
    # The sum and the minimum over all threads; not the updates in tasklet code, nor those of the partial map
    # One combined branch for the three updates
    assert code.count("dace::warp_same_location(") == 3 and code.count("warp_combine(") == 3

    import cupy as cp  # Only GPU tests use CuPy

    rows = 32
    A = np.random.default_rng(0).random((rows, 64))
    out = cp.zeros((rows, 5))
    grand_total = cp.zeros(1)
    sdfg(A=cp.asarray(A), out=out, grand_total=grand_total, N=rows)
    assert np.allclose(cp.asnumpy(grand_total), A.sum())
    expected = np.stack(
        [A.sum(axis=1), A.min(axis=1), A[:, 0::2].sum(axis=1), A[:, 1::2].sum(axis=1), A[:, :40].sum(axis=1)], axis=1
    )
    assert np.allclose(cp.asnumpy(out), expected)


@dace.program
def broadcast_count(A: dace.int32[N, WIDTH], counts: dace.int32[N], out: dace.int32[N, THREADS]):
    for i in dace.map[0:N]:
        # One thread counts (long enough for the other threads to arrive at the synchronization first)
        for _ in dace.map[0:1] @ TB:
            with dace.tasklet:
                row << A[i, :]
                c >> counts[i]
                total = 0
                for j in range(WIDTH):
                    if row[j] > 0:
                        total = total + 1
                c = total
        # Every thread reads the count that thread wrote
        for th in dace.map[0:THREADS] @ TB:
            with dace.tasklet:
                c << counts[i]
                o >> out[i, th]
                o = c


@pytest.mark.gpu
def test_global_memory_written_by_one_thread():
    sdfg = broadcast_count.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.params == ["i"]:
            node.map.schedule = dace.ScheduleType.GPU_Device
    code = next(c for c in sdfg.generate_code() if c.language == "cu").clean_code
    # The threads communicate through "counts", which the compiler must then not assume is modified by nobody else
    kernel = code[code.index("__global__") :]
    kernel = kernel[: kernel.index("{")]
    assert "int * counts" in kernel and "__restrict__ counts" not in kernel
    assert "const int * __restrict__ A" in kernel  # Only read: keeps its qualifiers

    import cupy as cp  # Only GPU tests use CuPy

    rows = 64
    A = np.random.default_rng(0).integers(-5, 5, (rows, WIDTH)).astype(np.int32)
    out = cp.zeros((rows, THREADS), dtype=cp.int32)
    sdfg(A=cp.asarray(A), counts=cp.zeros(rows, dtype=cp.int32), out=out, N=rows)
    assert np.array_equal(cp.asnumpy(out), np.repeat((A > 0).sum(axis=1, keepdims=True), THREADS, axis=1))


if __name__ == "__main__":
    test_block_reduction_across_states()
    test_global_memory_written_by_one_thread()
    test_warp_aggregated_shared_atomics()
