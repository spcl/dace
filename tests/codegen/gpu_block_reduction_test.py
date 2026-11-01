# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Thread-block reductions into shared memory across the states of a kernel: one thread initializes the result, all
threads add to it, then the threads read it. The threads of the block synchronize between the states."""

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


if __name__ == "__main__":
    test_block_reduction_across_states()
