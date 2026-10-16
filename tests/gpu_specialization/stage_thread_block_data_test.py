# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""StageThreadBlockData stages the tile a thread block reads, and on request writes, through shared memory.

The structural checks need no GPU; the numeric ones run on the device, at sizes that leave the last tile partial.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.stage_thread_block_data import StageThreadBlockData
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync, is_barrier

M, N = (dace.symbol(s) for s in "MN")


def weighted_columns(dtype: dace.typeclass = dace.float64):
    """The program in ``dtype``: every row of a tile reads the same weights and indices (a block reuses them), and
    gathers ``A``."""

    @dace.program
    def weighted_columns(w: dtype[3, N], idx: dace.int32[N], A: dtype[M, N], B: dtype[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            B[i, j] = w[0, j] * A[i, j] + w[1, j] * A[i, idx[j]] + w[2, j]

    return weighted_columns


def canonical_gpu(name: str, dtype: dace.typeclass = dace.float64) -> dace.SDFG:
    sdfg = weighted_columns(dtype).to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    return sdfg


def nodes_of(sdfg: dace.SDFG, kind) -> list:
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, kind)]


def test_loads_are_staged_with_one_barrier():
    """The weights and indices a tile reads are copied to shared memory collectively, one barrier after all of
    them; ``A`` (gathered across the whole row) is not bounded by the tile and stays in global memory."""
    sdfg = canonical_gpu("stage_loads")
    staged = StageThreadBlockData().apply_pass(sdfg, {})
    assert staged and sorted(next(iter(staged.values()))) == ["load:idx", "load:w"]
    shared = {name for name, desc in sdfg.arrays.items() if desc.storage == dtypes.StorageType.GPU_Shared}
    assert shared == {"shm_idx", "shm_w"}
    copies = nodes_of(sdfg, CopyLibraryNode)
    assert len(copies) == 2 and all(not c.sync and c.aligned_words for c in copies)
    assert len([t for t in nodes_of(sdfg, dace.nodes.Tasklet) if is_barrier(t)]) == 1
    sdfg.validate()


def test_stores_are_staged_behind_a_barrier():
    """With ``stage_stores`` the output tile is written to shared memory and copied out after a barrier."""
    sdfg = canonical_gpu("stage_stores")
    staged = StageThreadBlockData(stage_stores=True).apply_pass(sdfg, {})
    assert "store:B" in next(iter(staged.values()))
    assert len([t for t in nodes_of(sdfg, dace.nodes.Tasklet) if is_barrier(t)]) == 2
    sdfg.validate()


def test_the_default_sync_adds_no_second_barrier():
    """The code generator's shared-memory sync sees the stage's barrier and adds none of its own."""
    sdfg = canonical_gpu("stage_sync")
    StageThreadBlockData(stage_stores=True).apply_pass(sdfg, {})
    DefaultSharedMemorySync().apply_pass(sdfg, None)
    assert len([t for t in nodes_of(sdfg, dace.nodes.Tasklet) if is_barrier(t)]) == 2


def test_a_tile_larger_than_the_budget_is_not_staged():
    sdfg = canonical_gpu("stage_budget")
    assert StageThreadBlockData(shared_memory_bytes=16).apply_pass(sdfg, {}) is None


@pytest.mark.gpu
@pytest.mark.parametrize("stores", [False, True])
@pytest.mark.parametrize("dtype", [dace.float64, dace.float32])
def test_staged_kernels_compute_what_numpy_does(stores, dtype):
    import cupy as cp

    m, n = 37, 1013
    sdfg = canonical_gpu(f"stage_numeric_{int(stores)}_{dtype.to_string()}", dtype)
    StageThreadBlockData(stage_stores=stores).apply_pass(sdfg, {})
    rng = np.random.default_rng(0)
    npt = dtype.as_numpy_dtype()
    w, a = rng.random((3, n)).astype(npt), rng.random((m, n)).astype(npt)
    idx = rng.integers(0, n, n).astype(np.int32)
    b = cp.zeros((m, n), dtype=npt)
    sdfg(w=cp.asarray(w), idx=cp.asarray(idx), A=cp.asarray(a), B=b, M=m, N=n)
    np.testing.assert_allclose(cp.asnumpy(b), w[0] * a + w[1] * a[:, idx] + w[2], rtol=1e-5)


if __name__ == "__main__":
    test_loads_are_staged_with_one_barrier()
    test_stores_are_staged_behind_a_barrier()
    test_the_default_sync_adds_no_second_barrier()
    test_a_tile_larger_than_the_budget_is_not_staged()
