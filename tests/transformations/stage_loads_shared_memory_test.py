# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""StageLoadsThroughSharedMemory stages the global arrays a kernel's threads reuse through shared memory.

The structural checks need no GPU; the numeric ones run on the device, at sizes that leave the last tile partial.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.transformation.dataflow.stage_loads_shared_memory import StageLoadsThroughSharedMemory
from dace.transformation.passes.shared_memory_synchronization import is_barrier

N, K = (dace.symbol(s) for s in "NK")


@dace.program
def stencil(A: dace.float64[N + 2], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 2.0 * A[i + 1] + A[i + 2]


@dace.program
def matmul(A: dace.float64[40, 24], B: dace.float64[24, 72], C: dace.float64[40, 72]):
    for i, j in dace.map[0:40, 0:72]:
        s = 0.0
        for k in range(24):
            s += A[i, k] * B[k, j]
        C[i, j] = s


@dace.program
def matmul_symbolic(A: dace.float64[N, K], B: dace.float64[K, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        s = 0.0
        for k in range(K):
            s += A[i, k] * B[k, j]
        C[i, j] = s


@dace.program
def scale(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = 2.0 * A[i]


def gpu_sdfg(program, name: str, gpu: bool = True) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    if gpu:
        sdfg.apply_gpu_transformations()
    return sdfg


def kernel_of(sdfg: dace.SDFG) -> tuple[dace.SDFGState, dace.nodes.MapEntry]:
    (state,) = sdfg.states()
    (entry,) = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is None]
    return state, entry


def stage(sdfg: dace.SDFG, **options) -> list[str]:
    return StageLoadsThroughSharedMemory.apply_to(sdfg, options=options, map_entry=kernel_of(sdfg)[1])


def refused(sdfg: dace.SDFG, **options) -> bool:
    return not StageLoadsThroughSharedMemory.can_be_applied_to(sdfg, options=options, map_entry=kernel_of(sdfg)[1])


def test_stencil_reads_go_through_one_shared_tile_after_a_barrier():
    """The tile of ``A`` (halo included) is a shared transient copied in before one barrier; every read of ``A``
    inside the thread-block map comes from it, and ``A`` itself is read only by the copy."""
    sdfg = gpu_sdfg(stencil, "smem_stencil_structure")
    assert stage(sdfg, block_size=[32, 1, 1]) == ["A_gpu"]
    state, kernel = kernel_of(sdfg)
    shared = [n for n, d in sdfg.arrays.items() if d.transient and d.storage == dtypes.StorageType.GPU_Shared]
    assert shared == ["shm_A_gpu"]
    assert sdfg.arrays["shm_A_gpu"].shape == (34,)
    (block,) = [n for n in state.scope_children()[kernel] if isinstance(n, dace.nodes.MapEntry)]
    assert block.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    reads = {e.data.data for e in state.out_edges(block) if not e.data.is_empty()}
    assert reads == {"shm_A_gpu"}
    (copy_node,) = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
    assert {e.data.data for e in state.in_edges(copy_node)} == {"A_gpu"}
    barriers = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and is_barrier(n)]
    assert len(barriers) == 1 and state.edges_between(barriers[0], block)
    sdfg.validate()


def test_matmul_stages_both_operand_tiles():
    sdfg = gpu_sdfg(matmul, "smem_matmul_structure")
    assert stage(sdfg, block_size=[16, 8, 1]) == ["A_gpu", "B_gpu"]
    shared = {n for n, d in sdfg.arrays.items() if d.storage == dtypes.StorageType.GPU_Shared}
    assert shared == {"shm_A_gpu", "shm_B_gpu"}
    sdfg.validate()


def test_arrays_restricts_what_is_staged():
    sdfg = gpu_sdfg(matmul, "smem_matmul_only_b")
    assert stage(sdfg, arrays=["B_gpu"], block_size=[16, 8, 1]) == ["B_gpu"]
    assert [n for n, d in sdfg.arrays.items() if d.storage == dtypes.StorageType.GPU_Shared] == ["shm_B_gpu"]


def test_refusals():
    """No reuse (each element read once), a CPU map, a footprint that is not a compile-time constant, and an
    ``arrays`` choice naming nothing the kernel reuses are all refused."""
    assert refused(gpu_sdfg(scale, "smem_no_reuse"))
    assert refused(gpu_sdfg(stencil, "smem_cpu", gpu=False))
    assert refused(gpu_sdfg(matmul_symbolic, "smem_symbolic_k"))
    assert refused(gpu_sdfg(stencil, "smem_wrong_array"), arrays=["B_gpu"])


def test_a_staged_kernel_is_not_staged_again():
    sdfg = gpu_sdfg(stencil, "smem_twice")
    stage(sdfg, block_size=[32, 1, 1])
    assert refused(sdfg)


@pytest.mark.gpu
def test_stencil_on_the_gpu():
    n = 1000
    sdfg = gpu_sdfg(stencil, "smem_stencil_numeric")
    stage(sdfg, block_size=[64, 1, 1])
    a = np.random.default_rng(0).random(n + 2)
    b = np.zeros(n)
    sdfg(A=a, B=b, N=n)
    np.testing.assert_array_equal(b, a[:-2] + 2.0 * a[1:-1] + a[2:])


@pytest.mark.gpu
def test_matmul_on_the_gpu():
    rng = np.random.default_rng(0)
    a, b = rng.random((40, 24)), rng.random((24, 72))
    reference = np.zeros((40, 72))
    unstaged = gpu_sdfg(matmul, "smem_matmul_reference")
    kernel_of(unstaged)[1].map.gpu_block_size = [16, 16, 1]
    unstaged(A=a, B=b, C=reference)
    sdfg = gpu_sdfg(matmul, "smem_matmul_numeric")
    stage(sdfg, block_size=[16, 16, 1])
    c = np.zeros((40, 72))
    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, reference)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12)


if __name__ == "__main__":
    test_stencil_reads_go_through_one_shared_tile_after_a_barrier()
    test_matmul_stages_both_operand_tiles()
    test_arrays_restricts_what_is_staged()
    test_refusals()
    test_a_staged_kernel_is_not_staged_again()
    test_stencil_on_the_gpu()
    test_matmul_on_the_gpu()
