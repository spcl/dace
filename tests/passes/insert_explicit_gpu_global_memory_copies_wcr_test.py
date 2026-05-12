# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end pins for WCR (write-conflict-resolution / atomic
accumulator) survival through the experimental GPU codegen pipeline.

``InsertExplicitGPUGlobalMemoryCopies`` demotes per-thread transient
GPU_Global scalars to ``Register`` to avoid a host-side ``cudaMalloc``,
but it must skip the demotion when the array has any incoming WCR
memlet — otherwise the kernel loses atomic semantics and silently
produces wrong totals.

Storages and schedules are annotated directly on the ``@dace.program``
so the tests don't depend on ``auto_optimize``'s tile-size heuristic.
"""
import numpy as np
import pytest

import dace


@pytest.mark.gpu
def test_wcr_via_augmented_assign():
    """``acc[0] += A[i]`` inside a GPU_Device map: DaCe emits a memlet
    with ``wcr = lambda a, b: a + b``. The codegen must atomically
    accumulate; the experimental pipeline must not demote the
    accumulator to a per-thread Register."""

    @dace.program
    def aug_assign(A: dace.float64[64] @ dace.StorageType.GPU_Global,
                   acc: dace.float64[1] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:64] @ dace.ScheduleType.GPU_Device:
            acc[0] += A[i]

    import cupy as cp
    A = cp.arange(64, dtype=cp.float64)
    acc = cp.zeros(1, dtype=cp.float64)
    aug_assign(A=A, acc=acc)
    assert float(acc[0]) == float(cp.sum(A))


@pytest.mark.gpu
def test_wcr_via_reduction_kernel():
    """Row-reduction kernel: 2D map atomically accumulating each row of
    ``A`` into ``row_sums[i]`` via WCR. Multiple WCR targets (one per
    row), each with cross-thread atomic adds along the ``j`` axis."""

    @dace.program
    def row_reduce(A: dace.float64[8, 8] @ dace.StorageType.GPU_Global,
                   row_sums: dace.float64[8] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:8, 0:8] @ dace.ScheduleType.GPU_Device:
            row_sums[i] += A[i, j]

    import cupy as cp
    A = cp.arange(64, dtype=cp.float64).reshape(8, 8)
    row_sums = cp.zeros(8, dtype=cp.float64)
    row_reduce(A=A, row_sums=row_sums)
    cp.testing.assert_array_equal(row_sums, A.sum(axis=1))


@pytest.mark.gpu
def test_wcr_np_sum_small_n_auto_staging():
    """``total[0] = np.sum(A)`` over 64 elements with NO explicit
    storage annotations. DaCe's runtime auto-stages host arrays to GPU
    around the compiled SDFG; ``auto_optimize``'s decision to
    sequentialize the small reduce map is then valid because the inputs
    are visible on the host. Reference behaviour for the explicit-storage
    sibling test below."""
    from dace.dtypes import DeviceType
    from dace.transformation.auto.auto_optimize import auto_optimize

    @dace.program
    def reduce_sum(A: dace.float64[64], total: dace.float64[1]):
        total[0] = np.sum(A)

    sdfg = reduce_sum.to_sdfg()
    auto_optimize(sdfg, DeviceType.GPU)

    A = np.arange(64, dtype=np.float64)
    total = np.zeros(1, dtype=np.float64)
    sdfg(A=A, total=total)
    assert total[0] == np.sum(A)


@pytest.mark.gpu
@pytest.mark.xfail(
    reason="auto_optimize sequentializes the reduce map for N < its tile "
    "threshold (here N=64) but leaves the input array as GPU_Global. The "
    "resulting Sequential (host) tasklet then reads GPU memory, and the "
    "SDFG validator rejects the edge with 'Data container <X> is stored "
    "as GPU_Global but accessed on host'. The auto-staging sibling above "
    "dodges this by leaving storage implicit; the explicit-storage form "
    "needs a proper fix in auto_optimize (or the validator).",
    strict=True,
)
def test_wcr_np_sum_small_n_explicit_gpu_storage_threshold_bug():
    """Natural ``total[0] = np.sum(A)`` over a 64-element input with
    explicit ``GPU_Global`` annotations on both parameters. Expected:
    ``auto_optimize`` GPU-fies the SDFG, the Reduce libnode lowers via
    WCR atomics, the kernel runs, ``total`` matches ``cupy.sum(A)``.

    Actual: ``auto_optimize`` decides ``N=64`` is below its tile
    threshold and sequentializes the reduce map, but does not propagate
    that schedule decision to the storage layer. The Sequential (host)
    tasklet then reads ``GPU_Global`` data and the validator aborts."""
    from dace.dtypes import DeviceType
    from dace.transformation.auto.auto_optimize import auto_optimize

    @dace.program
    def reduce_sum(A: dace.float64[64] @ dace.StorageType.GPU_Global,
                   total: dace.float64[1] @ dace.StorageType.GPU_Global):
        total[0] = np.sum(A)

    sdfg = reduce_sum.to_sdfg()
    auto_optimize(sdfg, DeviceType.GPU)

    import cupy as cp
    A = cp.arange(64, dtype=cp.float64)
    total = cp.zeros(1, dtype=cp.float64)
    sdfg(A=A, total=total)
    assert float(total[0]) == float(cp.sum(A))
