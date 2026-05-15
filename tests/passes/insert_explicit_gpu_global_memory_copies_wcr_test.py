# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end pins that ``InsertExplicitGPUGlobalMemoryCopies`` does not demote a WCR (atomic
accumulator) array to ``Register`` -- doing so would lose atomic semantics and produce wrong totals."""
import numpy as np
import pytest

import dace


@pytest.mark.gpu
def test_wcr_via_augmented_assign():
    """``acc[0] += A[i]`` in a GPU_Device map accumulates atomically; the accumulator is not demoted."""

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
    """Row-reduction kernel: a 2D map atomically accumulates each row of ``A`` into ``row_sums[i]``."""

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
    """``total[0] = np.sum(A)`` with no storage annotations: runtime auto-staging makes
    ``auto_optimize``'s sequentialized small reduce valid. Reference for the explicit-storage tests."""
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
