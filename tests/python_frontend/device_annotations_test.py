# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import numpy as np

from dace.dtypes import StorageType, DeviceType, ScheduleType
from dace import dtypes

try:
    import cupy
except (ImportError, ModuleNotFoundError):
    cupy = None


@pytest.mark.gpu
def test_storage():

    @dace.program
    def add(X: dace.float32[32, 32] @ StorageType.GPU_Global):
        return X + 1

    sdfg = add.to_sdfg()
    sdfg.apply_gpu_transformations()

    X = cupy.random.random((32, 32)).astype(cupy.float32)
    Y = sdfg(X=X)
    assert cupy.allclose(Y, X + 1)


@pytest.mark.gpu
def test_schedule():
    Seq = ScheduleType.Sequential
    N = dace.symbol('N')

    @dace.program
    def add2(X: dace.float32[32, 32] @ StorageType.GPU_Global):
        for i in dace.map[0:N] @ ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ dace.ScheduleType.Sequential:
                X[i, j] = X[i, j] + 1
        for i in dace.map[0:32] @ dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ Seq:
                X[i, j] = X[i, j] + 1
        return X

    X = cupy.random.random((32, 32)).astype(cupy.float32)
    Y = X.copy()
    add2(X=X, N=32)
    assert cupy.allclose(Y + 2, X)


@pytest.mark.gpu
def test_pythonmode():

    def runs_on_gpu(a: dace.float64[20] @ StorageType.GPU_Global, b: dace.float64[20] @ StorageType.GPU_Global):
        # This map will become a GPU kernel
        for i in dace.map[0:20] @ ScheduleType.GPU_Device:
            b[i] = a[i] + 1.0

    gpu_a = cupy.random.rand(20)
    gpu_b = cupy.random.rand(20)
    runs_on_gpu(gpu_a, gpu_b)
    assert cupy.allclose(gpu_b, gpu_a + 1)


def test_inline_storage_hint():
    N = dace.symbol('N')

    @dace.program
    def tester():
        b = np.ones(N, dtype=np.float32) @ dace.StorageType.CPU_ThreadLocal
        return b + 1

    sdfg = tester.to_sdfg(simplify=False)
    assert sdfg.arrays['b'].storage == StorageType.CPU_ThreadLocal

    b = tester(N=10)
    assert np.allclose(b, 2)


def test_annotated_storage_hint():
    N = dace.symbol('N')

    @dace.program
    def tester():
        b: dace.float32[N] @ dace.StorageType.CPU_ThreadLocal = np.ones(N, dtype=np.float32)
        return b + 1

    sdfg = tester.to_sdfg(simplify=False)
    assert sdfg.arrays['b'].storage == StorageType.CPU_ThreadLocal

    b = tester(N=10)
    assert np.allclose(b, 2)


if __name__ == "__main__":
    if cupy is not None:
        test_storage()
        test_schedule()
        test_pythonmode()
    test_inline_storage_hint()
    test_annotated_storage_hint()
