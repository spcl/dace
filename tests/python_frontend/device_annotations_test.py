import dace
import pytest

from dace.dtypes import StorageType, DeviceType, ScheduleType
from dace import dtypes

cupy = pytest.importorskip("cupy")


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

    @dace.program
    def add2(X: dace.float32[32, 32] @ StorageType.GPU_Global):
        for i in dace.map[0:32] @ ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ dace.ScheduleType.Sequential:
                X[i, j] = X[i, j] + 1
        for i in dace.map[0:32] @ dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ Seq:
                X[i, j] = X[i, j] + 1
        return X

    X = cupy.random.random((32, 32)).astype(cupy.float32)
    Y = X.copy()
    add2(X=X)
    assert cupy.allclose(Y + 2, X)


if __name__ == "__main__":
    test_storage()
    test_schedule()
