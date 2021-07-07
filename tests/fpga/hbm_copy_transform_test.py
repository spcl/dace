import dace
from hbm_copy_fpga import mkc
from dace.dtypes import StorageType
from dace.transformation.dataflow import hbm_copy_transform
from dace.transformation import optimizer
import numpy as np


def test_even_split_3d():
    sdfg = dace.SDFG("hbm_copy_transform_even_split_3d")
    s, a, b = mkc(sdfg, None, "a", "b", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [100, 100, 100], [8, 50, 50, 50], "a")
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=hbm_copy_transform.HbmCopyTransform):
        xform.apply(sdfg)
    a = np.random.uniform(0, 100, [100, 100, 100]).astype(np.int32)
    sdfg(a=a, b=b)
    assert np.allclose(a[0:50, 0:50, 0:50], b[0, :, :, :])
    assert np.allclose(a[50:100, 50:100, 50:100], b[7, :, :, :])
    assert np.allclose(a[0:50, 50:100, 0:50], b[2, :, :, :])


def test_second_dim_split_2d():
    sdfg = dace.SDFG("hbm_copy_transform_sec_dim_split2d")
    s, a, b = mkc(sdfg, None, "a", "b", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [10, 100], [10, 10, 10], "b")
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=hbm_copy_transform.HbmCopyTransform):
        xform.split_array_info = [1, 10]
        xform.apply(sdfg)
    a = np.random.uniform(0, 10, [10, 100]).astype(np.int32)
    sdfg(a=a, b=b)
    for i in range(10):
        assert np.allclose(a[0:10, 10 * i:(10 * i + 10)], b[i])


def test_explicit_split_3d():
    sdfg = dace.SDFG("hbm_copy_transform_explicit_3d")
    s, a, b = mkc(sdfg, None, "a", "b", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [120, 100, 100], [24, 40, 50, 25])
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=hbm_copy_transform.HbmCopyTransform):
        xform.split_array_info = [3, 2, 4]
        xform.apply(sdfg)
    a = np.random.uniform(0, 100, [120, 100, 100]).astype(np.int32)
    sdfg(a=a, b=b)
    assert np.allclose(a[80:120, 50:100, 75:100], b[23])
    assert np.allclose(a[0:40, 50:100, 75:100], b[7])
    assert np.allclose(a[40:80, 0:50, 25:50], b[9])


if __name__ == "__main__":
    test_even_split_3d()
    test_second_dim_split_2d()
    test_explicit_split_3d()
