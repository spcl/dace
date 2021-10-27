import dace
from multibank_copy_fpga_test import mkc
from dace.dtypes import StorageType
from dace.transformation.dataflow import BankSplit
from dace.transformation import optimizer
import numpy as np


def test_simple_split():
    sdfg = dace.SDFG("hbm_bank_split_first_dim")
    _, b, a = mkc(sdfg, None, "b", "a", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [4, 10, 10], [40, 10], "b")
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=BankSplit):
        xform.apply(sdfg)
    sdfg(a=a, b=b)
    assert np.allclose(b[1], a[10:20, :])
    assert np.allclose(b[3], a[30:40, :])


def test_even_split_3d():
    sdfg = dace.SDFG("hbm_bank_split_even_split_3d")
    s, b, a = mkc(sdfg, None, "b", "a", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [8, 50, 50, 50], [100, 100, 100], "b")
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=BankSplit):
        xform.split_array_info = [2, 2, 2]
        xform.apply(sdfg)
    b = np.random.uniform(0, 100, [8, 50, 50, 50]).astype(np.int32)
    sdfg(a=a, b=b)
    assert np.allclose(a[0:50, 0:50, 0:50], b[0, :, :, :])
    assert np.allclose(a[50:100, 50:100, 50:100], b[7, :, :, :])
    assert np.allclose(a[0:50, 50:100, 0:50], b[2, :, :, :])


def test_second_dim_split_2d():
    sdfg = dace.SDFG("hbm_bank_split_sec_dim_split2d")
    s, a, b = mkc(sdfg, None, "a", "b", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [10, 100], [10, 10, 10], "b")
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=BankSplit):
        xform.split_array_info = [1, 10]
        xform.apply(sdfg)
    a = np.random.uniform(0, 10, [10, 100]).astype(np.int32)
    sdfg(a=a, b=b)
    for i in range(10):
        assert np.allclose(a[0:10, 10 * i:(10 * i + 10)], b[i])


def test_explicit_split_3d():
    sdfg = dace.SDFG("hbm_bank_split_explicit_3d")
    s, a, b = mkc(sdfg, None, "a", "b", StorageType.CPU_Heap,
                  StorageType.CPU_Heap, [120, 100, 100], [24, 40, 50, 25])
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=BankSplit):
        xform.split_array_info = [3, 2, 4]
        xform.apply(sdfg)
    a = np.random.uniform(0, 100, [120, 100, 100]).astype(np.int32)
    sdfg(a=a, b=b)
    assert np.allclose(a[80:120, 50:100, 75:100], b[23]) 
    assert np.allclose(a[0:40, 50:100, 75:100], b[7])
    assert np.allclose(a[40:80, 0:50, 25:50], b[9])


if __name__ == "__main__":
    test_simple_split()
    test_even_split_3d()
    test_second_dim_split_2d()
    test_explicit_split_3d()
