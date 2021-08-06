# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace.dtypes import StorageType
from dace.codegen.targets.fpga import _FPGA_STORAGE_TYPES
from dace.fpga_testing import xilinx_test
from hbm_copy_fpga_test import mkc

#Tests copy of 2 and 3d blocks between host and device

@xilinx_test()
def test_check_host2dev1():
    sdfg = dace.SDFG("h2d1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default,
                  StorageType.FPGA_Global, [5, 5], [3, 3],
                  "a[2:4, 2:4]->1:3, 0:2")
    s, _, _ = mkc(sdfg, s, "b", "z", None, StorageType.FPGA_Global, None,
                  [3, 3], "b")

    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [3, 3],
                  "z")

    sdfg(a=a, c=c)
    assert np.allclose(c[1:3, 0:2], a[2:4, 2:4])
    return sdfg

@xilinx_test()
def test_check_dev2host1():
    sdfg = dace.SDFG("d2h1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default,
                  StorageType.FPGA_Global, [5, 5, 5], [5, 5, 5], "a")
    s, _, _ = mkc(sdfg, s, "b", "z", None, StorageType.FPGA_Global, None,
                  [5, 5, 5], "b")

    s, _, c = mkc(sdfg, s, "b", "c", None, StorageType.Default, None, [3, 3],
                  "b[2:5, 2:4, 3]->0:3, 0:2")

    sdfg(a=a, c=c)
    assert np.allclose(a[2:5, 2:4, 3], c[0:3, 0:2])
    return sdfg

@xilinx_test()
def test_check_dev2dev1():
    sdfg = dace.SDFG("d2d1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
                  StorageType.FPGA_Global, [5, 5, 5], [5, 5, 5], "a")
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global, None, [10],
                  "x[2, 2, 1:5]->2:6")
    s.location["is_FPGA_kernel"] = 0
    s, _, _ = mkc(sdfg, s, "y", "z", None, StorageType.FPGA_Global, None, [10],
                  "y", None)

    _, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [10], "z")

    sdfg(a=a, c=c)
    assert np.allclose(c[2:6], a[2, 2, 1:5])
    return sdfg

@xilinx_test()
def test_checkhost2dev2():
    sdfg = dace.SDFG("h2d2")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default, StorageType.FPGA_Global,
         [7, 4, 5], [6, 3, 4], "a[1:5, 1:4, 1:4]->1:5, 0:3, 1:4")
    s, _, _ = mkc(sdfg, s, "x", "z", StorageType.FPGA_Global,
                  StorageType.FPGA_Global, [6, 3, 4], [6, 3, 4], "x")
    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None,
                  [6, 3, 4], "z")

    sdfg(a=a, c=c)
    assert np.allclose(a[1:5, 1:4, 1:4], c[1:5, 0:3, 1:4])
    return sdfg

@xilinx_test()
def test_checkdev2host2():
    sdfg = dace.SDFG("d2h2")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
                  StorageType.FPGA_Global, [40], [5, 5], "a[5:30]")
    s, _, _ = mkc(sdfg, s, "x", "z", None, StorageType.FPGA_Global, None,
                  [5, 5], "x", None)
    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [9, 6],
                  "z[0:5, 0:5]->2:7, 1:6")
    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [9, 6],
                  "z[3:5, 0:5]->0:2, 0:5")

    sdfg(a=a, c=c)
    expect = np.reshape(a[5:30], (5, 5), order='C')
    assert np.allclose(expect, c[2:7, 1:6])
    assert np.allclose(expect[3:5, 0:5], c[0:2, 0:5])
    return sdfg

if __name__ == "__main__":
    test_check_host2dev1(None)
    test_check_dev2host1(None)
    test_check_dev2dev1(None)
    test_checkhost2dev2(None)
    test_checkdev2host2(None)
