# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace.dtypes import StorageType
from dace.codegen.targets.fpga import _FPGA_STORAGE_TYPES

#A test checking copies involving HBM-arrays in some way


def print_result(a, c, expect):
    print("A:")
    print(a)
    print("C")
    print(c)
    print("E")
    print(expect)


#helper MaKe_Copy that creates and appends states performing exactly one copy. If a provided
#arrayname already exists it will use the old array, and ignore all newly passed values
def mkc(sdfg: dace.SDFG,
        statebefore,
        src_name,
        dst_name,
        src_storage=None,
        dst_storage=None,
        src_shape=None,
        dst_shape=None,
        copy_expr=None,
        src_loc=None,
        dst_loc=None):
    if copy_expr is None:
        copy_expr = src_name
    if (statebefore == None):
        state = sdfg.add_state(is_start_state=True)
    else:
        state = sdfg.add_state_after(statebefore)

    def mkarray(name, shape, storage, loc):
        if (name in sdfg.arrays):
            return sdfg.arrays[name]
        isTransient = False
        if (storage in _FPGA_STORAGE_TYPES):
            isTransient = True
        arr = sdfg.add_array(name,
                             shape,
                             dace.int32,
                             storage,
                             transient=isTransient)
        if loc is not None and loc[0] == "hbmbank":
            arr[1].location[loc[0]] = sbs.Range.from_string(loc[1])
        elif loc is not None:
            arr[1].location[loc[0]] = loc[1]
        return arr

    a = mkarray(src_name, src_shape, src_storage, src_loc)
    b = mkarray(dst_name, dst_shape, dst_storage, dst_loc)

    aAcc = state.add_access(src_name)
    bAcc = state.add_access(dst_name)

    edge = state.add_edge(aAcc, None, bAcc, None, mem.Memlet(copy_expr))

    aNpArr, bNpArr = None, None
    if src_shape is not None:
        try:
            aNpArr = np.zeros(src_shape, dtype=np.int32)
        except:
            pass
    if dst_shape is not None:
        try:
            bNpArr = np.zeros(dst_shape, dtype=np.int32)
        except:
            pass
    return (state, aNpArr, bNpArr)

"""
def check_host2copy1():
    sdfg = dace.SDFG("ch2c1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default,
                  StorageType.FPGA_Global, [5, 5], [2, 3, 3],
                  "a[2:5, 2:5]->0, 0:3, 0:3", None, ("hbmbank", "0:2"))
    s, _, _ = mkc(sdfg, s, "a", "b", copy_expr="a[0:3, 0:3]->1, 0:3, 0:3")

    #Dummy copy on FPGA because otherwise it will not compile
    s, _, _ = mkc(sdfg, s, "b", "z", None, StorageType.FPGA_Global, None,
                  [2, 3, 3], "b")

    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [2, 3, 3],
                  "z")

    a.fill(1)
    a[4, 4] = 4
    a[0, 0] = 5
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 2, 2] = 4
    expect[1, 0, 0] = 5

    sdfg(a=a, c=c)
    assert np.allclose(c, expect)


def check_dev2host1():
    sdfg = dace.SDFG("d2h1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default,
                  StorageType.FPGA_Global, [3, 5, 5, 5], [3, 5, 5, 5], "a",
                  None, ("hbmbank", "0:3"))

    #Dummy copy on FPGA because otherwise it will not compile
    s, _, _ = mkc(sdfg, s, "b", "z", None, StorageType.FPGA_Global, None,
                  [3, 5, 5, 5], "b", None, ("hbmbank", "3:6"))

    s, _, c = mkc(sdfg, s, "b", "c", None, StorageType.Default, None, [3, 3],
                  "b[2, 2:5, 2:5, 4]->0:3, 0:3")

    a.fill(1)
    a[2, 4, 2:4, 2:4] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0:2, 0:2] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c, expect)


def check_dev2dev1():
    sdfg = dace.SDFG("d2d1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
                  StorageType.FPGA_Global, [3, 5, 5, 5], [3, 5, 5, 5], "a",
                  None, ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global, None,
                  [2, 10], "x[1, 0:5, 2, 2]->0, 0:5", None, ("hbmbank", "3:5"))
    s.location["is_FPGA_kernel"] = 0

    #Dummy copy on FPGA because otherwise it will not compile
    s, _, _ = mkc(sdfg, s, "y", "z", None, StorageType.FPGA_Global, None,
                  [2, 10], "y", None, ("hbmbank", "5:7"))

    _, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None, [2, 10],
                  "z")

    a.fill(1)
    a[1, 2, 2, 0:5] += 2
    expect = np.copy(c)
    expect[0, 0:5] = 3
    sdfg(a=a, c=c)
    assert np.allclose(c[0, 0:5], expect[0, 0:5])
"""

def check_hbm2hbm1():
    sdfg = dace.SDFG("hbm2hbm1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
                  StorageType.FPGA_Global, [3, 4, 4], [3, 4, 4], "a", None,
                  ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global, None,
                  [2, 4, 4, 4], "x[1, 1:4, 1:4]->1, 1:4, 1:4, 1", None,
                  ("hbmbank", "3:5"))
    s, _, _ = mkc(sdfg, s, "y", "z", None, StorageType.FPGA_Global, None,
                  [1, 4, 4, 4], "y[1, 0:4, 0:4, 0:4]->0, 0:4, 0:4, 0:4", None,
                  ("hbmbank", "5:6"))
    s, _, _ = mkc(sdfg, s, "z", "w", None, StorageType.FPGA_Global, None,
                  [1, 4, 4, 4], "z", None, ("hbmbank", "6:7"))
    s, _, c = mkc(sdfg, s, "w", "c", None, StorageType.Default, None,
                  [1, 4, 4, 4], "w")

    a.fill(1)
    a[1, 0:4, 1] += 2
    a[1, 1, 0:4] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 1:5, 1, 1] += 2
    expect[0, 1, 1:5, 1] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c[0, 1:4, 1:4, 1], expect[0, 1:4, 1:4, 1])

def check_hbm2ddr1():
    sdfg = dace.SDFG("hbm2ddr1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
                  StorageType.FPGA_Global, [3, 5, 5], [3, 5, 5], "a", None,
                  ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "d1", None, StorageType.FPGA_Global, None,
                  [3, 5, 5], "x[2, 0:5, 0:5]->1, 0:5, 0:5", None, ("bank", 1))
    s, _, _ = mkc(sdfg, s, "d1", "y", None, StorageType.FPGA_Global, None,
                  [1, 7, 7], "d1[1, 0:5,0:5]->0, 2:7, 2:7", None,
                  ("hbmbank", "3:4"))
    s, _, c = mkc(sdfg, s, "y", "c", None, StorageType.Default, None, [1, 7, 7],
                  "y")

    a.fill(1)
    a[2, 2:4, 2:4] += 3
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 4:6, 4:6] += 3
    sdfg(a=a, c=c)
    assert np.allclose(c[2:7], expect[2:7])


if __name__ == "__main__":
    check_hbm2hbm1()
    check_hbm2ddr1()
