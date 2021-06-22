# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace.dtypes import StorageType
from dace.codegen.targets.fpga import _FPGA_STORAGE_TYPES

def print_result(a, c, expect):
    print("A:")
    print(a)
    print("C")
    print(c)
    print("E")
    print(expect)

#Tests copy of 2 and 3d blocks between host and device

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
            aNpArr = np.random.randint(1, 9, src_shape, dtype=np.int32)
        except:
            pass
    if dst_shape is not None:
        try:
            bNpArr = np.random.randint(1, 9, dst_shape, dtype=np.int32)
        except:
            pass
    return (state, aNpArr, bNpArr)


def check_host2dev1():
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


def check_dev2host1():
    sdfg = dace.SDFG("d2h1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default,
                  StorageType.FPGA_Global, [5, 5, 5], [5, 5, 5], "a")
    s, _, _ = mkc(sdfg, s, "b", "z", None, StorageType.FPGA_Global, None,
                  [5, 5, 5], "b")

    s, _, c = mkc(sdfg, s, "b", "c", None, StorageType.Default, None, [3, 3],
                  "b[2:5, 2:4, 3]->0:3, 0:2")

    sdfg(a=a, c=c)
    assert np.allclose(a[2:5, 2:4, 3], c[0:3, 0:2])


def check_dev2dev1():
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


def checkhost2dev2():
    sdfg = dace.SDFG("h2d2")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default, StorageType.FPGA_Global,
         [7, 4, 5], [6, 3, 4], "a[1:5, 1:4, 1:4]->1:5, 0:3, 1:4")
    s, _, _ = mkc(sdfg, s, "x", "z", StorageType.FPGA_Global,
                  StorageType.FPGA_Global, [6, 3, 4], [6, 3, 4], "x")
    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default, None,
                  [6, 3, 4], "z")

    sdfg(a=a, c=c)
    assert np.allclose(a[1:5, 1:4, 1:4], c[1:5, 0:3, 1:4])


def checkdev2host2():
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

if __name__ == "__main__":
    check_host2dev1()
    check_dev2host1()
    check_dev2dev1()
    checkhost2dev2()
    checkdev2host2()
