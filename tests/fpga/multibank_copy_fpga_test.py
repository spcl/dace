# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace.codegen.targets.fpga import _FPGA_STORAGE_TYPES
from dace.dtypes import StorageType
from dace.fpga_testing import fpga_test, xilinx_test

# A test checking copies involving Multibank-arrays using HBM and DDR in some way


def mkc(sdfg: dace.SDFG,
        state_before,
        src_name,
        dst_name,
        src_storage=None,
        dst_storage=None,
        src_shape=None,
        dst_shape=None,
        copy_expr=None,
        src_loc=None,
        dst_loc=None):
    """
    Helper MaKe_Copy that creates and appends states performing exactly one copy. If a provided
    arrayname already exists it will use the old array, and ignore all newly passed values
    """

    if copy_expr is None:
        copy_expr = src_name
    if (state_before == None):
        state = sdfg.add_state(is_start_state=True)
    else:
        state = sdfg.add_state_after(state_before)

    def mkarray(name, shape, storage, loc):
        if (name in sdfg.arrays):
            return sdfg.arrays[name]
        is_transient = False
        if (storage in _FPGA_STORAGE_TYPES):
            is_transient = True
        arr = sdfg.add_array(name, shape, dace.int32, storage, transient=is_transient)
        if loc is not None:
            arr[1].location["memorytype"] = loc[0]
            arr[1].location["bank"] = loc[1]
        return arr

    a = mkarray(src_name, src_shape, src_storage, src_loc)
    b = mkarray(dst_name, dst_shape, dst_storage, dst_loc)

    aAcc = state.add_access(src_name)
    bAcc = state.add_access(dst_name)

    edge = state.add_edge(aAcc, None, bAcc, None, mem.Memlet(copy_expr))

    a_np_arr, b_np_arr = None, None
    if src_shape is not None:
        try:
            a_np_arr = np.zeros(src_shape, dtype=np.int32)
        except:
            pass
    if dst_shape is not None:
        try:
            b_np_arr = np.zeros(dst_shape, dtype=np.int32)
        except:
            pass
    return (state, a_np_arr, b_np_arr)


# Note, usually there are only 4 ddr banks but much more hmb banks.
# Since the tests run in simulation mode, this should not be an issue.


def copy_multibank_1_mem_type(mem_type):
    sdfg = dace.SDFG("copy_multibank_1_mem_type_" + mem_type)
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default, StorageType.FPGA_Global, [3, 4, 4], [3, 4, 4], "a", None,
                  (mem_type, "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global, None, [2, 4, 4, 4],
                  "x[1, 1:4, 1:4]->1, 1:4, 1:4, 1", None, (mem_type, "3:5"))
    s, _, _ = mkc(sdfg, s, "y", "z", None, StorageType.FPGA_Global, None, [1, 4, 4, 4],
                  "y[1, 0:4, 0:4, 0:4]->0, 0:4, 0:4, 0:4", None, (mem_type, "5:6"))
    s, _, _ = mkc(sdfg, s, "z", "w", None, StorageType.FPGA_Global, None, [1, 4, 4, 4], "z", None, (mem_type, "6:7"))
    s, _, c = mkc(sdfg, s, "w", "c", None, StorageType.Default, None, [1, 4, 4, 4], "w")

    a.fill(1)
    a[1, 0:4, 1] += 2
    a[1, 1, 0:4] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 1:5, 1, 1] += 2
    expect[0, 1, 1:5, 1] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c[0, 1:4, 1:4, 1], expect[0, 1:4, 1:4, 1])
    return sdfg


def copy_multibank_2_mem_type(mem_type_1, mem_type_2):
    sdfg = dace.SDFG("copy_multibank_2_mem_type_" + mem_type_1 + "_" + mem_type_2)
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default, StorageType.FPGA_Global, [3, 5, 5], [3, 5, 5], "a", None,
                  (mem_type_1, "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "d1", None, StorageType.FPGA_Global, None, [3, 5, 5], "x[2, 0:5, 0:5]->1, 0:5, 0:5",
                  None, (mem_type_2, "1:4"))
    s, _, _ = mkc(sdfg, s, "d1", "y", None, StorageType.FPGA_Global, None, [1, 7, 7], "d1[1, 0:5,0:5]->0, 2:7, 2:7",
                  None, (mem_type_1, "3:4"))
    s, _, c = mkc(sdfg, s, "y", "c", None, StorageType.Default, None, [1, 7, 7], "y")

    a.fill(1)
    a[2, 2:4, 2:4] += 3
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 4:6, 4:6] += 3
    sdfg(a=a, c=c)
    assert np.allclose(c[2:7], expect[2:7])
    return sdfg


@xilinx_test()
def test_copy_hbm2hbm():
    return copy_multibank_1_mem_type(mem_type="hbm")


@xilinx_test()
def test_copy_ddr2ddr():
    return copy_multibank_1_mem_type(mem_type="ddr")


@xilinx_test()
def test_copy_hbm2ddr():
    return copy_multibank_2_mem_type(mem_type_1="hbm", mem_type_2="ddr")


@xilinx_test()
def test_copy_ddr2hbm():
    return copy_multibank_2_mem_type(mem_type_1="ddr", mem_type_2="hbm")


if __name__ == "__main__":
    test_copy_hbm2hbm(None)  # HBM to HBM to HBM
    test_copy_ddr2ddr(None)  # DDR to DDR to DDR
    test_copy_hbm2ddr(None)  # HBM to DDR to HBM
    test_copy_ddr2hbm(None)  # DDR to HBM to DDR
