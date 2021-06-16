"""
Performs several tests for HBM-copies and nesting of accesses to HBM.
Checked:
ND-Copy between Host/Device and on Device (with openCL) 
Nested unrolls
Nested sdfgs
Copy between HBM and DDR -no
Copy between HBM and StreamArray -no
"""

from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np

#This does not follow the import rules, but the test gets more readable
from dace.dtypes import StorageType
from dace.codegen.targets.fpga import _FPGA_STORAGE_TYPES

#helper MaKe_Copy that creates and appends states performing exactly one copy. If a provided
#name already exists it will use the old array
def mkc(sdfg : dace.SDFG, statebefore, src_name, dst_name, src_storage=None, dst_storage=None, 
    src_shape=None, dst_shape=None, copy_expr=None, src_loc = None, dst_loc = None, returnCreatedObjects = False):
    if copy_expr is None:
        copy_expr = src_name
    if(statebefore == None):
        state = sdfg.add_state(is_start_state=True)
    else:
        state = sdfg.add_state_after(statebefore)
    
    def mkarray(name, shape, storage, loc):
        if(name in sdfg.arrays):
            return sdfg.arrays[name]
        isTransient = False
        if(storage in _FPGA_STORAGE_TYPES):
            isTransient = True
        arr = sdfg.add_array(name, shape, dace.int32, storage, transient=isTransient)
        if loc is not None and loc[0] == "hbmbank":
            arr[1].location[loc[0]] = sbs.Range.from_string(loc[1])
        elif loc is not None:
            arr[1].location[loc[0]] = loc[1]
        return arr
    
    a = mkarray(src_name, src_shape, src_storage, src_loc)
    b = mkarray(dst_name, dst_shape, dst_storage, dst_loc)

    aAcc = state.add_access(src_name)
    bAcc = state.add_access(dst_name)

    edge = state.add_edge(aAcc, None, bAcc, None, 
        mem.Memlet(copy_expr))
    
    aNpArr = np.zeros(src_shape, dtype=np.int32)
    bNpArr = np.zeros(dst_shape, dtype=np.int32)
    if returnCreatedObjects:
        (state, aNpArr, bNpArr, aAcc, bAcc, edge)
    else:
        return (state, aNpArr, bNpArr)

def check_host2copy1():
    sdfg = dace.SDFG("ch2c1")
    s, a, _ = mkc(sdfg, None, "a", "b", StorageType.Default, 
            StorageType.FPGA_Global, [5, 5], [2, 3, 3],
            "a[2:5, 2:5]->0, 0:3, 0:3", None, ("hbmbank", "0:2"))
    s, _, _ = mkc(sdfg, s, "a", "b", copy_expr="a[0:3, 0:3]->1, 0:3, 0:3")
    s, _, c = mkc(sdfg, s, "b", "c", None, StorageType.Default,
        None, [2, 3, 3], "b")
    
    #sdfg.view()
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
        StorageType.FPGA_Global, [3, 5, 5, 5], [3, 5, 5, 5],
        "a", None, ("hbmbank", "0:3"))
    s, _, c = mkc(sdfg, s, "b", "c", None, StorageType.Default,
        None, [3, 3], "b[2, 2:5, 2:5, 4]->0:3, 0:3")
    
    #sdfg.view()
    a.fill(1)
    a[2, 2:4, 2:4, 4] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0:2, 0:2] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c, expect)

def check_dev2dev1():
    sdfg = dace.SDFG("d2d1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
        StorageType.FPGA_Global, [3, 5, 5, 5], [3, 5, 5, 5],
        "a", None, ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global, 
        None, [2, 10], "x[1, 2, 0:5, 2]->0, 0:5", None, ("hbmbank", "3:5"))
    s.add_access("a") #prevents from falling in device code
    _, _, c = mkc(sdfg, s, "y", "c", None, StorageType.Default,
        None, [2, 10], "y")
    
    #sdfg.view()
    a.fill(1)
    a[1, 2, 0:5, 2] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 0:5] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c, expect)

#TODO: Maybe use this check later for streams
"""
def check_hbm_and_streams1():
    sdfg = dace.SDFG("h2s1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
        StorageType.FPGA_Global, [4, 5], [4, 5], 
        "a", None, ("hbmbank", "0:4"))
    
    s = sdfg.add_state_after(s)
    sdfg.add_stream("st", dace.int32, 1, [4], StorageType.FPGA_Global)
    sdfg.add_array("y", [4, 5], dace.int32, StorageType.FPGA_Global)
    sdfg.arrays["y"].location["hbmbank"] = sbs.Range.from_string("4:8")
    xAcc = s.add_read("x")
    yAcc = s.add_write("y")
    stAcc = s.add_access("st")
    s.add_edge(xAcc, None, stAcc, None, mem.Memlet("x"))
    s.add_edge(stAcc, None, yAcc, None, mem.Memlet("st"))

    s, _, c = mkc(sdfg, s, "y", "c", None, StorageType.Default,
        None, [4, 5], "y")

    #sdfg.view()
    a.fill(1)
    a[3, 0] = 8
    expect = np.copy(c)
    expect.fill(1)
    expect[3, 0] = 8
    sdfg(a=a, c=c)
"""

def create_deeply_nested_copy_sdfg():
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = dace.SDFG('deepnestcopytest')
    state = sdfg.add_state('deepnestcopytest', True)

    sdfg.add_array("in1", [M, N], dace.int32)
    in1_fpga_glob = sdfg.add_array("in1_fpga", [4, M // 2, N // 2], dace.int32,
        dtypes.StorageType.FPGA_Global, transient=True)
    in1_fpga_glob[1].location["hbmbank"] = subsets.Range.from_string(f"0:{4}")
    readin1_glob = state.add_read("in1")
    outwrite_glob = state.add_write("in1_fpga")

    map0_enter, map0_exit = state.add_map("scal_inner_map", dict(n=f"0:2"))
    map0_enter.map.schedule = dtypes.ScheduleType.Unrolled
    map1_enter, map1_exit = state.add_map("scal_inner_map", dict(m=f"0:2"))
    map1_enter.map.schedule = dtypes.ScheduleType.Unrolled

    nsdfg = dace.SDFG("CopyNested")
    in1_n = nsdfg.add_array("in1__", [M, N], dace.int32)
    in1_fpga_n = nsdfg.add_array("in1_fpga__", [4, M // 2, N // 2], dace.int32,
        dtypes.StorageType.FPGA_Global)
    in1_fpga_n[1].location["hbmbank"] = subsets.Range.from_string(f"0:{4}")
    nstate = nsdfg.add_state('copy', True)

    readin1 = nstate.add_read("in1__")
    outwrite = nstate.add_write("in1_fpga__")
    memTo = dace.Memlet("in1__[m*(M//2), n*(N//2)]->m*2+n, 0:(M//2), 0:(N//2)", 
        volume="(M//2)*(N//2)", allow_oob=True)
    nstate.add_memlet_path(readin1, outwrite, memlet=memTo)
    
    nsdfg_node = state.add_nested_sdfg(nsdfg, None,
        set(["in1__"]), set(["in1_fpga__"]))

    state.add_memlet_path(readin1_glob, map0_enter, map1_enter, nsdfg_node, memlet=dace.Memlet("in1[0:M, 0:N]"), dst_conn="in1__")
    state.add_memlet_path(nsdfg_node, map1_exit, map0_exit, outwrite_glob, memlet=dace.Memlet("in1_fpga[0:4, 0:(M//2), 0:(N//2)]")
    , src_conn="in1_fpga__")

    hbm_copy_fpga.mkc(sdfg, state, "in1_fpga", "out", None, dtypes.StorageType.Default,
        None, [4, M // 2, N // 2], "in1_fpga")
    
    return sdfg

def exec_deeply_nested_copy_test():
    sdfg = create_deeply_nested_copy_sdfg()
    a = np.ones((10, 10), dtype=np.int32)
    a[3:7, 2:8] = 15
    expect = np.zeros((4, 5, 5), dtype=np.int32)
    expect[0, 0:5, 0:5] = a[0:5, 0:5]
    expect[1, 0:5, 0:5] = a[0:5, 5:10]
    expect[2, 0:5, 0:5] = a[5:10, 0:5]
    expect[3, 0:5, 0:5] = a[5:10, 5:10]
    out = np.zeros((4, 5, 5), dtype=np.int32)
    sdfg(in1=a, out=out, M=10, N=10)
    assert np.allclose(expect, out)

def check_hbm2hbm1():
    sdfg = dace.SDFG("hbm2hbm1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
        StorageType.FPGA_Global, [3, 4, 4], [3, 4, 4], "a", None,
        ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "y", None, StorageType.FPGA_Global,
        None, [2, 3, 3, 3], "x[2, 1:3, 1:3]->1, 1, 1:3, 1:3",
        None, ("hbmbank", "3:5"))
    s, _, _ = mkc(sdfg, s, "y", "z", None, StorageType.FPGA_Global,
        None, [1, 3, 3, 3], "y[1, 0:3, 0:3, 0:3]", None, ("hbmbank", "5:6"))
    s, _, _ = mkc(sdfg, s, "z", "w", None, StorageType.FPGA_Global,
        None, [1, 3, 3, 3], "z", None, ("hbmbank", "6:7"))
    s, _, c = mkc(sdfg, s, "z", "c", None, StorageType.Default,
        None, [2, 3, 3, 3], "z")

    a.fill(1)
    a[2, 1:3, 2] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[1, 1, 1:3, 2] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c, expect)

def check_hbm2ddr1():
    sdfg = dace.SDFG("hbm2ddr1")
    s, a, _ = mkc(sdfg, None, "a", "x", StorageType.Default,
        StorageType.FPGA_Global, [3, 5, 5], [3, 5, 5],
        "a", None, ("hbmbank", "0:3"))
    s, _, _ = mkc(sdfg, s, "x", "d1", None, StorageType.FPGA_Global,
        None, [3, 5, 5], "x[2, 0:5, 0:5]->2, 0:5, 0:5", None, ("bank", 1))
    s, _, _ = mkc(sdfg, s, "d1", "y", None, StorageType.FPGA_Global,
        None, [1, 7, 7], "d1[2, 0:5,0:5]->0, 2:7, 2:7", None,
        ("hbmbank", "3:4"))
    s, _, c = mkc(sdfg, s, "y", "c", None, StorageType.Default,
        None, [1, 7, 7], "y")

    a.fill(1)
    a[2, 1:4, 1:4] += 2
    expect = np.copy(c)
    expect.fill(1)
    expect[0, 3:6, 4:6] += 2
    sdfg(a=a, c=c)
    assert np.allclose(c, expect)


#check_host2copy1()
#check_dev2host1()
#check_dev2dev1()
check_hbm2hbm1()
check_hbm2ddr1()