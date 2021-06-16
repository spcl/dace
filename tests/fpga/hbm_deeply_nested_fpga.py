from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace import subsets
import hbm_copy_fpga

def create_deeply_nested_sdfg():
    sdfg = dace.SDFG("deepnest_test")
    state : dace.SDFGState = sdfg.add_state("init")
    xarr = state.add_array("x", [4, 10], dace.float32)
    sdfg.arrays["x"].location["hbmbank"] = sbs.Range.from_string("0:4")
    yarr = state.add_array("y", [4, 10], dace.float32)
    sdfg.arrays["y"].location["hbmbank"] = sbs.Range.from_string("4:8")

    topMapEntry, topMapExit = state.add_map("topmap", dict(k="0:2"))
    topMapEntry.schedule = dtypes.ScheduleType.Unrolled
    
    nsdfg = dace.SDFG("nest")
    nstate = nsdfg.add_state("nested_state")
    xRead = nstate.add_array("xin", [4, 10], dace.float32, dtypes.StorageType.FPGA_Global)
    xWrite = nstate.add_array("xout", [4, 10], dace.float32, dtypes.StorageType.FPGA_Global)
    nsdfg.arrays["xin"].location["hbmbank"] = sbs.Range.from_string("0:4")
    nsdfg.arrays["xout"].location["hbmbank"] = sbs.Range.from_string("4:8")
    mapEntry, mapExit = nstate.add_map("map1", dict(w="0:2"))
    mapEntry.schedule = dtypes.ScheduleType.Unrolled
    imapEntry, imapExit = nstate.add_map("map2", dict(i="0:10"))
    nope = nstate.add_tasklet("nop", dict(_in=None), dict(_out=None),
                                "_out = _in")
    inputMem = mem.Memlet("xin[2*k+w, i]")
    outputMem = mem.Memlet("xout[2*k+w, i]")
    nstate.add_memlet_path(xRead, mapEntry, imapEntry, nope, memlet=inputMem, dst_conn="_in")
    nstate.add_memlet_path(nope, imapExit, mapExit, xWrite, memlet=outputMem, src_conn="_out")
    nsdfg_node = state.add_nested_sdfg(nsdfg, state, set(["xin"]), set(['xout']))

    state.add_memlet_path(xarr, topMapEntry, nsdfg_node, 
                        memlet=mem.Memlet.from_array("x", sdfg.arrays["x"]), dst_conn="xin")
    state.add_memlet_path(nsdfg_node, topMapExit, yarr, 
                        memlet=mem.Memlet.from_array("y", sdfg.arrays["y"]), src_conn="xout")
    sdfg.apply_fpga_transformations()

    return sdfg

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

def exec_deeply_nested_test():
    sdfg = create_deeply_nested_sdfg()
    a = np.zeros((4, 10), np.float32)
    a[2, 4:9] += 1
    a[3, 3:8] += 2
    a[0, 7] += 3
    c = np.ones((4, 10), np.float32)
    sdfg(x=a, y=c)
    assert np.allclose(a, c, 10e-6)

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

if __name__ == "__main__":
    #exec_deeply_nested_test()
    exec_deeply_nested_copy_test()