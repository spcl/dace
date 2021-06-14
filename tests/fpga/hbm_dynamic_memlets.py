from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np

def create_dynamic_memlet_sdfg():
    sdfg = dace.SDFG("dyn_memlet")
    state : dace.SDFGState = sdfg.add_state("dyn_memlet")
    xarr = state.add_array("x", [4, 10], dace.int32)
    sdfg.arrays["x"].location["hbmbank"] = sbs.Range.from_string("0:4")
    yarr = state.add_array("y", [4, 10], dace.int32)
    sdfg.arrays["y"].location["hbmbank"] = sbs.Range.from_string("4:8")

    hbmmapenter, hbmmapexit = state.add_map("hbmmap", dict(k="0:4"), dtypes.ScheduleType.Unrolled)
    arrmapenter, arrmapexit = state.add_map("map", dict(i="0:_dynbound"))
    tasklet = state.add_tasklet("dyn", set(["_in"]), set(["_out"]),
        ("if(i == 2):\n"
        "   _out = 2\n"
        "elif (_in != 2):\n"
        "   _out = _in\n"))

    state.add_memlet_path(xarr, hbmmapenter, arrmapenter, tasklet, 
        memlet=mem.Memlet("x[k, i]", dynamic=True), dst_conn="_in")
    state.add_memlet_path(tasklet, arrmapexit, hbmmapexit, yarr, 
        memlet=mem.Memlet("y[k, i]", dynamic=True), src_conn="_out")
    state.add_memlet_path(xarr, hbmmapenter, arrmapenter,
        memlet=mem.Memlet("x[1, 0]") ,dst_conn="_dynbound")
    sdfg.apply_fpga_transformations()
    return sdfg

def exec_dynamic_memlet_test():
    sdfg = create_dynamic_memlet_sdfg()
    x = np.zeros((4, 10), dtype=np.int32)
    #has to be copied to sdfg
    y = np.ones((4, 10), dtype=np.int32) 
    x[0:4, 8] = 2
    x[1, 0] = 10
    expected = np.copy(x)
    expected[0:4, 2] = 2
    expected[0:4, 8] = 1
    sdfg(x=x, y=y)
    assert np.allclose(y, expected)

if __name__ == "__main__":
    exec_dynamic_memlet_test()