# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace import subsets

# A test checking HBM in the context of nested maps and nested sdfgs


def create_deeply_nested_sdfg():
    sdfg = dace.SDFG("deepnest_test")
    state: dace.SDFGState = sdfg.add_state("init")
    xarr = state.add_array("x", [4, 10], dace.float32)
    sdfg.arrays["x"].location["bank"] = "hbm.0:4"
    yarr = state.add_array("y", [4, 10], dace.float32)
    sdfg.arrays["y"].location["bank"] = "hbm.4:8"

    topMapEntry, topMapExit = state.add_map("topmap", dict(k="0:2"))
    topMapEntry.schedule = dtypes.ScheduleType.Unrolled

    nsdfg = dace.SDFG("nest")
    nstate = nsdfg.add_state("nested_state")
    xRead = nstate.add_array("xin", [4, 10], dace.float32,
                             dtypes.StorageType.FPGA_Global)
    xWrite = nstate.add_array("xout", [4, 10], dace.float32,
                              dtypes.StorageType.FPGA_Global)
    nsdfg.arrays["xin"].location["bank"] = "hbm.0:4"
    nsdfg.arrays["xout"].location["bank"] = "hbm.4:8"
    mapEntry, mapExit = nstate.add_map("map1", dict(w="0:2"))
    mapEntry.schedule = dtypes.ScheduleType.Unrolled
    imapEntry, imapExit = nstate.add_map("map2", dict(i="0:10"))
    nope = nstate.add_tasklet("nop", dict(_in=None), dict(_out=None),
                              "_out = _in")
    inputMem = mem.Memlet("xin[2*k+w, i]")
    outputMem = mem.Memlet("xout[2*k+w, i]")
    nstate.add_memlet_path(xRead,
                           mapEntry,
                           imapEntry,
                           nope,
                           memlet=inputMem,
                           dst_conn="_in")
    nstate.add_memlet_path(nope,
                           imapExit,
                           mapExit,
                           xWrite,
                           memlet=outputMem,
                           src_conn="_out")
    nsdfg_node = state.add_nested_sdfg(nsdfg, state, set(["xin"]),
                                       set(['xout']))

    state.add_memlet_path(xarr,
                          topMapEntry,
                          nsdfg_node,
                          memlet=mem.Memlet.from_array("x", sdfg.arrays["x"]),
                          dst_conn="xin")
    state.add_memlet_path(nsdfg_node,
                          topMapExit,
                          yarr,
                          memlet=mem.Memlet.from_array("y", sdfg.arrays["y"]),
                          src_conn="xout")
    sdfg.apply_fpga_transformations()

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


if __name__ == "__main__":
    exec_deeply_nested_test()
