# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg.sdfg import InterstateEdge
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np


def create_deeply_nested_sdfg():
    sdfg = dace.SDFG("deepnest_test")
    state: dace.SDFGState = sdfg.add_state("init")
    xarr = state.add_array("x", [4, 100], dace.float32)
    yarr = state.add_array("y", [4, 100], dace.float32)

    topMapEntry, topMapExit = state.add_map("topmap", dict(k="0:2"))
    topMapEntry.schedule = dtypes.ScheduleType.Unrolled

    nsdfg = dace.SDFG("nest")
    nstate = nsdfg.add_state("nested_state", True)
    xRead = nstate.add_array("xin", [4, 100], dace.float32)
    xWrite = nstate.add_array("xout", [4, 100], dace.float32)
    mapEntry, mapExit = nstate.add_map("map1", dict(w="0:2"))
    mapEntry.schedule = dtypes.ScheduleType.Unrolled
    noUnrollEntry, noUnrollExit = nstate.add_map("map2", dict(i="0:100"))
    nope = nstate.add_tasklet("nop", dict(_in=None), dict(_out=None), "_out = _in")
    inputMem = mem.Memlet("xin[2*k+w, i]")
    outputMem = mem.Memlet("xout[2*k+w, i]")
    nstate.add_memlet_path(
        xRead,
        mapEntry,
        noUnrollEntry,
        nope,
        memlet=inputMem,
        dst_conn="_in",
    )
    nstate.add_memlet_path(
        nope,
        noUnrollExit,
        mapExit,
        xWrite,
        memlet=outputMem,
        src_conn="_out",
    )

    nstate2 = nsdfg.add_state("second_nest")
    tasklet = nstate2.add_tasklet("overwrite", set(), set(["_out"]), "_out = 15.0")
    xWrite2 = nstate2.add_write("xout")
    nstate2.add_memlet_path(
        tasklet,
        xWrite2,
        memlet=mem.Memlet("xout[mpt, 0]"),
        src_conn="_out",
    )

    nsdfg.add_edge(nstate, nstate2, InterstateEdge(None, dict(mpt="k")))
    nsdfg_node = state.add_nested_sdfg(nsdfg, state, set(["xin"]), set(['xout']))
    nsdfg_node.unique_name = "SomeUniqueName"

    state.add_memlet_path(
        xarr,
        topMapEntry,
        nsdfg_node,
        memlet=mem.Memlet.from_array("x", sdfg.arrays["x"]),
        dst_conn="xin",
    )
    state.add_memlet_path(
        nsdfg_node,
        topMapExit,
        yarr,
        memlet=mem.Memlet.from_array("y", sdfg.arrays["y"]),
        src_conn="xout",
    )

    return sdfg


def test_unrolled_deeply_nested():
    sdfg = create_deeply_nested_sdfg()
    passed = np.full((4, 100), 42.0, dtype=np.float32)
    returns = np.zeros((4, 100), np.float32)
    sdfg(x=passed, y=returns)
    expected = passed
    expected[1, 0] = 15.0
    expected[0, 0] = 15.0
    assert (np.allclose(expected, returns, 1e-6))


def create_simple_unrolled_sdfg():
    @dace.program
    def ucopy(input: dace.float32[4], output: dace.float32[4]):
        for i in dace.map[0:4]:
            output[i] = input[i]

    sdfg = ucopy.to_sdfg()
    for node in sdfg.states()[0].nodes():
        if (isinstance(node, dace.sdfg.nodes.MapEntry)):
            node.schedule = dace.ScheduleType.Unrolled
    return sdfg


def test_unrolled_simple_map():
    sdfg = create_simple_unrolled_sdfg()
    passed = np.full((4), 42.0, dtype=np.float32)
    returns = np.zeros((4), np.float32)
    sdfg(input=passed, output=returns)
    assert (np.allclose(passed, returns, 1e-6))


if __name__ == "__main__":
    test_unrolled_deeply_nested()
    test_unrolled_simple_map()
