# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import pytest

import dace
from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGInterstateEdgeError


@pytest.mark.gpu
def test_gpu_access_on_host_interstate_ok():
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map("map", dict(i="0:20"), dace.ScheduleType.GPU_Device)

    nsdfg = dace.SDFG("inner")
    nsdfg.add_array("a", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state1 = nsdfg.add_state()
    state2 = nsdfg.add_state()
    nsdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(s="a[i]")))

    nnode = state.add_nested_sdfg(nsdfg, {"a"}, {}, {"i": "i"})
    r = state.add_read("A")
    state.add_memlet_path(r, me, nnode, dst_conn="a", memlet=dace.Memlet("A[0:20]"))
    state.add_nedge(nnode, mx, dace.Memlet())

    sdfg.validate()


@pytest.mark.gpu
def test_gpu_access_on_host_interstate_invalid():
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(s="A[4]")))

    with pytest.raises(InvalidSDFGInterstateEdgeError):
        sdfg.validate()


@pytest.mark.gpu
def test_gpu_access_on_host_tasklet():

    @dace.program
    def tester(a: dace.float64[20] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:20] @ dace.ScheduleType.CPU_Multicore:
            a[i] = 1

    with pytest.raises(InvalidSDFGEdgeError):
        tester.to_sdfg(validate=True)


@pytest.mark.gpu
def test_gpu_access_on_device_interstate_edge_default():
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()

    me, mx = state.add_map("test", dict(i="0:20"))

    nsdfg = dace.SDFG("nester")
    nsdfg.add_array("A", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state1 = nsdfg.add_state()
    state2 = nsdfg.add_state()
    nsdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(s="A[4]")))

    nsdfg_node = state.add_nested_sdfg(nsdfg, {"A"}, {})
    state.add_memlet_path(state.add_read("A"), me, nsdfg_node, dst_conn="A", memlet=dace.Memlet("A[0:20]"))
    state.add_nedge(nsdfg_node, mx, dace.Memlet())

    sdfg.validate()


def test_an_invalid_edge_inside_a_loop_is_reported_at_that_edge():
    """The edge's index is its index among the edges of the region that OWNS it. Looked up among the
    SDFG's own edges -- none here, the loop is the only block -- formatting the message raised
    ``IndexError`` instead of naming the edge (HB kmp's GPU render), and with more top-level edges it
    named the wrong one."""
    sdfg = dace.SDFG("loop_reading_device_memory_on_the_host")
    sdfg.add_array("A", [20], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_symbol("s", dace.float64)
    loop = dace.sdfg.state.LoopRegion("over_i", "i < 20", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    second = loop.add_state("second")
    third = loop.add_state("third")
    loop.add_edge(first, second, dace.InterstateEdge())
    loop.add_edge(second, third, dace.InterstateEdge(assignments=dict(s="A[i]")))

    with pytest.raises(InvalidSDFGInterstateEdgeError) as raised:
        sdfg.validate()
    assert "(at edge second -> third)" in str(raised.value), str(raised.value)
    assert raised.value.to_json()["cfg_id"] == loop.cfg_id


if __name__ == "__main__":
    test_gpu_access_on_host_interstate_ok()
    test_gpu_access_on_host_interstate_invalid()
    test_gpu_access_on_host_tasklet()
    test_gpu_access_on_device_interstate_edge_default()
    test_an_invalid_edge_inside_a_loop_is_reported_at_that_edge()
