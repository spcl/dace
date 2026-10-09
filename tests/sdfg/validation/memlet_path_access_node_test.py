# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace


def _producer_consumer(name: str):
    sdfg = dace.SDFG(name)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_scalar("tmp", dace.float64, transient=True)
    state = sdfg.add_state()
    producer = state.add_tasklet("producer", {}, {"out"}, "out = 1.0")
    consumer = state.add_tasklet("consumer", {"inp"}, {"out"}, "out = inp + 1.0")
    return sdfg, state, producer, consumer


def test_a_memlet_between_two_tasklets_is_rejected():
    sdfg, state, producer, consumer = _producer_consumer("memlet_between_two_tasklets")
    state.add_edge(producer, "out", consumer, "inp", dace.Memlet("tmp[0]"))
    state.add_edge(consumer, "out", state.add_write("B"), None, dace.Memlet("B[0]"))

    with pytest.raises(dace.sdfg.InvalidSDFGEdgeError, match="must be rooted at an AccessNode"):
        sdfg.validate()


def test_a_memlet_path_through_a_map_between_two_tasklets_is_rejected():
    """The rule holds for the whole tree, so a map scope between producer and consumer does not hide the missing
    AccessNode."""
    sdfg, state, producer, consumer = _producer_consumer("memlet_path_through_map_between_two_tasklets")
    map_entry, map_exit = state.add_map("m", dict(i="0:1"))
    state.add_memlet_path(producer, map_entry, consumer, src_conn="out", dst_conn="inp", memlet=dace.Memlet("tmp[0]"))
    state.add_memlet_path(consumer, map_exit, state.add_write("B"), src_conn="out", memlet=dace.Memlet("B[0]"))

    with pytest.raises(dace.sdfg.InvalidSDFGEdgeError, match="must be rooted at an AccessNode"):
        sdfg.validate()


def test_a_tree_rooted_at_a_tasklet_is_rejected_even_if_it_ends_at_an_access_node():
    """Ending at an AccessNode inside the map does not make the data live outside it: the root is the producer."""
    sdfg, state, producer, consumer = _producer_consumer("tree_rooted_at_a_tasklet")
    map_entry, map_exit = state.add_map("m", dict(i="0:1"))
    tmp = state.add_access("tmp")
    state.add_memlet_path(producer, map_entry, tmp, src_conn="out", memlet=dace.Memlet("tmp[0]"))
    state.add_edge(tmp, None, consumer, "inp", dace.Memlet("tmp[0]"))
    state.add_memlet_path(consumer, map_exit, state.add_write("B"), src_conn="out", memlet=dace.Memlet("B[0]"))

    with pytest.raises(dace.sdfg.InvalidSDFGEdgeError, match="must be rooted at an AccessNode"):
        sdfg.validate()


def test_a_value_routed_through_an_access_node_validates():
    sdfg, state, producer, consumer = _producer_consumer("value_routed_through_access_node")
    tmp = state.add_access("tmp")
    state.add_edge(producer, "out", tmp, None, dace.Memlet("tmp[0]"))
    state.add_edge(tmp, None, consumer, "inp", dace.Memlet("tmp[0]"))
    state.add_edge(consumer, "out", state.add_write("B"), None, dace.Memlet("B[0]"))

    sdfg.validate()


if __name__ == "__main__":
    test_a_memlet_between_two_tasklets_is_rejected()
    test_a_memlet_path_through_a_map_between_two_tasklets_is_rejected()
    test_a_tree_rooted_at_a_tasklet_is_rejected_even_if_it_ends_at_an_access_node()
    test_a_value_routed_through_an_access_node_validates()
