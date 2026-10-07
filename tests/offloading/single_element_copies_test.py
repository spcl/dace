# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A single-element copy moves into the map that reads it, unless the map takes it as something other than data."""

import dace
from dace import Memlet
from dace.transformation.passes.offloading.single_element import single_element_copies_into_map

LENGTH = 16


def copy_into_map(dynamic_range: bool) -> tuple[dace.SDFG, dace.nodes.MapEntry]:
    """``L[0] -> s -> map``, where ``s`` is the extent of the map (a dynamic input) or a value its tasklet reads."""
    sdfg = dace.SDFG(f"copy_into_map_{dynamic_range}")
    sdfg.add_array("L", [1], dace.int32)
    sdfg.add_array("out", [LENGTH], dace.float64)
    sdfg.add_scalar("s", dace.int32, transient=True)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:n" if dynamic_range else f"0:{LENGTH}"})
    scalar = state.add_access("s")
    state.add_edge(state.add_read("L"), None, scalar, None, Memlet("L[0]"))
    if dynamic_range:
        entry.add_in_connector("n")
        state.add_edge(scalar, None, entry, "n", Memlet("s[0]"))
        tasklet = state.add_tasklet("t", {}, {"o"}, "o = 1.0")
        state.add_edge(entry, None, tasklet, None, Memlet())
    else:
        tasklet = state.add_tasklet("t", {"v"}, {"o"}, "o = v")
        entry.add_in_connector("IN_s")
        entry.add_out_connector("OUT_s")
        state.add_edge(scalar, None, entry, "IN_s", Memlet("s[0]"))
        state.add_edge(entry, "OUT_s", tasklet, "v", Memlet("s[0]"))
    exit_.add_in_connector("IN_out")
    exit_.add_out_connector("OUT_out")
    state.add_edge(tasklet, "o", exit_, "IN_out", Memlet("out[i]"))
    state.add_edge(exit_, "OUT_out", state.add_write("out"), None, Memlet(f"out[0:{LENGTH}]"))
    sdfg.validate()
    return sdfg, entry


def test_a_copy_read_through_a_pass_through_connector_moves_into_the_map():
    sdfg, entry = copy_into_map(dynamic_range=False)

    single_element_copies_into_map(sdfg)

    sdfg.validate()
    (edge,) = [e for e in sdfg.start_state.in_edges(entry) if not e.data.is_empty()]
    assert isinstance(edge.src, dace.nodes.AccessNode) and edge.src.data == "L", edge.src


def test_a_copy_feeding_a_dynamic_map_input_stays_where_it_is():
    """Moving it drops the connector the map range names, which turns ``n`` into a free symbol."""
    sdfg, entry = copy_into_map(dynamic_range=True)

    single_element_copies_into_map(sdfg)

    assert "n" in entry.in_connectors, sorted(entry.in_connectors)
    (edge,) = [e for e in sdfg.start_state.in_edges(entry) if e.dst_conn == "n"]
    assert edge.src.data == "s", edge.src


if __name__ == "__main__":
    test_a_copy_read_through_a_pass_through_connector_moves_into_the_map()
    test_a_copy_feeding_a_dynamic_map_input_stays_where_it_is()
