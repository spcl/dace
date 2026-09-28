# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace import nodes
from dace.sdfg import utils


def test_traverse_sdfg_with_defined_symbols() -> None:
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("my_symbol", dace.int32)

    start = sdfg.add_state("start", is_start_block=True)
    start.add_tasklet("noop", set(), set(), "")
    sdfg.add_state_after(start, "next")

    for _state, _node, defined_symbols in utils.traverse_sdfg_with_defined_symbols(sdfg):
        assert "my_symbol" in defined_symbols


def test_get_view_node() -> None:

    def repro() -> tuple[dace.SDFGState, nodes.AccessNode, nodes.AccessNode]:
        sdfg = dace.SDFG("tester")
        a_name, a_desc = sdfg.add_array("A", [10, 20], dace.float32)
        b_name, b_desc = sdfg.add_array("B", [10, 20], dace.float32)
        scalar_name, scalar_desc = sdfg.add_scalar("scalar", dace.float32)

        state = sdfg.add_state("start_state", is_start_block=True)
        A_read = state.add_read(a_name)
        scalar_read = state.add_read("scalar")
        map_entry_i, map_exit_i = state.add_map("i_map", {"__i": dace.subsets.Range.from_string("0:10")})
        map_entry_j, map_exit_j = state.add_map("j_map", {"__j": dace.subsets.Range.from_string("0:20")})
        tasklet = state.add_tasklet(
            "increment",
            inputs={
                "a": None,
                "inc": None
            },
            outputs={"b": None},
            code="b = a + inc",
        )
        B_write = state.add_write(b_name)

        # old
        # state.add_memlet_path(A_read, map_entry_i, map_entry_j, tasklet, dst_conn="a", memlet=dace.Memlet("A[__i, __j]"))

        # new with indirection in A_view__i
        _, av__i_desc = sdfg.add_view("Av__i", [20], dace.float32)
        av__i = state.add_read("Av__i")

        map_entry_i.add_in_connector("IN_A")
        map_entry_i.add_out_connector("OUT_A")
        map_entry_j.add_in_connector("IN_A")
        map_entry_j.add_out_connector("OUT_A")

        state.add_edge(A_read, None, map_entry_i, "IN_A", memlet=dace.Memlet.from_array("A", a_desc))
        state.add_edge(map_entry_i, "OUT_A", av__i, None, dace.Memlet("A[__i, 0]"))
        state.add_edge(av__i, None, map_entry_j, "IN_A", dace.Memlet.from_array("Av__i", av__i_desc))
        state.add_edge(map_entry_j, "OUT_A", tasklet, "a", dace.Memlet("Av__i[__j]"))

        state.add_memlet_path(
            scalar_read,
            map_entry_i,
            map_entry_j,
            tasklet,
            dst_conn="inc",
            memlet=dace.Memlet.from_array(scalar_name, scalar_desc),
        )
        state.add_memlet_path(tasklet, map_exit_j, map_exit_i, B_write, src_conn="b", memlet=dace.Memlet("B[__i, __j]"))

        return state, av__i, A_read

    state, view, expected = repro()

    viewed = utils.get_view_node(state, view)
    assert isinstance(viewed, nodes.AccessNode)
    assert viewed == expected


if __name__ == "__main":
    test_traverse_sdfg_with_defined_symbols()
    test_get_view_node()
