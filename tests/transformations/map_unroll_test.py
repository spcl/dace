# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.transformation.dataflow import MapUnroll


def test_map_unroll():

    sdfg = dace.SDFG("unroll_schedule")
    sdfg.add_array("input_array", [3, 2, 1], dace.int32)
    sdfg.add_array("output_array", [1, 2, 3], dace.int32)
    outer_state = sdfg.add_state("unroll_schedule")

    read = outer_state.add_read("input_array")
    write = outer_state.add_write("output_array")
    entry_outer, exit_outer = outer_state.add_map("map_outer", {"i": "0:3", "j": "0:1"})
    entry_inner, exit_inner = outer_state.add_map("map_inner", {"k": "0:2"})
    nsdfg = dace.SDFG("unroll_nested")
    nsdfg_node = outer_state.add_nested_sdfg(nsdfg, {"x"}, {"y"})
    outer_state.add_memlet_path(
        read, entry_outer, entry_inner, nsdfg_node, dst_conn="x", memlet=dace.Memlet("input_array[i, k, j]")
    )
    outer_state.add_memlet_path(
        nsdfg_node, exit_inner, exit_outer, write, src_conn="y", memlet=dace.Memlet("output_array[j, k, i]")
    )

    nsdfg.add_array("x", [1], dace.int32)
    nsdfg.add_array("y", [1], dace.int32)
    inner_state = nsdfg.add_state("unroll_nested")
    tasklet = inner_state.add_tasklet("tasklet", {"_x"}, {"_y"}, "_y = _x + 1")
    read = inner_state.add_read("x")
    write = inner_state.add_write("y")
    inner_state.add_memlet_path(read, tasklet, dst_conn="_x", memlet=dace.Memlet("x[0]"))
    inner_state.add_memlet_path(tasklet, write, src_conn="_y", memlet=dace.Memlet("y[0]"))

    nsdfg_node.integrate_into_parent()

    assert sdfg.apply_transformations_repeated(MapUnroll) == 4

    input_array = np.copy(np.arange(0, 6, dtype=np.int32).reshape((3, 2, 1)))
    output_array = np.empty((1, 2, 3), dtype=np.int32)
    sdfg(input_array=input_array, output_array=output_array)
    for i in range(3):
        for k in range(2):
            assert input_array[i, k, 0] + 1 == output_array[0, k, i]


def test_map_unroll_keeps_inputs_of_downstream_nodes():
    """A node after the map that also reads data the map does not read is in the same component."""
    sdfg = dace.SDFG("unroll_downstream_inputs")
    sdfg.add_array("A", [1], dace.int32)
    sdfg.add_array("B", [1], dace.int32)
    sdfg.add_array("C", [1], dace.int32)
    sdfg.add_transient("T", [1], dace.int32)
    state = sdfg.add_state()
    t = state.add_access("T")
    state.add_mapped_tasklet(
        "first",
        {"i": "0:1"},
        {"a": dace.Memlet("A[i]")},
        "t = a + 1",
        {"t": dace.Memlet("T[i]")},
        external_edges=True,
        output_nodes={"T": t},
    )
    second = state.add_tasklet("second", {"t", "b"}, {"c"}, "c = t + b")
    state.add_edge(t, None, second, "t", dace.Memlet("T[0]"))
    state.add_edge(state.add_read("B"), None, second, "b", dace.Memlet("B[0]"))
    state.add_edge(second, "c", state.add_write("C"), None, dace.Memlet("C[0]"))

    assert sdfg.apply_transformations(MapUnroll) == 1
    sdfg.validate()

    A = np.array([3], dtype=np.int32)
    B = np.array([5], dtype=np.int32)
    C = np.zeros(1, dtype=np.int32)
    sdfg(A=A, B=B, C=C)
    assert C[0] == 9


if __name__ == "__main__":
    test_map_unroll()
    test_map_unroll_keeps_inputs_of_downstream_nodes()
