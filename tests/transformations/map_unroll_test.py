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


def test_gpu_map_declined_unless_permissive():
    """A GPU map is declined by default (a device map IS the kernel launch) and offered under permissive."""
    sdfg = dace.SDFG("gpu_unroll")
    sdfg.add_array("A", [4], dace.float64, transient=True)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "fill",
        dict(i="0:4"),
        {},
        "out = 1.0",
        dict(out=dace.Memlet("A[i]")),
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry = next(n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry))

    assert not MapUnroll.can_be_applied_to(sdfg, map_entry=entry)
    assert MapUnroll.can_be_applied_to(sdfg, permissive=True, map_entry=entry)

    # Same map on a CPU schedule is offered either way -- the GPU schedule alone is the precondition.
    entry.map.schedule = dace.ScheduleType.Sequential
    assert MapUnroll.can_be_applied_to(sdfg, map_entry=entry)


def test_an_array_the_map_shares_with_a_computation_beside_it_keeps_feeding_that_computation():
    """``A`` feeds the unrolled map and a tasklet beside it; the tasklet still reads ``A`` once the map is unrolled."""
    sdfg = dace.SDFG("unroll_shared_read")
    sdfg.add_array("A", [3], dace.float64)
    sdfg.add_array("B", [3], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_read("A")
    entry, leave = state.add_map("double", {"i": "0:3"})
    twice = state.add_tasklet("twice", {"x"}, {"y"}, "y = 2 * x")
    state.add_memlet_path(a, entry, twice, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(twice, leave, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = x")
    state.add_edge(a, None, first, "x", dace.Memlet("A[0]"))
    state.add_edge(first, "y", state.add_write("C"), None, dace.Memlet("C[0]"))

    assert sdfg.apply_transformations(MapUnroll) == 1

    sdfg.validate()
    assert state.in_degree(first) == 1 and not any(isinstance(n, dace.sdfg.nodes.MapEntry) for n in state.nodes())
    A, B, C = np.array([1.0, 2.0, 3.0]), np.zeros(3), np.zeros(1)
    sdfg(A=A, B=B, C=C)
    assert np.array_equal(B, 2 * A) and C[0] == 1.0


if __name__ == "__main__":
    test_map_unroll()
    test_gpu_map_declined_unless_permissive()
    test_an_array_the_map_shares_with_a_computation_beside_it_keeps_feeding_that_computation()
