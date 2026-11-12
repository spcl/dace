# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.sdfg.dealias import convert_legacy_nested_sdfgs
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


def test_unrolled_copies_of_a_multistate_body_are_inlined():
    """Each unrolled copy of a body with two states ends at the top level of the state, where it is inlined."""
    inner = dace.SDFG("two_state_body")
    inner.add_array("a", [1], dace.float64)
    inner.add_array("b", [1], dace.float64)
    inner.add_scalar("t", dace.float64, transient=True)
    first = inner.add_state("first")
    task = first.add_tasklet("double", {"inp"}, {"res"}, "res = 2 * inp")
    first.add_edge(first.add_read("a"), None, task, "inp", dace.Memlet("a[0]"))
    first.add_edge(task, "res", first.add_write("t"), None, dace.Memlet("t[0]"))
    second = inner.add_state_after(first, "second")
    task = second.add_tasklet("inc", {"inp"}, {"res"}, "res = inp + 1")
    second.add_edge(second.add_read("t"), None, task, "inp", dace.Memlet("t[0]"))
    second.add_edge(task, "res", second.add_write("b"), None, dace.Memlet("b[0]"))

    sdfg = dace.SDFG("unroll_multistate_body")
    sdfg.add_array("A", [3], dace.float64)
    sdfg.add_array("B", [3], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:3"})
    body = state.add_nested_sdfg(inner, {"a"}, {"b"})
    state.add_memlet_path(state.add_read("A"), entry, body, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(body, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    convert_legacy_nested_sdfgs(sdfg)
    sdfg.validate()

    assert sdfg.apply_transformations(MapUnroll) == 1
    sdfg.validate()
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    a = np.arange(3, dtype=np.float64)
    b = np.zeros(3)
    sdfg(A=a, B=b)
    assert np.array_equal(b, 2 * a + 1)


if __name__ == "__main__":
    test_map_unroll()
    test_gpu_map_declined_unless_permissive()
    test_unrolled_copies_of_a_multistate_body_are_inlined()
