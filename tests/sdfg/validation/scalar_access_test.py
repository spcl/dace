# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace


def device_element_read_through_host_map(name: str, connector_type: dace.typeclass | None) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [1], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map("m", dict(i="0:1"), schedule=dace.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet("t", {"a": connector_type}, {"b"}, "b = 0.0")
    state.add_memlet_path(state.add_read("A"), map_entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[0]"))
    state.add_memlet_path(tasklet, map_exit, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[0]"))
    return sdfg


def test_a_pointer_connector_at_the_end_of_a_map_path_is_not_a_host_dereference():
    """The connector type is read off the tasklet the path ends at, not off the map entry it starts into."""
    device_element_read_through_host_map("pointer_connector_through_map", dace.pointer(dace.float64)).validate()


def test_a_scalar_connector_at_the_end_of_a_map_path_is_a_host_dereference():
    sdfg = device_element_read_through_host_map("scalar_connector_through_map", dace.float64)
    with pytest.raises(dace.sdfg.InvalidSDFGEdgeError, match="accessed on host"):
        sdfg.validate()


def test_a_device_element_passed_through_a_host_map_to_an_array_of_a_nested_sdfg_is_not_a_host_dereference():
    """Propagation narrows the edge to the one element the body uses, but the body binds an array: it is handed a
    pointer, and where the element is read is checked in the body."""
    body = dace.SDFG("body")
    body.add_array("A", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    body.add_array("B", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    body.add_state().add_mapped_tasklet(
        "kernel",
        {"k": "0:4"},
        {"a": dace.Memlet("A[2]")},
        "b = a",
        {"b": dace.Memlet("B[k]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    sdfg = dace.SDFG("device_element_through_host_map_to_nest")
    sdfg.add_array("A", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("B", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map("m", dict(i="0:1"), schedule=dace.ScheduleType.Sequential)
    nest = state.add_nested_sdfg(body, {"A"}, {"B"})
    state.add_memlet_path(state.add_read("A"), map_entry, nest, dst_conn="A", memlet=dace.Memlet("A[2]"))
    state.add_memlet_path(nest, map_exit, state.add_write("B"), src_conn="B", memlet=dace.Memlet("B[0:4]"))
    sdfg.validate()


if __name__ == "__main__":
    test_a_device_element_passed_through_a_host_map_to_an_array_of_a_nested_sdfg_is_not_a_host_dereference()
    test_a_pointer_connector_at_the_end_of_a_map_path_is_not_a_host_dereference()
    test_a_scalar_connector_at_the_end_of_a_map_path_is_a_host_dereference()
