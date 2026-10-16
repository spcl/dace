# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host levels below the top: a map kept on the host launches the kernels in it, and a nested SDFG at a host
level places its own data."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol("N")


def row_sums_with_a_device_reduce() -> dace.SDFG:
    """A map over rows around a reduce whose chosen expansion is a host-issued cub call."""
    sdfg = dace.SDFG("row_sums_with_a_device_reduce")
    sdfg.add_array("A", [8, 64], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    state = sdfg.add_state("rows")
    entry, exit_node = state.add_map("rows", {"i": "0:8"})
    reduce = state.add_reduce("lambda a, b: a + b", None, 0)
    reduce.implementation = "CUDA (device)"
    state.add_memlet_path(state.add_read("A"), entry, reduce, memlet=dace.Memlet("A[i, 0:64]"))
    state.add_memlet_path(reduce, exit_node, state.add_write("out"), memlet=dace.Memlet("out[i]"))
    sdfg.validate()
    return sdfg


def test_a_map_around_a_host_issued_library_call_launches_it_from_the_host():
    """Inside a kernel the cub device reduce did not compile; the map now stays host code around the call."""
    sdfg = row_sums_with_a_device_reduce()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    schedules = {
        node.label: node.schedule
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, (dace.nodes.MapEntry, dace.nodes.LibraryNode))
    }
    assert schedules == {"rows": dtypes.ScheduleType.Sequential, "Reduce": dtypes.ScheduleType.GPU_Device}
    state = next(state for state in sdfg.states() if state.label == "rows")
    reduce = next(node for node in state.nodes() if isinstance(node, dace.nodes.LibraryNode))
    operands = [edge.data.data for edge in state.memlet_path(state.in_edges(reduce)[0])]
    assert all(sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global for name in operands), operands


def test_a_map_around_an_in_kernel_reduce_is_the_kernel():
    """A cub block reduce expands to device code, so the map around it is the kernel."""
    sdfg = row_sums_with_a_device_reduce()
    next(
        node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.LibraryNode)
    ).implementation = "CUDA (block)"
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    rows = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry))
    assert rows.map.schedule == dtypes.ScheduleType.GPU_Device


@pytest.mark.gpu
def test_a_host_launched_device_reduce_computes_what_numpy_computes():
    sdfg = row_sums_with_a_device_reduce()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    A = np.random.default_rng(0).random((8, 64))
    out = np.zeros(8)
    sdfg(A=A, out=out)
    np.testing.assert_allclose(out, A.sum(axis=1))


@dace.program
def double(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = x[i] * 2.0


@dace.program
def calls_double(x: dace.float64[N], y: dace.float64[N]):
    double(x, y)


def test_a_nested_sdfg_at_the_top_level_is_a_host_level_of_its_own():
    """Its map is the kernel, and the arrays it binds reach it in device memory."""
    sdfg = calls_double.to_sdfg(simplify=False)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    nested = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG))
    kernels = [node for node, _ in nested.sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert kernels and all(entry.map.schedule == dtypes.ScheduleType.GPU_Device for entry in kernels)
    assert all(
        desc.storage == dtypes.StorageType.GPU_Global for name, desc in nested.sdfg.arrays.items() if not desc.transient
    ), nested.sdfg.arrays


@pytest.mark.gpu
def test_a_nested_host_level_computes_what_numpy_computes():
    sdfg = calls_double.to_sdfg(simplify=False)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    x = np.arange(16, dtype=np.float64)
    y = np.zeros(16)
    sdfg(x=x, y=y, N=16)
    np.testing.assert_allclose(y, x * 2.0)


def body_reading_a_device_element_on_the_host() -> dace.SDFG:
    """``A`` handed over in device memory as the connector ``s`` of a nested SDFG whose host loop adds ``s[0]`` to
    every ``out[i]`` (No-View nested SDFGs pass the whole array, read at its element inside)."""
    body = dace.SDFG("add_on_the_host")
    body.add_array("s", [8], dace.float64)
    body.add_array("out", [8], dace.float64)
    loop = dace.sdfg.state.LoopRegion("over_out", "i < 8", "i", "i = 0", "i = i + 1")
    body.add_node(loop, is_start_block=True)
    step = loop.add_state("add", is_start_block=True)
    add = step.add_tasklet("add", {"o_in": None, "s_in": None}, {"o": None}, "o = o_in + s_in")
    step.add_edge(step.add_read("out"), None, add, "o_in", dace.Memlet("out[i]"))
    step.add_edge(step.add_read("s"), None, add, "s_in", dace.Memlet("s[0]"))
    step.add_edge(add, "o", step.add_write("out"), None, dace.Memlet("out[i]"))

    sdfg = dace.SDFG("body_reading_a_device_element_on_the_host")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    state = sdfg.add_state("call")
    call = state.add_nested_sdfg(body, {"s": None, "out": None}, {"out": None})
    state.add_edge(state.add_read("A"), None, call, "s", dace.Memlet("A[0:8]"))
    state.add_edge(state.add_read("out"), None, call, "out", dace.Memlet("out[0:8]"))
    state.add_edge(call, "out", state.add_write("out"), None, dace.Memlet("out[0:8]"))
    sdfg.validate()
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


def test_a_device_element_a_body_reads_on_the_host_is_staged_on_the_host():
    """npbench azimint_hist: a scalar connector bound to device memory and read by host code in the body."""
    sdfg = body_reading_a_device_element_on_the_host()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    state = next(state for state in sdfg.states() if state.label == "call")
    call = next(node for node in state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    bound = [edge.data.data for edge in state.in_edges(call) if edge.dst_conn == "s"]
    assert bound and all(sdfg.arrays[name].storage == dtypes.StorageType.Default for name in bound), bound


@pytest.mark.gpu
def test_a_device_element_staged_for_a_host_body_computes_what_numpy_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    sdfg = body_reading_a_device_element_on_the_host()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    A = cupy.asarray(np.arange(1, 9, dtype=np.float64))
    out = cupy.zeros(8)
    sdfg(A=A, out=out)
    np.testing.assert_allclose(out.get(), np.full(8, 1.0))


def rows_scaled_by_a_device_element() -> dace.SDFG:
    """A host map over rows around a nested SDFG that reads the scalar connector ``s`` on the host (an interstate
    assignment) and scales its row in a kernel. ``s`` is bound through the map's entry to ``t[0]``, a transient a
    kernel before the map fills, so it lives in device memory (cp2k_grid_integrate's ``border_width_index``)."""
    body = dace.SDFG("scale_row")
    body.add_scalar("s", dace.float64)
    body.add_array("out", [4, 8], dace.float64)
    body.add_symbol("j", dace.int64)
    body.add_symbol("n", dace.float64)
    first = body.add_state("first", is_start_block=True)
    work = body.add_state("work")
    body.add_edge(first, work, dace.InterstateEdge(assignments={"n": "s"}))
    work.add_mapped_tasklet(
        "scale",
        {"i": "0:8"},
        {"o_in": dace.Memlet("out[j, i]")},
        "o = o_in * n",
        {"o": dace.Memlet("out[j, i]")},
        external_edges=True,
    )

    sdfg = dace.SDFG("rows_scaled_by_a_device_element")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("out", [4, 8], dace.float64)
    sdfg.add_array("t", [1], dace.float64, transient=True)
    init = sdfg.add_state("init")
    init.add_mapped_tasklet(
        "total",
        {"k": "0:4"},
        {"a": dace.Memlet("A[k]")},
        "r = a",
        {"r": dace.Memlet("t[0]", wcr="lambda x, y: x + y")},
        external_edges=True,
    )
    state = sdfg.add_state_after(init, "rows")
    entry, exit_node = state.add_map("rows", {"j": "0:4"})
    call = state.add_nested_sdfg(body, {"s": None, "out": None}, {"out": None}, symbol_mapping={"j": "j"})
    state.add_memlet_path(state.add_read("t"), entry, call, dst_conn="s", memlet=dace.Memlet("t[0]", dynamic=True))
    state.add_memlet_path(state.add_read("out"), entry, call, dst_conn="out", memlet=dace.Memlet("out[0:4, 0:8]"))
    state.add_memlet_path(call, exit_node, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:4, 0:8]"))
    sdfg.validate()
    for name in ("A", "out"):
        sdfg.arrays[name].storage = dtypes.StorageType.GPU_Global
    return sdfg


def test_a_device_element_bound_through_a_taskloop_is_staged_once_outside_it():
    """Staging used to read a fresh access node outside the map with the map's ``OUT_t`` as its connector
    and drop the map's own edge, leaving ``OUT_t`` dangling ("Dangling out-connector", cp2k_grid_integrate).
    The element is now copied to the host once, at the access node the binding starts from, and the host
    copy -- not the device scalar -- enters the host map."""
    sdfg = rows_scaled_by_a_device_element()
    ppl.Pipeline([OffloadToAccelerator(taskloop_overrides={"rows": True})]).apply_pass(sdfg, {})
    sdfg.validate()
    state = next(state for state in sdfg.states() if state.label == "rows")
    entry = next(node for node in state.nodes() if isinstance(node, dace.nodes.MapEntry))
    assert entry.map.schedule == dtypes.ScheduleType.Sequential, "the taskloop stays on the host"
    call = next(node for node in state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    (binding,) = [edge for edge in state.in_edges(call) if edge.dst_conn == "s"]
    staged = binding.data.data
    assert sdfg.arrays[staged].storage == dtypes.StorageType.Default, staged
    (root,) = [edge for edge in state.in_edges(entry) if edge.data.data == staged]
    assert isinstance(root.src, dace.nodes.AccessNode), "the copy sits outside the map"
    assert [edge.src.data for edge in state.in_edges(root.src)] == ["t"], "copied from the device element"
    assert not any(edge.data.data == "t" for edge in state.in_edges(entry)), "no device scalar enters the host map"


@pytest.mark.gpu
def test_a_device_element_bound_through_a_taskloop_computes_what_numpy_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    sdfg = rows_scaled_by_a_device_element()
    ppl.Pipeline([OffloadToAccelerator(taskloop_overrides={"rows": True})]).apply_pass(sdfg, {})
    A = cupy.asarray(np.arange(1, 5, dtype=np.float64))
    out = cupy.ones((4, 8))
    sdfg(A=A, out=out)
    np.testing.assert_allclose(out.get(), np.full((4, 8), 10.0))


if __name__ == "__main__":
    test_a_map_around_a_host_issued_library_call_launches_it_from_the_host()
    test_a_map_around_an_in_kernel_reduce_is_the_kernel()
    test_a_nested_sdfg_at_the_top_level_is_a_host_level_of_its_own()
    test_a_device_element_a_body_reads_on_the_host_is_staged_on_the_host()
    test_a_device_element_bound_through_a_taskloop_is_staged_once_outside_it()
