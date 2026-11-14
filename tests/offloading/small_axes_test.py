# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel never puts a SMALL axis on its grid.

``nblks`` is declared ``SMALL``: a handful of blocks, much smaller than ``nproma`` and ``nlev``. A map over both kinds
becomes a ``GPU_Device`` map over the other axes, in their order, around a sequential map over the ``SMALL`` ones. The
structure is all these tests check, so they need no GPU.
"""

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.symbolic import Scale
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import GPUTransformMap
from dace.transformation.passes.offloading import OffloadToAccelerator

GPU = dtypes.ScheduleType.GPU_Device
SEQ = dtypes.ScheduleType.Sequential


def blocked_sdfg(name: str, small: bool = True) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("nblks", dace.int64, scale=Scale.SMALL if small else Scale.DEFAULT)
    sdfg.add_symbol("nproma", dace.int64)
    sdfg.add_symbol("nlev", dace.int64)
    return sdfg


def mapped_copy(sdfg: dace.SDFG, shape: list[str], ranges: dict[str, str], index: str) -> dace.SDFG:
    """``B[index] = 2 * A[index]`` over ``ranges``, in one map."""
    sdfg.add_array("A", shape, dace.float64)
    sdfg.add_array("B", shape, dace.float64)
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "scale",
        ranges,
        {"x": dace.Memlet(f"A[{index}]")},
        "y = 2.0 * x",
        {"y": dace.Memlet(f"B[{index}]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def offload(sdfg: dace.SDFG) -> dict[tuple[str, ...], dtypes.ScheduleType]:
    """Offload ``sdfg`` and return each map's parameters with the schedule it got."""
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return {
        tuple(node.map.params): node.map.schedule
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    }


def map_nest(sdfg: dace.SDFG) -> list[list[str]]:
    """The parameters of the maps in the only compute state, outermost first, for a linear map nest."""
    state = next(s for s in sdfg.states() if s.label == "compute")
    scopes = state.scope_dict()
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    return [e.map.params for e in sorted(entries, key=lambda e: depth(scopes, e))]


def depth(scopes: dict, node: nodes.Node) -> int:
    level = 0
    while scopes[node] is not None:
        node = scopes[node]
        level += 1
    return level


def test_a_block_axis_leaves_the_grid_for_a_sequential_inner_map():
    sdfg = mapped_copy(blocked_sdfg("block_outer"), ["nblks", "nproma"], {"b": "0:nblks", "i": "0:nproma"}, "b, i")
    assert offload(sdfg) == {("i",): GPU, ("b",): SEQ}
    assert map_nest(sdfg) == [["i"], ["b"]]


def test_the_large_axes_keep_their_order_on_the_grid():
    sdfg = mapped_copy(
        blocked_sdfg("block_middle"),
        ["nlev", "nblks", "nproma"],
        {"k": "0:nlev", "b": "0:nblks", "i": "0:nproma"},
        "k, b, i",
    )
    assert offload(sdfg) == {("k", "i"): GPU, ("b",): SEQ}
    assert map_nest(sdfg) == [["k", "i"], ["b"]]


def test_two_block_axes_share_one_sequential_map():
    sdfg = blocked_sdfg("two_blocks")
    sdfg.add_symbol("nbat", dace.int64, scale=Scale.SMALL)
    mapped_copy(
        sdfg,
        ["nbat", "nproma", "nblks"],
        {"t": "0:nbat", "i": "0:nproma", "b": "0:nblks"},
        "t, i, b",
    )
    assert offload(sdfg) == {("i",): GPU, ("t", "b"): SEQ}
    assert map_nest(sdfg) == [["i"], ["t", "b"]]


def test_a_block_map_inside_a_kernel_stays_sequential():
    sdfg = blocked_sdfg("block_inside")
    sdfg.add_array("A", ["nproma", "nblks"], dace.float64)
    sdfg.add_array("B", ["nproma", "nblks"], dace.float64)
    state = sdfg.add_state("compute")
    outer_entry, outer_exit = state.add_map("rows", {"i": "0:nproma"})
    inner_entry, inner_exit = state.add_map("blocks", {"b": "0:nblks"})
    tasklet = state.add_tasklet("scale", {"a"}, {"o"}, "o = 2.0 * a")
    state.add_memlet_path(
        state.add_read("A"), outer_entry, inner_entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i, b]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("B"), src_conn="o", memlet=dace.Memlet("B[i, b]")
    )
    sdfg.validate()
    assert offload(sdfg) == {("i",): GPU, ("b",): SEQ}


def test_a_kernel_over_block_axes_only_stays_a_kernel():
    """With no other axis there is no other grid: the map over ``nblks`` alone is still the kernel."""
    sdfg = mapped_copy(blocked_sdfg("blocks_only"), ["nblks"], {"b": "0:nblks"}, "b")
    assert offload(sdfg) == {("b",): GPU}


def block_map_around(sdfg: dace.SDFG, inner_range: str) -> dace.SDFG:
    """``B = 2 * A`` in a map over ``b`` around a map over ``i``."""
    sdfg.add_array("A", ["nblks", "nproma"], dace.float64)
    sdfg.add_array("B", ["nblks", "nproma"], dace.float64)
    state = sdfg.add_state("compute")
    outer_entry, outer_exit = state.add_map("blocks", {"b": "0:nblks"})
    inner_entry, inner_exit = state.add_map("rows", {"i": inner_range})
    tasklet = state.add_tasklet("scale", {"x"}, {"y"}, "y = 2.0 * x")
    state.add_memlet_path(
        state.add_read("A"), outer_entry, inner_entry, tasklet, dst_conn="x", memlet=dace.Memlet("A[b, i]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[b, i]")
    )
    sdfg.validate()
    return sdfg


def test_a_block_map_around_a_row_map_launches_over_the_rows():
    """Collapsed with the map it holds, the block map leaves the grid like any other block axis."""
    sdfg = block_map_around(blocked_sdfg("block_around_rows"), "0:nproma")
    assert offload(sdfg) == {("i",): GPU, ("b",): SEQ}
    assert map_nest(sdfg) == [["i"], ["b"]]


def test_a_row_map_whose_range_reads_the_block_keeps_the_block_kernel():
    """Rows that depend on the block cannot collapse with it: the block map stays the kernel."""
    sdfg = block_map_around(blocked_sdfg("block_dependent_rows"), "0:nproma - b")
    assert offload(sdfg) == {("b",): GPU, ("i",): SEQ}


def test_without_small_symbols_the_kernel_keeps_every_axis():
    sdfg = mapped_copy(
        blocked_sdfg("no_small", small=False), ["nblks", "nproma"], {"b": "0:nblks", "i": "0:nproma"}, "b, i"
    )
    assert offload(sdfg) == {("b", "i"): GPU}


def test_a_short_constant_axis_leaves_the_grid_like_a_small_symbol():
    """A literal extent below the small-extent bound is as small as a SMALL symbol for a schedule."""
    sdfg = mapped_copy(blocked_sdfg("constant_axis"), ["4", "nproma"], {"c": "0:4", "i": "0:nproma"}, "c, i")
    assert offload(sdfg) == {("c",): SEQ, ("i",): GPU}


def test_a_long_constant_axis_stays_in_the_grid():
    sdfg = mapped_copy(blocked_sdfg("long_constant_axis"), ["128", "nproma"], {"c": "0:128", "i": "0:nproma"}, "c, i")
    assert offload(sdfg) == {("c", "i"): GPU}


def test_a_block_axis_of_a_nested_kernel_leaves_the_grid():
    """A nested SDFG at the host level holds kernels of its own, with the SMALL scale mapped in from outside."""
    outer = blocked_sdfg("nested_outer")
    outer.add_array("A", ["nblks", "nproma"], dace.float64)
    outer.add_array("B", ["nblks", "nproma"], dace.float64)
    inner = dace.SDFG("nested_inner")
    inner.add_symbol("nb", dace.int64)
    inner.add_symbol("np", dace.int64)
    mapped_copy(inner, ["nb", "np"], {"b": "0:nb", "i": "0:np"}, "b, i")
    state = outer.add_state("host")
    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"B"}, symbol_mapping={"nb": "nblks", "np": "nproma"})
    state.add_edge(state.add_read("A"), None, nsdfg, "A", dace.Memlet("A[0:nblks, 0:nproma]"))
    state.add_edge(nsdfg, "B", state.add_write("B"), None, dace.Memlet("B[0:nblks, 0:nproma]"))
    outer.validate()
    assert offload(outer) == {("i",): GPU, ("b",): SEQ}
    assert map_nest(inner) == [["i"], ["b"]]


def test_gpu_transform_map_keeps_a_block_axis_off_the_grid():
    """``GPUTransformMap`` offloads the map as a nested SDFG of its own; the scale still comes from the root."""
    sdfg = mapped_copy(blocked_sdfg("transform_map"), ["nblks", "nproma"], {"b": "0:nblks", "i": "0:nproma"}, "b, i")
    entry = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    GPUTransformMap.apply_to(sdfg, map_entry=entry)
    sdfg.validate()
    schedules = {
        tuple(node.map.params): node.map.schedule
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    }
    # ``Default`` under a kernel is sequential: the transformation unschedules the maps inside its kernels.
    assert schedules == {("i",): GPU, ("b",): dtypes.ScheduleType.Default}


if __name__ == "__main__":
    test_a_block_axis_leaves_the_grid_for_a_sequential_inner_map()
    test_the_large_axes_keep_their_order_on_the_grid()
    test_two_block_axes_share_one_sequential_map()
    test_a_block_map_inside_a_kernel_stays_sequential()
    test_a_kernel_over_block_axes_only_stays_a_kernel()
    test_a_block_map_around_a_row_map_launches_over_the_rows()
    test_a_row_map_whose_range_reads_the_block_keeps_the_block_kernel()
    test_without_small_symbols_the_kernel_keeps_every_axis()
    test_a_short_constant_axis_leaves_the_grid_like_a_small_symbol()
    test_a_long_constant_axis_stays_in_the_grid()
    test_a_block_axis_of_a_nested_kernel_leaves_the_grid()
    test_gpu_transform_map_keeps_a_block_axis_off_the_grid()
