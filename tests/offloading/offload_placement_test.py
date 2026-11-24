# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Where ``OffloadToAccelerator`` places data and copies: one graph per defect, plus its numeric companion."""

import numpy as np
import pytest
from ordered_set import OrderedSet

import dace
from dace import data, dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode

#: Extent of the ordering-edge graph.
LENGTH = 16


ROWS = dace.symbol("ROWS")


def cpu_heap_sdfg() -> dace.SDFG:
    """One map writing a non-transient array, with every descriptor on ``CPU_Heap``.

    That is the storage ``auto_optimize(DeviceType.CPU)`` leaves behind, and the shape the rename
    below was blind to.
    """
    sdfg = dace.SDFG("placement_cpu_heap_offload")
    for name in ("A", "B"):
        sdfg.add_array(name, [32], dace.float64, storage=dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("compute")
    state.add_mapped_tasklet(
        "double",
        {"i": "0:32"},
        {"a": dace.Memlet("A[i]")},
        "b = a * 2.0",
        {"b": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


#: Blocks in a chain, comfortably past CPython's 1000-frame default. CloudSC canonicalized for the
#: GPU carries about twice this, which is the graph that produced the failure below.


def test_a_tail_contributes_none_of_its_remaining_siblings():
    """The recursive walk stopped at the FIRST child that was the close node and skipped the rest,
    which is what makes a branching section report one tail per arm rather than per edge. The
    iterative walk has to keep that, so a node with a close child and a further child contributes
    itself and nothing below that child."""
    sdfg = dace.SDFG("placement_ir_branch")
    open_node = OffloadingIRNode.new_open_node(sdfg.add_state("section"))
    head = OffloadingIRNode.new_state_node(sdfg.add_state("head"), OrderedSet(), OrderedSet())
    skipped = OffloadingIRNode.new_state_node(sdfg.add_state("skipped"), OrderedSet(), OrderedSet())
    open_node.append_node(head)
    head.append_node(open_node.close)
    head.append_node(skipped)
    skipped.append_node(open_node.close)
    assert open_node.get_all_tails() == [head]


def ordering_edge_after_a_kernel() -> dace.SDFG:
    """A kernel writes ``A``; an empty memlet orders the read of scalar ``s`` by a second kernel after it.

    CloudSC's ``zpsupsatsrce`` orders some thirty reads this way, ``ptsphy`` among them.
    """
    sdfg = dace.SDFG("placement_ordering_edge_after_a_kernel")
    sdfg.add_array("A", [LENGTH], dace.float64)
    sdfg.add_array("B", [LENGTH], dace.float64)
    sdfg.add_scalar("s", dace.float64)
    state = sdfg.add_state("ordered")
    _, _, exit_a = state.add_mapped_tasklet(
        "write_a", {"i": f"0:{LENGTH}"}, {}, "o = 1.0", {"o": dace.Memlet("A[i]")}, external_edges=True
    )
    written = state.out_edges(exit_a)[0].dst
    _, entry_b, _ = state.add_mapped_tasklet(
        "scale_b",
        {"i": f"0:{LENGTH}"},
        {"inp": dace.Memlet("s[0]")},
        "o = inp * 2.0",
        {"o": dace.Memlet("B[i]")},
        external_edges=True,
    )
    state.add_nedge(written, state.in_edges(entry_b)[0].src, dace.Memlet())
    sdfg.validate()
    return sdfg


def test_an_ordering_edge_does_not_make_a_scalar_device_written():
    """A scalar no kernel writes stays a by-value scalar: a device copy of it would be copied in mid-run."""
    sdfg = ordering_edge_after_a_kernel()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert isinstance(sdfg.arrays["s"], data.Scalar), sdfg.arrays["s"]
    assert not [name for name in sdfg.arrays if name != "s" and name.endswith("_s")], list(sdfg.arrays)


def test_the_pass_reports_what_it_placed_on_the_device():
    """A Pipeline reads the result as "did anything change", so an offload must not answer None."""
    sdfg = cpu_heap_sdfg()
    placed = OffloadToAccelerator().apply_pass(sdfg, {})
    assert placed is not None and {name.split(".", 1)[1] for name in placed} == {"A_gpu", "B_gpu"}, placed


def test_a_graph_with_nothing_to_offload_reports_no_change():
    sdfg = dace.SDFG("host_only_a_graph_with_nothing_to_offload_reports_no_change")
    sdfg.add_scalar("s", dace.float64)
    state = sdfg.add_state("host")
    one = state.add_tasklet("one", {}, {"o"}, "o = 1.0")
    state.add_edge(one, "o", state.add_write("s"), None, dace.Memlet("s[0]"))
    assert OffloadToAccelerator().apply_pass(sdfg, {}) is None


def one_element_read_by_two_maps() -> dace.SDFG:
    """``s = A[3]`` on the host, then two maps that both read ``s``."""
    sdfg = dace.SDFG("one_element_read_by_two_maps")
    for name in "ABC":
        sdfg.add_array(name, [LENGTH], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state("main")
    source = state.add_read("A")
    element = state.add_access("s")
    state.add_edge(source, None, element, None, dace.Memlet("A[3] -> [0]"))
    for name in "BC":
        entry, exit_node = state.add_map(f"scale_{name}", {"i": f"0:{LENGTH}"})
        tasklet = state.add_tasklet(f"times_{name}", {"x": None, "y": None}, {"o": None}, "o = x * y")
        state.add_memlet_path(element, entry, tasklet, dst_conn="x", memlet=dace.Memlet("s[0]"))
        state.add_memlet_path(source, entry, tasklet, dst_conn="y", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tasklet, exit_node, state.add_write(name), src_conn="o", memlet=dace.Memlet(f"{name}[i]"))
    sdfg.validate()
    return sdfg


def test_a_single_element_two_maps_read_stays_outside_both():
    """Moving the copy into the first map left the second reading a node inside the first one's scope,
    and the scope tree put ``times_B`` under ``scale_C``."""
    sdfg = one_element_read_by_two_maps()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    state = next(state for state in sdfg.states() if state.label == "main")
    scopes = state.scope_dict()
    tasklets = {node.label: scopes[node].map.label for node in state.nodes() if isinstance(node, dace.nodes.Tasklet)}
    assert tasklets == {"times_B": "scale_B", "times_C": "scale_C"}, tasklets
    assert all(scopes[node] is None for node in state.data_nodes() if node.data == "s")


def per_iteration_scratch() -> dace.SDFG:
    """``tmp`` is written and read inside the scope of one map."""
    sdfg = dace.SDFG("per_iteration_scratch")
    sdfg.add_array("A", [LENGTH], dace.float64)
    sdfg.add_array("B", [LENGTH], dace.float64)
    sdfg.add_transient("tmp", [2], dace.float64)
    state = sdfg.add_state("kernel")
    entry, exit_node = state.add_map("twice", {"i": f"0:{LENGTH}"})
    fill = state.add_tasklet("fill", {"a": None}, {"t": None}, "t = a")
    scratch = state.add_access("tmp")
    use = state.add_tasklet("use", {"t": None}, {"b": None}, "b = 2 * t")
    state.add_memlet_path(state.add_read("A"), entry, fill, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_edge(fill, "t", scratch, None, dace.Memlet("tmp[0]"))
    state.add_edge(scratch, None, use, "t", dace.Memlet("tmp[0]"))
    state.add_memlet_path(use, exit_node, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


ROWS = dace.symbol("ROWS")
COLS = dace.symbol("COLS")


def kernel_with_a_one_iteration_inner_map() -> dace.SDFG:
    """A map whose body is a single-iteration map writing and reading the length-1 local ``acc``."""
    sdfg = dace.SDFG("placement_kernel_with_a_one_iteration_inner_map")
    sdfg.add_array("A", [256], dace.float64)
    sdfg.add_array("B", [256], dace.float64)
    sdfg.add_array("acc", [1], dace.float64, transient=True)
    state = sdfg.add_state("kernel")
    outer_entry, outer_exit = state.add_map("device", {"i": "0:256"})
    inner_entry, inner_exit = state.add_map("once", {"k": "0:1"})
    double = state.add_tasklet("double", {"inp": None}, {"out": None}, "out = inp * 2.0")
    acc = state.add_access("acc")
    bump = state.add_tasklet("bump", {"inp": None}, {"out": None}, "out = inp + 1.0")
    state.add_memlet_path(
        state.add_read("A"), outer_entry, inner_entry, double, dst_conn="inp", memlet=dace.Memlet("A[i]")
    )
    state.add_edge(double, "out", acc, None, dace.Memlet("acc[0]"))
    state.add_edge(acc, None, bump, "inp", dace.Memlet("acc[0]"))
    state.add_memlet_path(
        bump, inner_exit, outer_exit, state.add_write("B"), src_conn="out", memlet=dace.Memlet("B[i]")
    )
    sdfg.validate()
    return sdfg


def test_a_length_one_local_of_a_one_iteration_map_in_a_kernel_is_a_register():
    """Extended scalarized this after dropping the trivial map; here it only has to live in the kernel."""
    sdfg = kernel_with_a_one_iteration_inner_map()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    assert sdfg.arrays["acc"].storage == dtypes.StorageType.Register
    state = next(state for state in sdfg.states() if state.label == "kernel")
    kernels = [
        n.map.label
        for n in state.nodes()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
    ]
    assert kernels == ["device"], kernels


@pytest.mark.gpu
def test_a_length_one_local_of_a_one_iteration_map_computes_what_numpy_computes():
    sdfg = kernel_with_a_one_iteration_inner_map()
    OffloadToAccelerator().apply_pass(sdfg, {})
    A = np.random.default_rng(3).random(256)
    B = np.zeros(256)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B, A * 2.0 + 1.0)


def host_accumulator_beside_a_kernel_in_a_loop(length: int) -> dace.SDFG:
    """npbench nbody's shape: a kernel on ``A`` and a host tasklet bumping ``PE``, every iteration of a loop,
    with both signature arrays handed over in device memory."""
    sdfg = dace.SDFG(f"host_accumulator_beside_a_kernel_in_a_loop_{length}")
    sdfg.add_array("A", [64], dace.float64)
    sdfg.add_array("PE", [length], dace.float64)
    loop = LoopRegion("steps", "t < 10", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_mapped_tasklet(
        "scale",
        {"i": "0:64"},
        {"a": dace.Memlet("A[i]")},
        "b = a * 0.5",
        {"b": dace.Memlet("A[i]")},
        external_edges=True,
    )
    bump = body.add_tasklet("bump", {"p": None}, {"q": None}, "q = p + 1.0")
    body.add_edge(body.add_read("PE"), None, bump, "p", dace.Memlet("PE[0]"))
    body.add_edge(bump, "q", body.add_write("PE"), None, dace.Memlet("PE[0]"))
    sdfg.validate()
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 4])
def test_a_host_accumulator_beside_a_kernel_computes_what_numpy_computes(length):
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    sdfg = host_accumulator_beside_a_kernel_in_a_loop(length)
    OffloadToAccelerator().apply_pass(sdfg, {})
    A = cupy.asarray(np.arange(64, dtype=np.float64))
    PE = cupy.zeros(length)
    sdfg(A=A, PE=PE)
    np.testing.assert_allclose(A.get(), np.arange(64) * 0.5**10)
    assert PE.get()[0] == 10.0


def kernel_with_a_shared_tile() -> dace.SDFG:
    """A block map stages ``A`` in the ``GPU_Shared`` transient ``tile``, and a second inner map reads it."""
    sdfg = dace.SDFG("kernel_with_a_shared_tile")
    sdfg.add_array("A", [64], dace.float64)
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_transient("tile", [32], dace.float64, storage=dtypes.StorageType.GPU_Shared)
    state = sdfg.add_state("kernel")
    blocks, blocks_exit = state.add_map("blocks", {"b": "0:2"})
    load_entry, load_exit = state.add_map("load", {"t": "0:32"})
    use_entry, use_exit = state.add_map("use", {"t": "0:32"})
    load = state.add_tasklet("load", {"a": None}, {"o": None}, "o = a")
    use = state.add_tasklet("use", {"s": None}, {"o": None}, "o = s * 2")
    tile = state.add_access("tile")
    state.add_memlet_path(
        state.add_read("A"), blocks, load_entry, load, dst_conn="a", memlet=dace.Memlet("A[b * 32 + t]")
    )
    state.add_memlet_path(load, load_exit, tile, src_conn="o", memlet=dace.Memlet("tile[t]"))
    state.add_memlet_path(tile, use_entry, use, dst_conn="s", memlet=dace.Memlet("tile[t]"))
    state.add_memlet_path(
        use, use_exit, blocks_exit, state.add_write("B"), src_conn="o", memlet=dace.Memlet("B[b * 32 + t]")
    )
    sdfg.validate()
    return sdfg


def host_tasklets_on_both_sides_of_a_kernel() -> dace.SDFG:
    """``scale`` feeds a kernel whose sum ``divide`` reads; both host tasklets read ``C``, so they share one component."""
    sdfg = dace.SDFG("placement_host_tasklets_on_both_sides_of_a_kernel")
    sdfg.add_array("A", [LENGTH], dace.float64)
    sdfg.add_array("C", [2], dace.float64)
    sdfg.add_array("OUT", [2], dace.float64)
    sdfg.add_transient("factor", [1], dace.float64)
    sdfg.add_transient("total", [1], dace.float64)
    # The kernel accumulates into ``total``, so it starts at zero; device memory is not zeroed for it.
    init = sdfg.add_state("zero_total", is_start_block=True)
    zero = init.add_tasklet("zero", set(), {"z": None}, "z = 0.0")
    init.add_edge(zero, "z", init.add_write("total"), None, dace.Memlet("total[0]"))
    state = sdfg.add_state_after(init, "mixed")
    c = state.add_read("C")
    scale = state.add_tasklet("scale", {"c": None}, {"f": None}, "f = 2.0 * c")
    factor = state.add_access("factor")
    state.add_edge(c, None, scale, "c", dace.Memlet("C[0]"))
    state.add_edge(scale, "f", factor, None, dace.Memlet("factor[0]"))
    total = state.add_access("total")
    state.add_mapped_tasklet(
        "weighted_sum",
        {"i": f"0:{LENGTH}"},
        {"a": dace.Memlet("A[i]"), "f": dace.Memlet("factor[0]")},
        "s = a * f",
        {"s": dace.Memlet("total[0]", wcr="lambda x, y: x + y")},
        input_nodes={"A": state.add_read("A"), "factor": factor},
        output_nodes={"total": total},
        external_edges=True,
    )
    divide = state.add_tasklet("divide", {"c": None, "t": None}, {"o": None}, "o = c / t")
    state.add_edge(c, None, divide, "c", dace.Memlet("C[1]"))
    state.add_edge(total, None, divide, "t", dace.Memlet("total[0]"))
    state.add_edge(divide, "o", state.add_write("OUT"), None, dace.Memlet("OUT[0]"))
    sdfg.validate()
    return sdfg


def test_host_tasklets_on_both_sides_of_a_kernel_get_separate_wrappers():
    """bicgstab: one size-1 wrapper around ``scale`` and ``divide`` would feed the kernel and wait for it."""
    sdfg = host_tasklets_on_both_sides_of_a_kernel()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    state = next(s for s in sdfg.states() if s.label == "mixed")
    scopes = state.scope_dict()
    scale, divide = (
        next(n for n in state.nodes() if getattr(n, "label", None) == name) for name in ("scale", "divide")
    )
    assert scopes[scale] is not None and scopes[divide] is not None and scopes[scale] is not scopes[divide]


@pytest.mark.gpu
def test_host_tasklets_on_both_sides_of_a_kernel_compute_what_numpy_computes():
    sdfg = host_tasklets_on_both_sides_of_a_kernel()
    sdfg.name = "placement_host_tasklets_on_both_sides_of_a_kernel_run"
    sdfg.apply_gpu_transformations()
    out = np.zeros(2)
    sdfg(A=np.arange(LENGTH, dtype=np.float64), C=np.array([3.0, 5.0]), OUT=out)
    assert out[0] == 5.0 / (6.0 * np.arange(LENGTH).sum())


if __name__ == "__main__":
    test_a_tail_contributes_none_of_its_remaining_siblings()
    test_an_ordering_edge_does_not_make_a_scalar_device_written()
    test_the_pass_reports_what_it_placed_on_the_device()
    test_a_graph_with_nothing_to_offload_reports_no_change()
    test_a_single_element_two_maps_read_stays_outside_both()
    test_a_length_one_local_of_a_one_iteration_map_in_a_kernel_is_a_register()
    test_a_length_one_local_of_a_one_iteration_map_computes_what_numpy_computes()
    test_a_host_accumulator_beside_a_kernel_computes_what_numpy_computes(1)
    test_a_host_accumulator_beside_a_kernel_computes_what_numpy_computes(4)
    test_host_tasklets_on_both_sides_of_a_kernel_get_separate_wrappers()
    test_host_tasklets_on_both_sides_of_a_kernel_compute_what_numpy_computes()


def register_array_two_kernels_share() -> dace.SDFG:
    """dace-fortran's loop-local arrays are ``Register``: one map fills ``t``, the next reads it (ICON's
    ``z_w_con_c`` in a block body)."""
    sdfg = dace.SDFG("register_array_two_kernels_share")
    sdfg.add_array("A", [LENGTH], dace.float64)
    sdfg.add_array("B", [LENGTH], dace.float64)
    sdfg.add_transient("t", [LENGTH], dace.float64, storage=dace.StorageType.Register)
    fill = sdfg.add_state("fill")
    fill.add_mapped_tasklet(
        "double",
        {"i": f"0:{LENGTH}"},
        {"a": dace.Memlet("A[i]")},
        "o = 2 * a",
        {"o": dace.Memlet("t[i]")},
        external_edges=True,
    )
    use = sdfg.add_state_after(fill, "use")
    use.add_mapped_tasklet(
        "shift",
        {"i": f"0:{LENGTH}"},
        {"x": dace.Memlet("t[i]")},
        "o = x + 1",
        {"o": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def test_a_register_array_two_kernels_share_is_device_memory():
    """Kept in ``Register`` as if it were scope-local, it was a host stack array both kernels were handed."""
    sdfg = register_array_two_kernels_share()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    assert sdfg.arrays["t"].storage == dace.StorageType.GPU_Global
