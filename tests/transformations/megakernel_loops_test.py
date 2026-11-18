# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegakernelLoops``: one persistent kernel only for kernels launched inside a sequential loop, behind a runtime test
of their size on the GPU; straight-line kernels keep one launch each."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.megakernel.loops import MegakernelLoops

sys.path.insert(0, str(Path(__file__).parents[1] / "npbench" / "polybench"))
import jacobi_2d_test  # noqa: E402

N = dace.symbol("N")


def gpu_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    for array in "AB":
        sdfg.add_array(array, [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_symbol("T", dace.int64)
    return sdfg


def add_kernel(state: dace.SDFGState, label: str, src: str, dst: str) -> None:
    _, entry, _ = state.add_mapped_tasklet(
        label,
        {"i": "0:N"},
        {"a": dace.Memlet(f"{src}[i]")},
        "b = a + 1",
        {"b": dace.Memlet(f"{dst}[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry.map.gpu_block_size = [256, 1, 1]


def kernel_loop() -> dace.SDFG:
    """``for t in 0:T: B = A + 1; A = B + 1``, two dependent kernels launched in every iteration."""
    sdfg = gpu_sdfg("mk_loops_kernel_loop")
    loop = LoopRegion("time", "t < T", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    add_kernel(first, "forward", "A", "B")
    second = loop.add_state_after(first, "second")
    add_kernel(second, "back", "B", "A")
    loop.loop_condition = CodeBlock("t < T")
    return sdfg


def two_kernel_loops() -> dace.SDFG:
    """Two time loops, ``early`` then ``late``, each launching one kernel per iteration."""
    sdfg = gpu_sdfg("mk_loops_two_loops")
    previous = None
    for label, (src, dst) in (("early", "AB"), ("late", "BA")):
        loop = LoopRegion(label, "t < T", "t", "t = 0", "t = t + 1")
        sdfg.add_node(loop, is_start_block=previous is None)
        if previous is not None:
            sdfg.add_edge(previous, loop, dace.InterstateEdge())
        add_kernel(loop.add_state(f"{label}_body", is_start_block=True), f"{label}_kernel", src, dst)
        previous = loop
    return sdfg


def straight_kernels() -> dace.SDFG:
    """``B = A + 1; A = B + 1`` once each, no loop around them."""
    sdfg = gpu_sdfg("mk_loops_straight")
    first = sdfg.add_state("first", is_start_block=True)
    add_kernel(first, "forward", "A", "B")
    add_kernel(sdfg.add_state_after(first, "second"), "back", "B", "A")
    return sdfg


def cooperative_maps(graph) -> list:
    return [n for n, _ in graph.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.map.cooperative]


def test_a_kernel_loop_runs_persistently_behind_a_test_of_its_launch_size():
    sdfg = kernel_loop()
    assert MegakernelLoops().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (guard,) = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    (fits, persistent), (otherwise, per_launch) = guard.branches
    assert "N" in fits.as_string and "<=" in fits.as_string and otherwise is None
    assert len(cooperative_maps(persistent)) == 1
    loops = [r for r in per_launch.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert len(loops) == 1, "the per-launch loop is kept for launches that do not fit the device"


def test_only_the_named_loop_runs_persistently():
    sdfg = two_kernel_loops()
    assert MegakernelLoops(blocks={"late"}).apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (guard,) = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    assert len(cooperative_maps(guard)) == 1
    (early,) = [r for r in sdfg.nodes() if isinstance(r, LoopRegion)]
    assert early.label == "early" and not cooperative_maps(early), "the loop not named keeps one launch per kernel"


def three_straight_kernels() -> dace.SDFG:
    """``B = A + 1; A = B + 1; B = A + 1`` once each, in states ``first``, ``second``, ``third``."""
    sdfg = gpu_sdfg("mk_loops_three_straight")
    first = sdfg.add_state("first", is_start_block=True)
    add_kernel(first, "one", "A", "B")
    second = sdfg.add_state_after(first, "second")
    add_kernel(second, "two", "B", "A")
    add_kernel(sdfg.add_state_after(second, "third"), "three", "A", "B")
    return sdfg


def test_a_named_run_of_straight_kernels_is_one_persistent_kernel_behind_its_size_test():
    sdfg = three_straight_kernels()
    assert MegakernelLoops(blocks={"first", "second"}).apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (guard,) = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    (fits, persistent), (otherwise, per_launch) = guard.branches
    assert "N" in fits.as_string and otherwise is None
    assert len(cooperative_maps(persistent)) == 1
    assert len([n for n, _ in per_launch.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]) == 2
    assert (
        len(
            [
                n
                for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and n.map.label == "three_map" and not n.map.cooperative
            ]
        )
        == 1
    )


def test_named_blocks_that_do_not_follow_one_another_are_refused_and_change_nothing():
    sdfg = three_straight_kernels()
    before = sdfg.hash_sdfg()
    assert MegakernelLoops(blocks={"first", "third"}).apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


def test_kernels_outside_a_loop_keep_one_launch_each():
    sdfg = straight_kernels()
    assert MegakernelLoops().apply_pass(sdfg, {}) is None
    assert not cooperative_maps(sdfg)


def test_a_cpu_time_loop_is_one_omp_team_and_matches_the_per_map_version():
    sdfg = jacobi_2d_test.kernel.to_sdfg(simplify=True)
    auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
    fused = copy.deepcopy(sdfg)
    fused.name = f"{sdfg.name}_mk_loops_cpu"
    assert MegakernelLoops(dace.dtypes.DeviceType.CPU).apply_pass(fused, {}) == 1
    fused.validate()
    code = fused.generate_code()[0].clean_code
    assert code.count("#pragma omp parallel") == 1

    rng = np.random.default_rng(0)
    inputs = {name: rng.random((40, 40), dtype=np.float32) for name in "AB"}
    want = copy.deepcopy(inputs)
    sdfg(TSTEPS=9, N=40, **want)
    fused(TSTEPS=9, N=40, **inputs)
    for name in "AB":
        np.testing.assert_array_equal(inputs[name], want[name])


@pytest.mark.gpu
@pytest.mark.parametrize("size", [64, 4096], ids=["fits_the_device", "larger_than_the_device"])
def test_a_gpu_time_loop_matches_the_per_launch_version_on_both_sides_of_its_size_test(size):
    sdfg = jacobi_2d_test.kernel.to_sdfg(simplify=True)
    auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    fused = copy.deepcopy(sdfg)
    fused.name = f"{sdfg.name}_mk_loops_gpu_{size}"
    assert MegakernelLoops().apply_pass(fused, {}) == 1
    rng = np.random.default_rng(0)
    inputs = {name: rng.random((size, size), dtype=np.float32) for name in "AB"}
    want = copy.deepcopy(inputs)
    sdfg(TSTEPS=5, N=size, **want)
    # A cooperative launch is emitted by the new GPU code generator only
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        fused(TSTEPS=5, N=size, **inputs)
    for name in "AB":
        np.testing.assert_array_equal(inputs[name], want[name])


if __name__ == "__main__":
    test_a_kernel_loop_runs_persistently_behind_a_test_of_its_launch_size()
    test_only_the_named_loop_runs_persistently()
    test_a_named_run_of_straight_kernels_is_one_persistent_kernel_behind_its_size_test()
    test_named_blocks_that_do_not_follow_one_another_are_refused_and_change_nothing()
    test_kernels_outside_a_loop_keep_one_launch_each()
    test_a_cpu_time_loop_is_one_omp_team_and_matches_the_per_map_version()
    test_a_gpu_time_loop_matches_the_per_launch_version_on_both_sides_of_its_size_test(64)
    test_a_gpu_time_loop_matches_the_per_launch_version_on_both_sides_of_its_size_test(4096)
