# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegaKernelOptimization``: which chains of GPU kernels one persistent kernel runs, the cooperative kernel and its
host-side sizing, the grid barriers between dependent stages, and the fused kernel itself."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ComputePersistentKernelRange
from dace.libraries.standard.nodes.sync import Sync
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.megakernel import MegaKernelOptimization
from dace.transformation.passes.megakernel.discovery import find_chains
from dace.transformation.passes.megakernel.grid_sync import InsertGridSyncs, is_grid_sync

sys.path.insert(0, str(Path(__file__).parents[1] / "npbench" / "polybench"))
import heat_3d_test  # noqa: E402
import jacobi_2d_test  # noqa: E402

N = dace.symbol("N")


def chains_of(sdfg: dace.SDFG):
    return find_chains(sdfg, sdfg)


@pytest.mark.parametrize("program", [jacobi_2d_test.kernel, heat_3d_test.heat_3d_kernel], ids=["jacobi2d", "heat3d"])
def test_a_time_loop_of_stencil_kernels_is_one_chain(program):
    """The copies to and from the device stay on the host; the whole ``for t`` loop with both stencils is one chain."""
    sdfg = program.to_sdfg(simplify=True)
    auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    chains = chains_of(sdfg)
    assert len(chains) == 1
    assert len(chains[0].blocks) == 1 and isinstance(chains[0].blocks[0], LoopRegion)
    assert len(chains[0].kernels) == 2


def gpu_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    for array in "ABC":
        sdfg.add_array(array, [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_symbol("T", dace.int64)
    return sdfg


def add_kernel(state: dace.SDFGState, label: str, src: str = "A", dst: str = "A") -> None:
    state.add_mapped_tasklet(
        label,
        {"i": "0:N"},
        {"a": dace.Memlet(f"{src}[i]")},
        "b = a + 1",
        {"b": dace.Memlet(f"{dst}[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )


def add_time_loop(sdfg: dace.SDFG, label: str, condition: str = "t < T") -> LoopRegion:
    loop = LoopRegion(label, "t < T", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop)
    add_kernel(loop.add_state(f"{label}_body", is_start_block=True), f"{label}_kernel")
    # Set last: building the body infers the loop symbols from the condition, which may read data
    loop.loop_condition = CodeBlock(condition)
    return loop


def test_host_work_between_loops_splits_them_into_two_chains():
    sdfg = gpu_sdfg("megakernel_two_chains")
    sdfg.add_array("H", [1], dace.float64)
    first = add_time_loop(sdfg, "first")
    host = sdfg.add_state_after(first, "host")
    task = host.add_tasklet("count", {}, {"h"}, "h = 1")
    host.add_edge(task, "h", host.add_write("H"), None, dace.Memlet("H[0]"))
    second = add_time_loop(sdfg, "second")
    sdfg.add_edge(host, second, dace.InterstateEdge())
    sdfg.validate()

    assert [chain.blocks for chain in chains_of(sdfg)] == [[first], [second]]


def test_a_loop_whose_condition_reads_data_keeps_only_its_body():
    """``while A[0] < 10`` is not the same for every thread once the kernel writes ``A``: the loop stays on the host and
    only its body becomes a megakernel."""
    sdfg = gpu_sdfg("megakernel_data_condition")
    loop = add_time_loop(sdfg, "until", condition="A[0] < 10")
    sdfg.validate()

    chains = chains_of(sdfg)
    assert len(chains) == 1
    assert chains[0].region is loop and chains[0].blocks == [loop.start_block]


def new_gpu_codegen():
    """Megakernels are emitted by the new GPU code generator only."""
    return dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental")


def cooperative_counter() -> dace.SDFG:
    """``ComputePersistentKernelRange`` sizes a cooperative kernel of 256-thread blocks; every thread adds one."""
    sdfg = dace.SDFG("megakernel_cooperative_counter")
    sdfg.add_array("C", [1], dace.int64, storage=dace.StorageType.GPU_Global)
    sdfg.add_scalar("nblk", dace.int32, transient=True)
    size = sdfg.add_state("size", is_start_block=True)
    sizing = ComputePersistentKernelRange("sizing", kernel_handle="counter")
    size.add_node(sizing)
    size.add_edge(sizing, "resident_blocks", size.add_write("nblk"), None, dace.Memlet("nblk[0]"))

    run = sdfg.add_state_after(size, "run")
    entry, exit_ = run.add_map("counter", {"w": "0:nblk*256"}, schedule=dace.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [256, 1, 1]
    entry.map.cooperative = True
    entry.map.explicit_sync = True
    entry.map.kernel_handle = "counter"
    entry.add_in_connector("nblk")
    run.add_edge(run.add_read("nblk"), None, entry, "nblk", dace.Memlet("nblk[0]"))
    task = run.add_tasklet("one", {}, {"c"}, "c = 1")
    run.add_edge(entry, None, task, None, dace.Memlet())
    run.add_memlet_path(
        task, exit_, run.add_write("C"), src_conn="c", memlet=dace.Memlet("C[0]", wcr="lambda a, b: a + b")
    )
    return sdfg


def test_the_cooperative_properties_survive_serialization():
    sdfg = cooperative_counter()
    loaded = dace.SDFG.from_json(sdfg.to_json())
    kernel = next(n for n, state in loaded.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    assert kernel.map.cooperative and kernel.map.explicit_sync and kernel.map.kernel_handle == "counter"
    plain = dace.nodes.Map("plain", ["i"], dace.subsets.Range.from_string("0:4"))
    serialized = dace.serialize.all_properties_to_json(plain)
    assert not {"cooperative", "explicit_sync", "kernel_handle"} & serialized.keys()


def test_a_cooperative_kernel_launches_cooperatively_and_sizes_its_grid():
    """The kernel keeps the grid ``nblk`` it was given: an explicitly synchronized kernel is not re-gridded."""
    with new_gpu_codegen():
        code = "\n".join(obj.clean_code for obj in cooperative_counter().generate_code())
    launches = [line for line in code.splitlines() if "LaunchCooperativeKernel" in line]
    assert len(launches) == 1 and "dim3(nblk, 1, 1)" in launches[0]
    assert "LaunchKernel(" not in code
    assert "__dace_persistent_blocks_counter(__state)" in code  # the host asks before the launch
    assert "OccupancyMaxActiveBlocksPerMultiprocessor" in code


@pytest.mark.gpu
def test_the_sized_grid_runs_every_block_at_once():
    import cupy

    sdfg = cooperative_counter()
    counter = cupy.zeros(1, dtype=np.int64)
    with new_gpu_codegen():
        sdfg(C=counter)
    threads = int(counter[0])
    sms = cupy.cuda.Device().attributes["MultiProcessorCount"]
    assert threads > 0 and threads % 256 == 0
    assert threads // 256 <= sms * 2048 // 256  # never more blocks than the device holds resident


def top_level_maps(state: dace.SDFGState) -> list[dace.nodes.MapEntry]:
    return [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is None]


def straight(sdfg: dace.SDFG, kernels: list[tuple[str, str]]) -> dace.sdfg.state.ControlFlowRegion:
    state = sdfg.add_state("stages", is_start_block=True)
    for idx, (src, dst) in enumerate(kernels):
        add_kernel(state, f"k{idx}", src, dst)
    return sdfg


def looped(sdfg: dace.SDFG, kernels: list[tuple[str, str]]) -> dace.sdfg.state.ControlFlowRegion:
    loop = LoopRegion("time", "t < T", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("stages", is_start_block=True)
    for idx, (src, dst) in enumerate(kernels):
        add_kernel(state, f"k{idx}", src, dst)
    return loop


@pytest.mark.parametrize(
    "build, kernels, expected",
    [
        (straight, [("A", "A"), ("B", "B")], ["stage", "stage"]),
        (straight, [("A", "B"), ("B", "C")], ["stage", "sync", "stage"]),
        (looped, [("A", "A")], ["stage", "sync"]),
    ],
    ids=["independent", "read_after_write", "loop_latch"],
)
def test_grid_syncs_separate_exactly_the_dependent_stages(build, kernels, expected):
    """Every kernel becomes a stage; a barrier follows a stage only when the next stage, or the next iteration's first
    stage, touches what it wrote."""
    sdfg = gpu_sdfg("megakernel_grid_syncs")
    region = build(sdfg, kernels)
    InsertGridSyncs().apply_pass(sdfg, {})

    order = list(region.bfs_nodes(region.start_block))
    assert ["sync" if is_grid_sync(b) else "stage" for b in order] == expected
    assert all(len(top_level_maps(b)) == 1 for b in order if not is_grid_sync(b))


def test_a_small_dimension_stays_a_loop_in_every_thread():
    """With ``K`` small, the grid strides over ``i`` only, and every thread runs all of ``k``."""
    sdfg = gpu_sdfg("megakernel_small_dimension")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("D", [N, "K"], dace.float64, storage=dace.StorageType.GPU_Global)
    loop = LoopRegion("time", "t < T", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True).add_mapped_tasklet(
        "small",
        {"i": "0:N", "k": "0:K"},
        {"a": dace.Memlet("D[i, k]")},
        "b = a + 1",
        {"b": dace.Memlet("D[i, k]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    MegaKernelOptimization(small_symbols={"K"}).apply_pass(sdfg, {})
    sdfg.validate()

    body = next(n for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    maps = [n.map for n, s in body.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert [m.params for m in maps] == [["__mk_g"], ["k"]]
    assert maps[0].range[0][1] == N - 1
    assert all(m.schedule == dace.ScheduleType.Sequential for m in maps)


STENCILS = [(jacobi_2d_test.kernel, np.float32, 2, 300), (heat_3d_test.heat_3d_kernel, np.float64, 3, 40)]
STENCIL_IDS = ["jacobi2d", "heat3d"]


def fused_stencil(program) -> tuple[dace.SDFG, dace.SDFG]:
    """The GPU-optimized stencil, and a copy with its time loop fused into one megakernel."""
    sdfg = program.to_sdfg(simplify=True)
    auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    fused = copy.deepcopy(sdfg)
    fused.name = f"{sdfg.name}_megakernel"
    assert len(MegaKernelOptimization().apply_pass(fused, {})) == 1
    fused.validate()
    return sdfg, fused


@pytest.mark.parametrize("program", [stencil[0] for stencil in STENCILS], ids=STENCIL_IDS)
def test_a_stencil_time_loop_is_one_cooperative_launch_with_two_grid_syncs(program):
    """One launch runs every time step: a barrier after the first stencil (the second reads its output) and one at
    the end of the step (the next step reads the second stencil's output)."""
    fused = fused_stencil(program)[1]
    syncs = [n for n, s in fused.all_nodes_recursive() if isinstance(n, Sync)]
    assert len(syncs) == 2
    with new_gpu_codegen():
        code = "\n".join(obj.clean_code for obj in fused.generate_code())
    assert code.count("LaunchCooperativeKernel(") == 1 and "LaunchKernel(" not in code
    assert code.count("cooperative_groups::this_grid().sync();") == 2
    assert code.count("for (int64_t __mk_g = __mk_tid;") == 2


@pytest.mark.gpu
@pytest.mark.parametrize("program, dtype, rank, size", STENCILS, ids=STENCIL_IDS)
def test_the_fused_stencil_matches_the_kernel_per_step_version(program, dtype, rank, size):
    sdfg, fused = fused_stencil(program)
    rng = np.random.default_rng(0)
    inputs = {name: rng.random((size,) * rank).astype(dtype) for name in "AB"}
    want = copy.deepcopy(inputs)
    sdfg(TSTEPS=9, N=size, **want)
    with new_gpu_codegen():
        fused(TSTEPS=9, N=size, **inputs)
    for name in "AB":
        np.testing.assert_array_equal(inputs[name], want[name])


if __name__ == "__main__":
    test_a_time_loop_of_stencil_kernels_is_one_chain(jacobi_2d_test.kernel)
    test_a_time_loop_of_stencil_kernels_is_one_chain(heat_3d_test.heat_3d_kernel)
    test_host_work_between_loops_splits_them_into_two_chains()
    test_a_loop_whose_condition_reads_data_keeps_only_its_body()
    test_the_cooperative_properties_survive_serialization()
    test_a_cooperative_kernel_launches_cooperatively_and_sizes_its_grid()
    test_the_sized_grid_runs_every_block_at_once()
    test_grid_syncs_separate_exactly_the_dependent_stages(straight, [("A", "A"), ("B", "B")], ["stage", "stage"])
    test_grid_syncs_separate_exactly_the_dependent_stages(
        straight, [("A", "B"), ("B", "C")], ["stage", "sync", "stage"]
    )
    test_grid_syncs_separate_exactly_the_dependent_stages(looped, [("A", "A")], ["stage", "sync"])
    test_a_small_dimension_stays_a_loop_in_every_thread()
    for program in [stencil[0] for stencil in STENCILS]:
        test_a_stencil_time_loop_is_one_cooperative_launch_with_two_grid_syncs(program)
    for stencil in STENCILS:
        test_the_fused_stencil_matches_the_kernel_per_step_version(*stencil)
