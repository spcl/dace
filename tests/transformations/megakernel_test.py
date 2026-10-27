# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegaKernelOptimization`` stage 1: which chains of GPU kernels one persistent kernel runs, and the cooperative
kernel and its host-side sizing that the fused kernel is built from."""

import sys
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ComputePersistentKernelRange
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.megakernel import MegaKernelOptimization

sys.path.insert(0, str(Path(__file__).parents[1] / "npbench" / "polybench"))
import heat_3d_test  # noqa: E402
import jacobi_2d_test  # noqa: E402

N = dace.symbol("N")


def chains_of(sdfg: dace.SDFG):
    return MegaKernelOptimization().apply_pass(sdfg, {}) or []


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
    sdfg.add_array("A", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_symbol("T", dace.int64)
    return sdfg


def add_kernel(state: dace.SDFGState, label: str) -> None:
    state.add_mapped_tasklet(
        label,
        {"i": "0:N"},
        {"a": dace.Memlet("A[i]")},
        "b = a + 1",
        {"b": dace.Memlet("A[i]")},
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


if __name__ == "__main__":
    test_a_time_loop_of_stencil_kernels_is_one_chain(jacobi_2d_test.kernel)
    test_a_time_loop_of_stencil_kernels_is_one_chain(heat_3d_test.heat_3d_kernel)
    test_host_work_between_loops_splits_them_into_two_chains()
    test_a_loop_whose_condition_reads_data_keeps_only_its_body()
    test_the_cooperative_properties_survive_serialization()
    test_a_cooperative_kernel_launches_cooperatively_and_sizes_its_grid()
    test_the_sized_grid_runs_every_block_at_once()
