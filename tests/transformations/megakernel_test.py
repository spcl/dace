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
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
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
    tasklet, entry, exit_ = state.add_mapped_tasklet(
        label,
        {"i": "0:N"},
        {"a": dace.Memlet(f"{src}[i]")},
        "b = a + 1",
        {"b": dace.Memlet(f"{dst}[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry.map.gpu_block_size = [256, 1, 1]


def add_time_loop(sdfg: dace.SDFG, label: str, condition: str = "t < T") -> LoopRegion:
    loop = LoopRegion(label, "t < T", "t", "t = 0", "t = t + 1")
    sdfg.add_node(loop)
    add_kernel(loop.add_state(f"{label}_body", is_start_block=True), f"{label}_kernel")
    # Set last: building the body infers the loop symbols from the condition, which may read data
    loop.loop_condition = CodeBlock(condition)
    return loop


def test_a_kernel_state_with_a_view_forms_no_chain():
    """Grid barriers order containers by name, so a view (another name for a container) keeps a state out."""
    sdfg = gpu_sdfg("megakernel_view")
    sdfg.add_view("V", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state("kernels", is_start_block=True)
    add_kernel(state, "k0", "A", "B")
    view = state.add_access("V")
    state.add_edge(state.add_read("C"), None, view, "views", dace.Memlet("C[0:N]"))
    tasklet, entry, exit_ = state.add_mapped_tasklet(
        "k1",
        {"i": "0:N"},
        {"a": dace.Memlet("V[i]")},
        "b = a",
        {"b": dace.Memlet("A[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        input_nodes={"V": view},
        external_edges=True,
    )
    sdfg.validate()

    assert chains_of(sdfg) == []


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
    assert "occupancy query failed" in code  # a failed query reports its GPU error, not "does not fit"


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
    InsertGridSyncs(sync_all=False, ownership=False).apply_pass(sdfg, {})

    order = list(region.bfs_nodes(region.start_block))
    assert ["sync" if is_grid_sync(b) else "stage" for b in order] == expected
    assert all(len(top_level_maps(b)) == 1 for b in order if not is_grid_sync(b))


def test_sync_all_separates_independent_stages_too():
    sdfg = gpu_sdfg("megakernel_sync_all")
    region = straight(sdfg, [("A", "A"), ("B", "B")])
    assert InsertGridSyncs().apply_pass(sdfg, {}) == 1

    order = list(region.bfs_nodes(region.start_block))
    assert ["sync" if is_grid_sync(b) else "stage" for b in order] == ["stage", "sync", "stage"]


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


def kernels_under_a_host_loop() -> dace.SDFG:
    """``for b in 0:4 (host): nested { B = A + 1; A = B + 1 }``: two dependent kernels per host iteration."""
    body = gpu_sdfg("megakernel_block_body")
    straight(body, [("A", "B"), ("B", "A")])
    body.remove_data("C")
    body.remove_symbol("T")
    sdfg = gpu_sdfg("megakernel_host_loop")
    sdfg.remove_data("C")
    state = sdfg.add_state("blocks", is_start_block=True)
    entry, exit_ = state.add_map("blocks", {"b": "0:4"}, schedule=dace.ScheduleType.Sequential)
    node = state.add_nested_sdfg(body, {"A": None, "B": None}, {"A": None, "B": None})
    for name in "AB":
        state.add_memlet_path(state.add_read(name), entry, node, dst_conn=name, memlet=dace.Memlet(f"{name}[0:N]"))
        state.add_memlet_path(node, exit_, state.add_write(name), src_conn=name, memlet=dace.Memlet(f"{name}[0:N]"))
    return sdfg


def test_kernels_in_a_nested_sdfg_under_a_host_loop_form_a_chain():
    """The host loop stays on the host; the nested SDFG it calls is one chain, fused once per host iteration."""
    sdfg = kernels_under_a_host_loop()
    sdfg.validate()
    chains = chains_of(sdfg)
    assert len(chains) == 1 and chains[0].region.name == "megakernel_block_body"
    assert len(chains[0].kernels) == 2


@pytest.mark.gpu
def test_a_megakernel_per_host_iteration_matches_the_kernels():
    import cupy

    sdfg = kernels_under_a_host_loop()
    fused = copy.deepcopy(sdfg)
    fused.name = "megakernel_host_loop_fused"
    assert len(MegaKernelOptimization().apply_pass(fused, {})) == 1
    fused.validate()
    init = np.random.default_rng(0).random(1000)
    want = {name: cupy.asarray(init) for name in "AB"}
    got = {name: cupy.asarray(init) for name in "AB"}
    sdfg(N=1000, **want)
    with new_gpu_codegen():
        fused(N=1000, **got)
    for name in "AB":
        np.testing.assert_array_equal(cupy.asnumpy(got[name]), cupy.asnumpy(want[name]))


def add_kernel_from(
    state: dace.SDFGState, label: str, src: str, dst: str, code: str = "b = a + 1", extra: dict | None = None
) -> None:
    """A kernel over ``lo:N`` writing ``dst[i]`` from ``src[i]`` and the ``extra`` inputs."""
    inputs = {"a": dace.Memlet(f"{src}[i]")} | (extra or {})
    tasklet, entry, exit_ = state.add_mapped_tasklet(
        label,
        {"i": "lo:N"},
        inputs,
        code,
        {"b": dace.Memlet(f"{dst}[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry.map.gpu_block_size = [256, 1, 1]


def kernels_split_by_a_reassigned_bound() -> dace.SDFG:
    """Two kernel states that the edge between them splits: it reassigns their lower bound from a host array, and the
    second state computes on the host the scalar its kernel takes by value."""
    sdfg = gpu_sdfg("megakernel_split_by_bound")
    sdfg.add_array("bounds", [2], dace.int64)
    sdfg.add_scalar("scale", dace.float64, transient=True)
    sdfg.add_symbol("lo", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    first, second = sdfg.add_state("first"), sdfg.add_state("second")
    sdfg.add_edge(init, first, dace.InterstateEdge(assignments={"lo": "bounds[0]"}))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"lo": "bounds[1]"}))
    sdfg.add_state_after(second, "done")
    add_kernel_from(first, "k_first", "A", "B")
    add_kernel_from(second, "k_second", "B", "C", "b = a * s", {"s": dace.Memlet("scale[0]")})
    host = second.add_tasklet("set_scale", {}, {"out"}, "out = 2.0")
    scale = next(n for n in second.data_nodes() if n.data == "scale")
    second.add_edge(host, "out", scale, None, dace.Memlet("scale[0]"))
    return sdfg


def test_kernels_split_by_a_reassigned_bound_become_one_state_and_one_megakernel():
    """Renaming each write of ``lo``, sifting the assignments up and fusing the states leaves both kernels in one
    state, the host tasklet in a state before it, and one cooperative kernel."""
    sdfg = kernels_split_by_a_reassigned_bound()
    sdfg.validate()
    fused = copy.deepcopy(sdfg)
    fused.name = "megakernel_split_by_bound_fused"
    chains = MegaKernelOptimization().apply_pass(fused, {})
    fused.validate()

    assert len(chains) == 1 and len(chains[0].kernels) == 2
    kernel_states = {id(s) for s in chains[0].blocks for k in chains[0].kernels if k in s.nodes()}
    assert len(kernel_states) == 1
    cooperative = [
        n for n, _ in fused.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.map.cooperative
    ]
    assert len(cooperative) == 1
    kernel_bounds = {
        str(n.map.range[0][0])
        for n, _ in fused.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.Sequential
    }
    assert "lo" not in kernel_bounds


def test_kernels_split_by_a_reassigned_bound_match_as_one_megakernel():
    import cupy

    sdfg = kernels_split_by_a_reassigned_bound()
    fused = copy.deepcopy(sdfg)
    fused.name = "megakernel_split_by_bound_run"
    MegaKernelOptimization().apply_pass(fused, {})
    rng = np.random.default_rng(0)
    init = {name: rng.random(1000) for name in "ABC"}
    want = {name: cupy.asarray(v) for name, v in init.items()}
    got = {name: cupy.asarray(v) for name, v in init.items()}
    bounds = np.array([3, 17], dtype=np.int64)
    sdfg(N=1000, bounds=bounds, **want)
    with new_gpu_codegen():
        fused(N=1000, bounds=bounds, **got)
    for name in "ABC":
        np.testing.assert_array_equal(cupy.asnumpy(got[name]), cupy.asnumpy(want[name]))


def kernels_after_a_conditional_bound() -> dace.SDFG:
    """A conditional picks the lower bound ``lo`` of two kernels; host code after them reads ``lo`` too."""
    sdfg = gpu_sdfg("megakernel_conditional_bound")
    sdfg.add_symbol("lo", dace.int64)
    sdfg.add_symbol("flag", dace.int64)
    sdfg.add_array("out", [1], dace.int64)
    pick = ConditionalBlock("pick_lo")
    sdfg.add_node(pick, is_start_block=True)
    for condition, value in [("flag > 0", 3), (None, 5)]:
        branch = ControlFlowRegion(f"lo_{value}", sdfg=sdfg)
        set_lo = branch.add_state(f"set_{value}", is_start_block=True)
        branch.add_edge(
            set_lo, branch.add_state(f"set_{value}_done"), dace.InterstateEdge(assignments={"lo": str(value)})
        )
        pick.add_branch(CodeBlock(condition) if condition else None, branch)
    kernels = sdfg.add_state_after(pick, "kernels")
    add_kernel_from(kernels, "k_first", "A", "B")
    add_kernel_from(kernels, "k_second", "B", "C")
    host = sdfg.add_state_after(kernels, "host")
    tasklet = host.add_tasklet("read_lo", {}, {"o"}, "o = lo")
    host.add_edge(tasklet, "o", host.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_a_bound_the_host_reads_after_the_kernels_stays_on_the_host():
    """The conditional assigning ``lo`` leaves the chain: a kernel cannot hand ``lo`` to the host code after it."""
    import cupy

    sdfg = kernels_after_a_conditional_bound()
    sdfg.validate()
    fused = copy.deepcopy(sdfg)
    fused.name = "megakernel_conditional_bound_fused"
    chains = MegaKernelOptimization().apply_pass(fused, {})
    fused.validate()

    assert len(chains) == 1 and len(chains[0].kernels) == 2
    assert not any(isinstance(b, ConditionalBlock) for b in chains[0].blocks)
    nested = [n for n, _ in fused.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert not any(conn.startswith("__sym_out") for n in nested for conn in n.out_connectors)

    rng = np.random.default_rng(0)
    init = {name: rng.random(1000) for name in "ABC"}
    want = {name: cupy.asarray(v) for name, v in init.items()}
    got = {name: cupy.asarray(v) for name, v in init.items()}
    out_want, out_got = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    sdfg(N=1000, flag=1, out=out_want, **want)
    with new_gpu_codegen():
        fused(N=1000, flag=1, out=out_got, **got)
    assert out_got[0] == out_want[0] == 3
    for name in "ABC":
        np.testing.assert_array_equal(cupy.asnumpy(got[name]), cupy.asnumpy(want[name]))


@pytest.mark.parametrize("program, dtype, rank, size", STENCILS, ids=STENCIL_IDS)
def test_a_cpu_stencil_time_loop_is_one_omp_team_with_two_barriers(program, dtype, rank, size):
    """On the CPU, the time loop runs in one ``omp parallel`` team: each stencil is an ``omp for nowait`` and the
    same two grid barriers become ``omp barrier``s. Static scheduling gives a thread the same iterations in every stage."""
    sdfg = program.to_sdfg(simplify=True)
    auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
    fused = copy.deepcopy(sdfg)
    fused.name = f"{sdfg.name}_cpu_megakernel"
    assert len(MegaKernelOptimization(device=dace.dtypes.DeviceType.CPU).apply_pass(fused, {})) == 1
    fused.validate()
    team = next(n for n, s in fused.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    team.map.omp_num_threads = 4

    code = fused.generate_code()[0].clean_code
    assert code.count("#pragma omp parallel") == 1
    assert code.count("#pragma omp for nowait schedule(static)") == 2 and code.count("#pragma omp barrier") == 2

    rng = np.random.default_rng(0)
    inputs = {name: rng.random((size,) * rank).astype(dtype) for name in "AB"}
    want = copy.deepcopy(inputs)
    sdfg(TSTEPS=9, N=size, **want)
    fused(TSTEPS=9, N=size, **inputs)
    for name in "AB":
        np.testing.assert_array_equal(inputs[name], want[name])


#: How stage B reads what stage A wrote (memlet, tasklet code), and whether a grid barrier must separate them under
#: column ownership
COLUMN_READS = [
    ("X[k, i]", "y = x + 1", 0),  # Same element, B's levels start one below A's
    ("X[k - 1, i]", "y = x + 1", 0),  # The level above, in the same column
    ("X[k, (i + 1) % N]", "y = x + 1", 1),  # The next column: another thread
    ("X[0:K, 0:N]", "y = x[k, idx] + 1", 1),  # A column through an index array: any thread
]
COLUMN_READ_IDS = ["same_element", "level_above", "next_column", "indirect_column"]


def two_column_stages(read: str, code: str) -> dace.SDFG:
    """Stage A writes ``X[k, i]`` on levels ``0:K``; stage B, on levels ``1:K``, writes ``Y[k, i]`` from ``read``."""
    sdfg = dace.SDFG("megakernel_two_column_stages")
    sdfg.add_symbol("K", dace.int64)
    for name in "WXY":
        sdfg.add_array(name, ["K", N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("IDX", [N], dace.int64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state("stages", is_start_block=True)
    stages = [
        ("A", "0:K", {"w": dace.Memlet("W[k, i]")}, "x = 2 * w", "x", "X"),
        ("B", "1:K", {"x": dace.Memlet(read), "idx": dace.Memlet("IDX[i]")}, code, "y", "Y"),
    ]
    for label, levels, inputs, tasklet_code, out, dst in stages:
        tasklet, entry, exit_ = state.add_mapped_tasklet(
            label,
            {"k": levels, "i": "0:N"},
            inputs,
            tasklet_code,
            {out: dace.Memlet(f"{dst}[k, i]")},
            schedule=dace.ScheduleType.GPU_Device,
            external_edges=True,
        )
        entry.map.gpu_block_size = [256, 1, 1]
    return sdfg


@pytest.mark.parametrize("read, code, barriers", COLUMN_READS, ids=COLUMN_READ_IDS)
def test_a_stage_reading_only_its_own_column_needs_no_grid_barrier(read, code, barriers):
    """With one thread per column ``i`` and the levels ``k`` a loop in that thread, the thread that writes ``X[k, i]``
    also runs every read of column ``i``, whatever level range each stage covers. Only a read of another column
    (shifted or through ``IDX``) needs a grid barrier."""
    sdfg = two_column_stages(read, code)
    sdfg.validate()
    assert (InsertGridSyncs(sync_all=False).apply_pass(sdfg, {}) or 0) == barriers


def column_stage_in_a_nested_sdfg(read: str) -> dace.SDFG:
    """Stage A writes ``X[k, i]``; stage B's body is a nested SDFG that gets all of ``X`` and ``Y`` and reads ``read``
    (in its own symbols ``kk``, ``ii``) for ``Y[kk, ii]``."""
    sdfg = two_column_stages("X[k, i]", "y = x + 1")
    state = sdfg.start_block
    b_entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label.startswith("B"))
    state.remove_nodes_from(state.scope_subgraph(b_entry).nodes())
    for node in [n for n in state.data_nodes() if state.degree(n) == 0]:
        state.remove_node(node)

    body = dace.SDFG("column_body")
    body.add_symbol("kk", dace.int64)
    body.add_symbol("ii", dace.int64)
    for name in "XY":
        body.add_array(name, ["K", N], dace.float64, storage=dace.StorageType.GPU_Global)
    inner = body.add_state("body", is_start_block=True)
    tasklet = inner.add_tasklet("B", {"x"}, {"y"}, "y = x + 1")
    inner.add_edge(inner.add_read("X"), None, tasklet, "x", dace.Memlet(read))
    inner.add_edge(tasklet, "y", inner.add_write("Y"), None, dace.Memlet("Y[kk, ii]"))

    entry, exit_ = state.add_map("B", {"k": "1:K", "i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [256, 1, 1]
    node = state.add_nested_sdfg(body, {"X"}, {"Y"}, symbol_mapping={"K": "K", "N": "N", "kk": "k", "ii": "i"})
    x_written = next(n for n in state.data_nodes() if n.data == "X")
    state.add_memlet_path(x_written, entry, node, dst_conn="X", memlet=dace.Memlet("X[0:K, 0:N]"))
    state.add_memlet_path(node, exit_, state.add_write("Y"), src_conn="Y", memlet=dace.Memlet("Y[0:K, 0:N]"))
    return sdfg


@pytest.mark.parametrize("read, barriers", [("X[kk - 1, ii]", 0), ("X[kk, (ii + 1) % N]", 1)], ids=["own", "next"])
def test_a_nested_sdfg_stage_is_keyed_by_the_memlets_inside_it(read, barriers):
    """The whole-array connectors of a nested SDFG say nothing about which elements one iteration touches; the memlets
    inside it, in the iteration's symbols, do."""
    sdfg = column_stage_in_a_nested_sdfg(read)
    sdfg.validate()
    assert (InsertGridSyncs(sync_all=False).apply_pass(sdfg, {}) or 0) == barriers


def test_a_column_keys_whichever_dimension_it_indexes():
    """``T`` keeps its levels contiguous (``T[i, k]``, the transpose of the other arrays): stage B still reads the
    column ``i`` that its own thread wrote in stage A, so no barrier is needed."""
    sdfg = two_column_stages("X[k, i]", "y = x + 1")
    sdfg.add_array("T", [N, "K"], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.start_block
    for node in state.nodes():
        for edge in state.all_edges(node):
            if edge.data.data == "X":
                edge.data = dace.Memlet(str(edge.data).replace("X[", "T[").replace("[k, i]", "[i, k]"))
        if isinstance(node, dace.nodes.AccessNode) and node.data == "X":
            node.data = "T"
    sdfg.validate()
    assert (InsertGridSyncs(sync_all=False).apply_pass(sdfg, {}) or 0) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("read, code, barriers", COLUMN_READS, ids=COLUMN_READ_IDS)
def test_column_owned_stages_match_the_kernels(read, code, barriers):
    """The megakernel runs both stages by column ownership (a thread loops over the levels of the columns it owns,
    from the first one it owns in steps of the grid), with a barrier only where B reads another column."""
    import cupy

    sdfg = two_column_stages(read, code)
    fused = copy.deepcopy(sdfg)
    fused.name = "megakernel_two_column_stages_fused"
    mk = MegaKernelOptimization()
    mk.sync_all = False
    assert len(mk.apply_pass(fused, {})) == 1
    fused.validate()
    syncs = [n for n, s in fused.all_nodes_recursive() if isinstance(n, Sync)]
    assert len(syncs) == barriers
    loops = sorted(
        (n.map.params[0], str(n.map.range[0][2]))
        for n, s in fused.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.Sequential
    )
    # Owned where that saves the barrier: the columns stride over the grid, the levels loop inside; else flattened
    owned = [("i", "__mk_nthreads"), ("i", "__mk_nthreads"), ("k", "1"), ("k", "1")]
    assert loops == (owned if barriers == 0 else [("__mk_g", "__mk_nthreads")] * 2)

    rng = np.random.default_rng(0)
    size, levels = 70001, 5
    inputs = {"W": rng.random((levels, size)), "X": np.zeros((levels, size)), "Y": np.zeros((levels, size))}
    index = rng.integers(0, size, size)
    want = {name: cupy.asarray(value) for name, value in inputs.items()}
    got = {name: cupy.asarray(value) for name, value in inputs.items()}
    sdfg(K=levels, N=size, IDX=cupy.asarray(index), **want)
    with new_gpu_codegen():
        fused(K=levels, N=size, IDX=cupy.asarray(index), **got)
    for name in "XY":
        np.testing.assert_array_equal(cupy.asnumpy(got[name]), cupy.asnumpy(want[name]))


if __name__ == "__main__":
    test_a_time_loop_of_stencil_kernels_is_one_chain(jacobi_2d_test.kernel)
    test_a_time_loop_of_stencil_kernels_is_one_chain(heat_3d_test.heat_3d_kernel)
    test_a_kernel_state_with_a_view_forms_no_chain()
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
    test_sync_all_separates_independent_stages_too()
    test_a_small_dimension_stays_a_loop_in_every_thread()
    for program in [stencil[0] for stencil in STENCILS]:
        test_a_stencil_time_loop_is_one_cooperative_launch_with_two_grid_syncs(program)
    for stencil in STENCILS:
        test_the_fused_stencil_matches_the_kernel_per_step_version(*stencil)
    test_kernels_in_a_nested_sdfg_under_a_host_loop_form_a_chain()
    test_a_megakernel_per_host_iteration_matches_the_kernels()
    test_kernels_split_by_a_reassigned_bound_become_one_state_and_one_megakernel()
    test_kernels_split_by_a_reassigned_bound_match_as_one_megakernel()
    test_a_bound_the_host_reads_after_the_kernels_stays_on_the_host()
    for stencil in STENCILS:
        test_a_cpu_stencil_time_loop_is_one_omp_team_with_two_barriers(*stencil)
    for column_read in COLUMN_READS:
        test_a_stage_reading_only_its_own_column_needs_no_grid_barrier(*column_read)
    for column_read in COLUMN_READS:
        test_column_owned_stages_match_the_kernels(*column_read)
    test_a_nested_sdfg_stage_is_keyed_by_the_memlets_inside_it("X[kk - 1, ii]", 0)
    test_a_nested_sdfg_stage_is_keyed_by_the_memlets_inside_it("X[kk, (ii + 1) % N]", 1)
    test_a_column_keys_whichever_dimension_it_indexes()
