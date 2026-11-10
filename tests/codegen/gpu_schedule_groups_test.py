# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU schedule groups: kernel schedules, thread schedules, and the schedules without code generation yet."""

import pytest

import dace
from dace import dtypes
from dace.sdfg import infer_types

NEW_SCHEDULES = [
    dtypes.ScheduleType.GPU_Chiplet,
    dtypes.ScheduleType.GPU_ThreadBlockCluster,
    dtypes.ScheduleType.GPU_Warp,
]


def test_every_gpu_schedule_is_a_kernel_a_chiplet_or_a_thread_schedule_exactly_once():
    groups = dtypes.GPU_KERNEL_SCHEDULES + [dtypes.ScheduleType.GPU_Chiplet] + dtypes.GPU_THREAD_SCHEDULES
    assert sorted(groups, key=str) == sorted(dtypes.GPU_SCHEDULES, key=str)
    assert len(set(groups)) == len(groups)


def kernel_with_inner_map(inner: dtypes.ScheduleType) -> dace.SDFG:
    """A GPU_Device map over 64 elements holding one map of schedule ``inner`` that copies them."""
    sdfg = dace.SDFG(f"kernel_with_{inner.name.lower()}_map")
    sdfg.add_array("x", [64], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("y", [64], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("kernel", dict(i="0:64:32"), schedule=dtypes.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:32"), schedule=inner)
    tasklet = state.add_tasklet("copy", {"a"}, {"b"}, "b = a")
    state.add_memlet_path(
        state.add_read("x"), outer_entry, inner_entry, tasklet, dst_conn="a", memlet=dace.Memlet("x[i + j]")
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("y"), src_conn="b", memlet=dace.Memlet("y[i + j]")
    )
    return sdfg


@pytest.mark.parametrize(
    "schedule",
    [
        # The experimental CUDA codegen emits GPU_Warp maps
        pytest.param(s, marks=pytest.mark.old_gpu_codegen_only) if s == dtypes.ScheduleType.GPU_Warp else s
        for s in NEW_SCHEDULES
    ],
    ids=lambda s: s.name,
)
def test_a_schedule_without_code_generation_is_refused_by_name(schedule):
    """Generating code for one of them fails loudly, naming the schedule, instead of emitting something else."""
    sdfg = kernel_with_inner_map(schedule)
    with pytest.raises(NotImplementedError, match=schedule.name):
        sdfg.generate_code()


def test_schedule_inference_picks_thread_blocks_inside_a_kernel():
    """A Default map nested in a kernel becomes a thread-block map, never one of the schedules without code
    generation."""
    sdfg = kernel_with_inner_map(dtypes.ScheduleType.Default)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    inner = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and n.label == "inner")
    assert inner.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
