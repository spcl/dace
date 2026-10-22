# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the lowering of dynamic thread-block maps into thread-block maps that drive a work queue."""

import pytest

import dace
from dace.codegen import common
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.lower_dynamic_thread_block_maps import LowerDynamicThreadBlockMaps

H = dace.symbol("H")
W = dace.symbol("W")
nnz = dace.symbol("nnz")


def spmv_sdfg(name: str) -> dace.SDFG:
    """A CSR matrix-vector product whose row loop is a GPU kernel and whose inner map is a dynamic thread-block map."""

    @dace.program
    def spmv(
        A_row: dace.int32[H + 1],
        A_col: dace.int32[nnz],
        A_val: dace.float32[nnz],
        x: dace.float32[W],
        b: dace.float32[H],
    ):
        for i in dace.map[0:H]:
            for j in dace.map[A_row[i] : A_row[i + 1]]:
                b[i] += A_val[j] * x[A_col[j]]

    sdfg = spmv.to_sdfg(simplify=True)
    sdfg.name = name
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dace.ScheduleType.Sequential:
            node.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic
    return sdfg


def maps_of(sdfg: dace.SDFG, schedule: dace.ScheduleType) -> list[nodes.MapEntry]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.map.schedule == schedule]


def test_the_kernel_gets_a_thread_block_map_and_the_dynamic_map_a_queue():
    sdfg = spmv_sdfg("lower_dynamic_spmv")
    with dace.config.set_temporary("compiler", "cuda", "dynamic_map_block_size", value="128,1,1"):
        assert LowerDynamicThreadBlockMaps().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    assert not maps_of(sdfg, dace.ScheduleType.GPU_ThreadBlock_Dynamic)
    (thread_block,) = maps_of(sdfg, dace.ScheduleType.GPU_ThreadBlock)
    assert thread_block.map.range.num_elements() == 128
    # One round loop per phase of the queue, each with a sequential copy of the map for its batches
    rounds = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert sorted(r.label.rsplit("_", 2)[-2] for r in rounds) == ["block", "thread", "warp"]
    assert len(maps_of(sdfg, dace.ScheduleType.Sequential)) == 3
    # Out of bounds, a thread reads no bounds of its own (``A_row[i]``)
    guards = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]
    assert any("i <= (H - 1)" in condition.as_string for g in guards for condition, _ in g.branches if condition)


def test_a_kernel_without_dynamic_maps_is_left_alone():
    sdfg = spmv_sdfg("lower_dynamic_untouched")
    for entry in maps_of(sdfg, dace.ScheduleType.GPU_ThreadBlock_Dynamic):
        entry.map.schedule = dace.ScheduleType.Sequential
    assert LowerDynamicThreadBlockMaps().apply_pass(sdfg, {}) is None
    assert not maps_of(sdfg, dace.ScheduleType.GPU_ThreadBlock)


@pytest.mark.parametrize("backend,warp_size,state", [("cuda", 32, 8192), ("hip", 64, 4)])
def test_the_queue_follows_the_warp_size(backend: str, warp_size: int, state: int):
    """Fine-grained, CUDA lists the items of each warp in shared memory; HIP's 64-wide warps shuffle instead."""
    sdfg = spmv_sdfg(f"lower_dynamic_warp_{backend}")
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "cuda", "backend", value=backend)
        dace.config.Config.set("compiler", "cuda", "dynamic_map_block_size", value="128,1,1")
        common.get_gpu_backend.cache_clear()
        try:
            code = "\n".join(c.code for c in sdfg.generate_code())
        finally:
            common.get_gpu_backend.cache_clear()
    assert f"dace::DynamicMap<true, 128, {warp_size}, int>::block_round(" in code
    assert f"__shared__ int __dace_dynmap_state[{state}];" in code


if __name__ == "__main__":
    test_the_kernel_gets_a_thread_block_map_and_the_dynamic_map_a_queue()
    test_a_kernel_without_dynamic_maps_is_left_alone()
    test_the_queue_follows_the_warp_size("cuda", 32, 8192)
    test_the_queue_follows_the_warp_size("hip", 64, 4)
