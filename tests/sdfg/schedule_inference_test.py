# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for default storage/schedule inference. """
import dace
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.sdfg.infer_types import set_default_schedule_and_storage_types
from dace.transformation.helpers import get_parent_map
import pytest


def test_default_schedule_autodetect():
    @dace.program
    def add(a: dace.float32[10, 10], b: dace.float32[10, 10]):
        return a + b @ b

    sdfg = add.to_sdfg()
    set_default_schedule_and_storage_types(sdfg, None)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.MapEntry)):
            assert node.schedule == dace.ScheduleType.CPU_Multicore


def test_gpu_schedule_autodetect():
    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global,
            b: dace.float32[10, 10] @ dace.StorageType.GPU_Global):
        return a + b @ b

    sdfg = add.to_sdfg()
    set_default_schedule_and_storage_types(sdfg, None)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.MapEntry)):
            assert node.schedule == dace.ScheduleType.GPU_Device


def test_gpu_schedule_scalar_autodetect():
    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global,
            b: dace.float32[10, 10] @ dace.StorageType.GPU_Global, c: dace.float32[10] @ dace.StorageType.CPU_Heap):
        return a + b @ b + c

    sdfg = add.to_sdfg()
    set_default_schedule_and_storage_types(sdfg, None)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.MapEntry)):
            assert node.schedule == dace.ScheduleType.GPU_Device


def test_gpu_schedule_scalar_autodetect_2():
    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global, b: dace.float32):
        return a + b

    sdfg = add.to_sdfg()
    set_default_schedule_and_storage_types(sdfg, None)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.MapEntry)):
            assert node.schedule == dace.ScheduleType.GPU_Device


def test_nested_kernel_computation():
    @dace.program
    def nested(a, b):
        return a @ b

    @dace.program
    def top(a: dace.float64[20, 20, 300], b: dace.float64[20, 20, 300], c: dace.float64[20, 20, 300]):
        for i in dace.map[0:30] @ dace.ScheduleType.GPU_Device:
            for j in dace.map[0:10] @ dace.ScheduleType.Sequential:
                c[:, :, i * 10 + j] = nested(a[:, :, i * 10 + j], b[:, :, i * 10 + j])

    sdfg = top.to_sdfg(simplify=False)

    set_default_schedule_and_storage_types(sdfg, None)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.LibraryNode):
            assert node.schedule == dace.ScheduleType.GPU_Device


def test_nested_map_in_loop_schedule():
    @dace.program
    def top(a: dace.float64[20, 20], b: dace.float64[20, 20], c: dace.float64[20, 20]):
        for i in dace.map[0:20] @ dace.ScheduleType.GPU_Device:
            for _ in range(5):
                c[i] += a[i] + b[i]

    sdfg = top.to_sdfg(simplify=False)

    set_default_schedule_and_storage_types(sdfg, None)
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if get_parent_map(state, node) is None:
                assert node.schedule == dace.ScheduleType.GPU_Device
            else:
                assert node.schedule == dace.ScheduleType.Sequential


def test_ambiguous_schedule():
    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global, b: dace.float32[10, 10]):
        return a + b

    with pytest.raises(InvalidSDFGNodeError):
        sdfg = add.to_sdfg()
        set_default_schedule_and_storage_types(sdfg, None)


def test_semi_ambiguous_schedule():
    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global,
            b: dace.float32[10, 10] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:10] @ dace.ScheduleType.GPU_Device:
            shared = dace.define_local([10], dace.float32)
            for j in dace.map[0:10]:  # Should be inferred as thread-block
                b[i, j] = a[i, j] + shared[j]

    sdfg = add.to_sdfg()
    set_default_schedule_and_storage_types(sdfg, None)
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if get_parent_map(state, node) is None:
                assert node.schedule == dace.ScheduleType.GPU_Device
            else:
                assert node.schedule == dace.ScheduleType.GPU_ThreadBlock


if __name__ == '__main__':
    test_default_schedule_autodetect()
    test_gpu_schedule_autodetect()
    test_gpu_schedule_scalar_autodetect()
    test_gpu_schedule_scalar_autodetect_2()
    test_nested_kernel_computation()
    test_nested_map_in_loop_schedule()
    test_ambiguous_schedule()
    test_semi_ambiguous_schedule()
