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
        return a + b @ b + c[0]

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
                assert node.schedule == dace.ScheduleType.GPU_ThreadBlock


def test_nested_storage():

    @dace.program
    def nested(a: dace.float64[20, 20], b: dace.float64[20, 20]):
        tmp = dace.define_local([20, 20], dace.float64)
        tmp[:] = a
        b[:] = tmp

    @dace.program
    def top(a: dace.float64[20, 20], b: dace.float64[20, 20]):
        nested(a, b)

    sdfg = top.to_sdfg(simplify=False)

    set_default_schedule_and_storage_types(sdfg, None)
    for node, state in sdfg.all_nodes_recursive():
        nsdfg = state.parent
        if isinstance(node, dace.nodes.AccessNode):
            assert node.desc(nsdfg).storage == dace.StorageType.CPU_Heap


def test_nested_storage_equivalence():

    @dace.program
    def nested(a: dace.float64[20, 20], b: dace.float64[20, 20]):
        b[:] = a

    @dace.program
    def top(a: dace.float64[20, 20] @ dace.StorageType.CPU_Heap, b: dace.float64[20, 20] @ dace.StorageType.CPU_Pinned):
        nested(a, b)

    sdfg = top.to_sdfg(simplify=False)

    set_default_schedule_and_storage_types(sdfg, None)
    for node, state in sdfg.all_nodes_recursive():
        nsdfg = state.parent
        if isinstance(node, dace.nodes.AccessNode):
            if state.out_degree(node) > 0:  # Check for a in external and internal scopes
                assert node.desc(nsdfg).storage == dace.StorageType.CPU_Heap
            elif state.in_degree(node) > 0:  # Check for b in external and internal scopes
                assert node.desc(nsdfg).storage == dace.StorageType.CPU_Pinned


def test_ambiguous_schedule():

    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global, b: dace.float32[10, 10]):
        return a + b

    with pytest.raises(InvalidSDFGNodeError):
        sdfg = add.to_sdfg()
        set_default_schedule_and_storage_types(sdfg, None)


def test_ambiguous_schedule_2():

    @dace.program
    def add(a: dace.float32[10, 10] @ dace.StorageType.GPU_Global,
            b: dace.float32[10, 10] @ dace.StorageType.GPU_Global, c: dace.float32[10] @ dace.StorageType.CPU_Heap):
        return a + b @ b + c

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
    test_nested_storage()
    test_nested_storage_equivalence()
    test_ambiguous_schedule()
    test_ambiguous_schedule_2()
    test_semi_ambiguous_schedule()
