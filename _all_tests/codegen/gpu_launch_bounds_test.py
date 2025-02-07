# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest


@pytest.mark.gpu
def test_launch_bounds_default():
    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:100, 0:20] @ dace.ScheduleType.GPU_Device:
            a[i, j] = 1

    with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value='32,2,1'):
        assert '__launch_bounds__(64)' in prog.to_sdfg().generate_code()[1].code


@pytest.mark.gpu
def test_launch_bounds_implicit():
    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:50, 0:10] @ dace.ScheduleType.GPU_Device:
            for bi, bj in dace.map[0:2, 0:2] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i * 2 + bi, j * 2 + bj] = 1

    assert '__launch_bounds__(4)' in prog.to_sdfg().generate_code()[1].code


@pytest.mark.gpu
def test_launch_bounds_implicit_sym():
    B = dace.symbol('B')

    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:50, 0:10] @ dace.ScheduleType.GPU_Device:
            for bi, bj in dace.map[0:B, 0:B] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i * B + bi, j * B + bj] = 1

    assert '__launch_bounds__' not in prog.to_sdfg().generate_code()[1].code


@pytest.mark.gpu
def test_launch_bounds_explicit():
    B = 2

    @dace.program
    def prog(a: dace.float64[100, 20] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:50, 0:10] @ dace.ScheduleType.GPU_Device:
            for bi, bj in dace.map[0:B, 0:B] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i * B + bi, j * B + bj] = 1

    sdfg = prog.to_sdfg()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device:
            mapentry = n
            break

    mapentry.map.gpu_launch_bounds = '-1'
    assert '__launch_bounds__' not in sdfg.generate_code()[1].code
    mapentry.map.gpu_launch_bounds = '5, 1'
    assert '__launch_bounds__(5, 1)' in sdfg.generate_code()[1].code


if __name__ == '__main__':
    test_launch_bounds_default()
    test_launch_bounds_implicit()
    test_launch_bounds_implicit_sym()
    test_launch_bounds_explicit()
