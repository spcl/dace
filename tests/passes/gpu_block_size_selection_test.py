# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for domain-matched GPU device-map block-size selection.

None of these need a GPU: they build SDFGs, run the selection pass, and inspect
``map.gpu_block_size`` -- pure SDFG manipulation.
"""
import dace
from dace import dtypes, nodes, subsets
from dace.transformation.passes.gpu_block_size_selection import pick_gpu_block_size, select_gpu_device_block_size

N = dace.symbol('N')
M = dace.symbol('M')


def make_device_map(params, ranges):
    """A bare ``GPU_Device`` ``Map`` with the given params and (b, e, s) ranges."""
    return nodes.Map('kernel', list(params), subsets.Range(list(ranges)), schedule=dtypes.ScheduleType.GPU_Device)


def test_pick_block_size_1d():
    m = make_device_map(['i'], [(0, N - 1, 1)])
    assert pick_gpu_block_size(m) == [128, 1, 1]


def test_pick_block_size_2d_square_symbolic():
    # Symbolic extents are assumed large and roughly equal -> square block.
    m = make_device_map(['i', 'j'], [(0, N - 1, 1), (0, M - 1, 1)])
    assert pick_gpu_block_size(m) == [16, 16, 1]


def test_pick_block_size_2d_square_constant():
    m = make_device_map(['i', 'j'], [(0, 1023, 1), (0, 1023, 1)])
    assert pick_gpu_block_size(m) == [16, 16, 1]


def test_pick_block_size_2d_skewed_last_dim_wide():
    # Last (contiguous, threadIdx.x) dimension is 4x larger -> 32 on x.
    m = make_device_map(['i', 'j'], [(0, 255, 1), (0, 1023, 1)])
    assert pick_gpu_block_size(m) == [32, 16, 1]


def test_pick_block_size_2d_skewed_outer_dim_wide():
    # Outer (threadIdx.y) dimension is 4x larger -> 32 on y.
    m = make_device_map(['i', 'j'], [(0, 1023, 1), (0, 255, 1)])
    assert pick_gpu_block_size(m) == [16, 32, 1]


def test_pick_block_size_2d_mild_ratio_stays_square():
    # Ratio below the skew threshold (2x) stays square.
    m = make_device_map(['i', 'j'], [(0, 599, 1), (0, 1023, 1)])
    assert pick_gpu_block_size(m) == [16, 16, 1]


def test_pick_block_size_3d_leaves_default():
    m = make_device_map(['i', 'j', 'k'], [(0, N - 1, 1), (0, N - 1, 1), (0, N - 1, 1)])
    assert pick_gpu_block_size(m) is None


def build_2d_map_sdfg(schedule):
    """Minimal SDFG: one 64x64 map writing a 2-D array on the given schedule."""
    sdfg = dace.SDFG('single_map')
    sdfg.add_array('A', [64, 64], dace.float64)
    state = sdfg.add_state()
    _, me, _ = state.add_mapped_tasklet('kernel', {
        'i': '0:64',
        'j': '0:64'
    }, {},
                                        'o = 1.0', {'o': dace.Memlet('A[i, j]')},
                                        schedule=schedule,
                                        external_edges=True)
    return sdfg, me


def device_maps(sdfg):
    return [
        n.map for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def test_pass_assigns_2d_square():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    assigned = select_gpu_device_block_size(sdfg)
    assert assigned == {'kernel_map': [16, 16, 1]}
    assert device_maps(sdfg)[0].gpu_block_size == [16, 16, 1]


def test_pass_does_not_override_user_block_size():
    sdfg, me = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    me.map.gpu_block_size = [8, 8, 1]
    assert select_gpu_device_block_size(sdfg) == {}
    assert device_maps(sdfg)[0].gpu_block_size == [8, 8, 1]


def test_pass_ignores_cpu_map():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.CPU_Multicore)
    assert select_gpu_device_block_size(sdfg) == {}


def test_gpu_transform_end_to_end():
    # Build a plain 2-D map, GPU-transform the SDFG, then select block sizes.
    @dace.program
    def prog2d(a: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            a[i, j] = 1.0

    sdfg = prog2d.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    dmaps = device_maps(sdfg)
    assert len(dmaps) == 1
    assert dmaps[0].gpu_block_size is None  # transform leaves it unset
    select_gpu_device_block_size(sdfg)
    assert dmaps[0].gpu_block_size == [16, 16, 1]


def test_gpu_transform_1d_end_to_end():

    @dace.program
    def prog1d(a: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = 1.0

    sdfg = prog1d.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    dmaps = device_maps(sdfg)
    assert len(dmaps) == 1
    select_gpu_device_block_size(sdfg)
    assert dmaps[0].gpu_block_size == [128, 1, 1]


def test_config_default_block_size_is_128():
    assert dace.Config.get('compiler', 'cuda', 'default_block_size') == '128,1,1'


def test_pass_is_idempotent():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    select_gpu_device_block_size(sdfg)
    # Re-running assigns nothing (already set) and preserves the value.
    assert select_gpu_device_block_size(sdfg) == {}
    assert device_maps(sdfg)[0].gpu_block_size == [16, 16, 1]


if __name__ == '__main__':
    test_pick_block_size_1d()
    test_pick_block_size_2d_square_symbolic()
    test_pick_block_size_2d_square_constant()
    test_pick_block_size_2d_skewed_last_dim_wide()
    test_pick_block_size_2d_skewed_outer_dim_wide()
    test_pick_block_size_2d_mild_ratio_stays_square()
    test_pick_block_size_3d_leaves_default()
    test_pass_assigns_2d_square()
    test_pass_does_not_override_user_block_size()
    test_pass_ignores_cpu_map()
    test_gpu_transform_end_to_end()
    test_gpu_transform_1d_end_to_end()
    test_config_default_block_size_is_128()
    test_pass_is_idempotent()
    print('OK')
