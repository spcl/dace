import dace
import pytest


def create_assign_sdfg():
    sdfg = dace.SDFG('single_iteration_map')
    state = sdfg.add_state()
    array_size = 1
    A, _ = sdfg.add_array('A', [array_size], dace.float32)
    map_entry, map_exit = state.add_map('map_1_iter', {'i': '0:1'})
    tasklet = state.add_tasklet('set_to_1', {}, {'OUT__a'}, '_a = 1')
    map_exit.add_in_connector('IN__a')
    map_exit.add_out_connector('OUT__a')
    tasklet.add_out_connector('OUT__a')
    an = state.add_write('A')
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'OUT__a', map_exit, 'IN__a', dace.Memlet(f'A[0]'))
    state.add_edge(map_exit, 'OUT__a', an, None, dace.Memlet(f'A[0]'))
    sdfg.validate()
    return A, sdfg


def create_assign_sdfg_with_views():
    sdfg = dace.SDFG('single_iteration_map')
    state = sdfg.add_state()
    array_size = 5
    _, _ = sdfg.add_array('A', [array_size], dace.float32)
    v_A, _ = sdfg.add_view('v_A', [1], dace.float32)
    map_entry, map_exit = state.add_map('map_1_iter', {'i': '0:1'})
    tasklet = state.add_tasklet('set_to_1', {}, {'OUT__a'}, '_a = 1')
    map_exit.add_in_connector('IN__a')
    map_exit.add_out_connector('OUT__a')
    tasklet.add_out_connector('OUT__a')
    an = state.add_write('v_A')
    an2 = state.add_write('A')
    an.add_out_connector('views')
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet(None))
    state.add_edge(tasklet, 'OUT__a', map_exit, 'IN__a', dace.Memlet(f'v_A[0]'))
    state.add_edge(map_exit, 'OUT__a', an, None, dace.Memlet(f'v_A[0]'))
    state.add_edge(an, 'views', an2, None, dace.Memlet(f'A[0:1]'))
    sdfg.validate()
    return v_A, sdfg


def create_increment_sdfg():
    sdfg = dace.SDFG('increment_map')
    state = sdfg.add_state()
    array_size = 500
    A, _ = sdfg.add_array('A', [array_size], dace.float32)
    map_entry, map_exit = state.add_map('map_1_iter', {'i': f'0:{array_size}'})
    tasklet = state.add_tasklet('inc_by_1', {}, {'OUT__a'}, '_a = _a + 1')
    map_entry.add_in_connector('IN__a')
    map_entry.add_out_connector('OUT__a')
    map_exit.add_in_connector('IN__a')
    map_exit.add_out_connector('OUT__a')
    tasklet.add_in_connector('IN__a')
    tasklet.add_out_connector('OUT__a')
    an1 = state.add_read('A')
    an2 = state.add_write('A')
    state.add_edge(an1, None, map_entry, 'IN__a', dace.Memlet(f'A[0:{array_size}]'))
    state.add_edge(map_entry, 'OUT__a', tasklet, 'IN__a', dace.Memlet('A[i]'))
    state.add_edge(tasklet, 'OUT__a', map_exit, 'IN__a', dace.Memlet(f'A[i]'))
    state.add_edge(map_exit, 'OUT__a', an2, None, dace.Memlet(f'A[0:{array_size}]'))
    sdfg.validate()
    return A, sdfg


def create_increment_sdfg_with_views():
    sdfg = dace.SDFG('increment_map')
    state = sdfg.add_state()
    array_size = 500
    view_size = 100
    _, _ = sdfg.add_array('A', [array_size], dace.float32)
    v_A, _ = sdfg.add_view('v_A', [view_size], dace.float32)
    map_entry, map_exit = state.add_map('map_1_iter', {'i': f'0:{view_size}'})
    tasklet = state.add_tasklet('inc_by_1', {}, {'OUT__a'}, '_a = _a + 1')
    map_entry.add_in_connector('IN__a')
    map_entry.add_out_connector('OUT__a')
    map_exit.add_in_connector('IN__a')
    map_exit.add_out_connector('OUT__a')
    tasklet.add_in_connector('IN__a')
    tasklet.add_out_connector('OUT__a')
    an1 = state.add_read('A')
    an2 = state.add_write('A')
    an3 = state.add_read('v_A')
    an4 = state.add_write('v_A')
    an3.add_in_connector('views')
    an4.add_out_connector('views')
    state.add_edge(an1, None, an3, 'views', dace.Memlet(f'A[0:100]'))
    state.add_edge(an3, None, map_entry, 'IN__a', dace.Memlet(f'v_A[i]'))
    state.add_edge(map_entry, 'OUT__a', tasklet, 'IN__a', dace.Memlet('v_A[i]'))
    state.add_edge(tasklet, 'OUT__a', map_exit, 'IN__a', dace.Memlet(f'v_A[i]'))
    state.add_edge(map_exit, 'OUT__a', an4, None, dace.Memlet(f'v_A[i]'))
    state.add_edge(an4, 'views', an2, None, dace.Memlet(f'A[0:100]'))
    sdfg.validate()
    return v_A, sdfg


@pytest.mark.parametrize(
    "sdfg_creator",
    [create_assign_sdfg, create_increment_sdfg, create_assign_sdfg_with_views, create_increment_sdfg_with_views])
class TestHostDataHostMapParams:

    def test_host_data(self, sdfg_creator):
        """Test that arrays marked as host_data remain on host after GPU transformation."""
        A, sdfg = sdfg_creator()
        sdfg.apply_gpu_transformations(host_data=[A])
        sdfg.validate()

        assert sdfg.arrays['A'].storage != dace.dtypes.StorageType.GPU_Global

    def test_host_map(self, sdfg_creator):
        """Test that maps marked as host_maps remain on host after GPU transformation."""
        A, sdfg = sdfg_creator()
        host_maps = [n.guid for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.EntryNode)]
        sdfg.apply_gpu_transformations(host_maps=host_maps)
        sdfg.validate()
        assert sdfg.arrays['A'].storage != dace.dtypes.StorageType.GPU_Global

    @pytest.mark.parametrize("pass_empty", [True, False])
    def test_no_host_map_or_data(self, sdfg_creator, pass_empty):
        """Test default GPU transformation behavior with no host constraints."""
        A, sdfg = sdfg_creator()

        if pass_empty:
            sdfg.apply_gpu_transformations(host_maps=[], host_data=[])
        else:
            sdfg.apply_gpu_transformations()

        sdfg.validate()

        # Verify array storage locations
        assert 'A' in sdfg.arrays and 'gpu_A' in sdfg.arrays
        assert sdfg.arrays['A'].storage != dace.dtypes.StorageType.GPU_Global
        assert sdfg.arrays['gpu_A'].storage == dace.dtypes.StorageType.GPU_Global

        # Verify map schedules
        for s in sdfg.states():
            for n in s.nodes():
                if isinstance(n, dace.nodes.MapEntry):
                    assert n.map.schedule == dace.ScheduleType.GPU_Device


if __name__ == '__main__':
    pytest.main([__file__])
