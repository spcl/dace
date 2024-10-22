import dace

def create_assign_sdfg():
    sdfg = dace.SDFG('single_iteration_map')
    state = sdfg.add_state()
    array_size = 1
    A = state.add_array('A', [array_size], dace.float32)
    map_entry, map_exit = state.add_map('map_1_iter', {'i': '0:1'})
    tasklet = state.add_tasklet('set_to_1', {}, {'OUT__a'}, '_a = 1')
    map_exit.add_in_connector('IN__a')
    map_exit.add_out_connector('OUT__a')
    tasklet.add_out_connector('OUT__a')
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'OUT__a', map_exit, 'IN__a', dace.Memlet(f'A[0]'))
    state.add_edge(map_exit, 'OUT__a', A, None, dace.Memlet(f'A[0]'))
    sdfg.validate()

    return A, sdfg

def test_host_data():
    A, sdfg = create_assign_sdfg()
    sdfg.apply_gpu_transformations(host_data=['A'])
    sdfg.validate()
    sdfg.save(f"s_gpu1.sdfg")
    assert sdfg.arrays[A.data].storage != dace.dtypes.StorageType.GPU_Global

def test_host_map():
    A, sdfg = create_assign_sdfg()
    host_maps = []
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.EntryNode):
                host_maps.append(n.guid)

    sdfg.apply_gpu_transformations(host_maps=host_maps)
    sdfg.validate()
    sdfg.save(f"s_gpu2.sdfg")
    assert sdfg.arrays[A.data].storage != dace.dtypes.StorageType.GPU_Global

def test_no_host_map_or_data(pass_empty=False):
    A, sdfg = create_assign_sdfg()
    if pass_empty:
        host_maps = []
        host_data = []
        sdfg.apply_gpu_transformations(host_maps=host_maps, host_data=host_data)
    else:
        sdfg.apply_gpu_transformations()
    sdfg.validate()
    sdfg.save(f"s_gpu3_{pass_empty}.sdfg")

    assert sdfg.arrays[A.data].storage == dace.dtypes.StorageType.GPU_Global
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.nodes.MapEntry):
                assert n.map.schedule == dace.ScheduleType.GPU_Device

if __name__ == '__main__':
    test_host_data()
    test_host_map()
    test_no_host_map_or_data(True)
    test_no_host_map_or_data(False)