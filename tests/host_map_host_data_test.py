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
    assert sdfg.arrays[A.data].storage != dace.dtypes.StorageType.GPU_Global

if __name__ == '__main__':
    test_host_data()
    test_host_map()