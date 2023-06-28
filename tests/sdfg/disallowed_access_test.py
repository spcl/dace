# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.validation import InvalidSDFGInterstateEdgeError, InvalidSDFGEdgeError
import pytest


@pytest.mark.gpu
def test_gpu_access_on_host_interstate_ok():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map('map', dict(i='0:20'), dace.ScheduleType.GPU_Device)

    nsdfg = dace.SDFG('inner')
    nsdfg.add_array('a', [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state1 = nsdfg.add_state()
    state2 = nsdfg.add_state()
    nsdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(s='a[i]')))

    nnode = state.add_nested_sdfg(nsdfg, None, {'a'}, {}, {'i': 'i'})
    r = state.add_read('A')
    state.add_memlet_path(r, me, nnode, dst_conn='a', memlet=dace.Memlet('A[0:20]'))
    state.add_nedge(nnode, mx, dace.Memlet())

    sdfg.validate()


@pytest.mark.gpu
def test_gpu_access_on_host_interstate_invalid():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64, storage=dace.StorageType.GPU_Global)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(s='A[4]')))

    with pytest.raises(InvalidSDFGInterstateEdgeError):
        sdfg.validate()


@pytest.mark.gpu
def test_gpu_access_on_host_tasklet():
    @dace.program
    def tester(a: dace.float64[20] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:20] @ dace.ScheduleType.CPU_Multicore:
            a[i] = 1

    with pytest.raises(InvalidSDFGEdgeError):
        tester.to_sdfg(validate=True)


if __name__ == '__main__':
    # test_gpu_access_on_host_interstate_ok()
    test_gpu_access_on_host_interstate_invalid()
    # test_gpu_access_on_host_tasklet()
    