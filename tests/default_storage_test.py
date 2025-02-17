# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.infer_types import set_default_schedule_and_storage_types


def test_notbmap():
    sdfg = dace.SDFG('default_storage_test_1')
    sdfg.add_array('A', [20], dace.float64, dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    me, mx = state.add_map('kernel', dict(i='0:20'), dace.ScheduleType.GPU_Device)
    tmp = state.add_access('tmp')
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    w = state.add_write('A')

    state.add_memlet_path(r, me, tmp, memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(tmp, t, dst_conn='a', memlet=dace.Memlet.simple('tmp', '0'))
    state.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    set_default_schedule_and_storage_types(sdfg, None)
    assert sdfg.arrays['tmp'].storage == dace.StorageType.Register


def test_tbmap_sequential():
    sdfg = dace.SDFG('default_storage_test_2')
    sdfg.add_array('A', [20, 32], dace.float64, dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    ome, omx = state.add_map('kernel', dict(i='0:20'), dace.ScheduleType.GPU_Device)
    sme, smx = state.add_map('seq', dict(j='0:1'), dace.ScheduleType.Sequential)
    ime, imx = state.add_map('block', dict(ti='0:32'), dace.ScheduleType.GPU_ThreadBlock)
    tmp = state.add_access('tmp')
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    w = state.add_write('A')

    state.add_memlet_path(r, ome, sme, tmp, memlet=dace.Memlet.simple('A', 'i+j, 0:32'))
    state.add_memlet_path(tmp, ime, t, dst_conn='a', memlet=dace.Memlet.simple('tmp', '0, ti'))
    state.add_memlet_path(t, imx, smx, omx, w, src_conn='b', memlet=dace.Memlet.simple('A', 'i+j, ti'))
    set_default_schedule_and_storage_types(sdfg, None)
    assert sdfg.arrays['tmp'].storage == dace.StorageType.GPU_Shared


if __name__ == '__main__':
    test_notbmap()
    test_tbmap_sequential()
