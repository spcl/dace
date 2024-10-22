# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.infer_types import set_default_schedule_and_storage_types


def test_assign_map_with_deferred_allocation():
    sdfg = dace.SDFG('assign_with_deferred_allocation')
    sdfg.add_array('A', '_dace_defer', dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()

    an_A = dace.nodes.AccessNode(data='A')
    state.add_node(an_A)
    an_A.add_in_connector('IN_size')

    #me, mx = state.add_map('kernel', dict(i='0:20'), dace.ScheduleType.CPU_Heap)
    #tmp = state.add_access('tmp')
    #t = state.add_tasklet('assign', {'a'}, {'b'}, 'b = a')
    #w = state.add_write('A')

    #state.add_memlet_path(r, me, tmp, memlet=dace.Memlet.simple('A', 'i'))
    #state.add_memlet_path(tmp, t, dst_conn='a', memlet=dace.Memlet.simple('tmp', '0'))
    #state.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    sdfg.save("tmp.sdfg")


if __name__ == '__main__':
    test_assign_map_with_deferred_allocation()
