# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
from dace.sdfg import SDFG, InvalidSDFGError
from dace.memlet import Memlet

import pytest


def test():
    # Externals (parameters, symbols)
    N = dp.symbol('N')

    # Construct SDFG 1
    sdfg1 = SDFG('shouldntwork1')
    sdfg1.add_array('A', [N], dp.int32)
    sdfg1.add_array('B', [N], dp.int32)
    sdfg1.add_transient('T', [1], dp.int32)
    state = sdfg1.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    T = state.add_access('T')

    tasklet_gen = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    map_entry, map_exit = state.add_map('mymap', dict(k='0:N'))

    map_entry.add_in_connector('IN_1')
    map_entry.add_out_connector('OUT_1')
    map_exit.add_in_connector('IN_1')
    map_exit.add_out_connector('OUT_1')

    state.add_edge(B, None, map_entry, 'IN_1', Memlet.simple(B, '0'))
    state.add_edge(map_entry, 'OUT_1', T, None, Memlet.simple(T, '0'))
    state.add_edge(T, None, map_exit, 'IN_1', Memlet.simple(B, '0'))
    state.add_edge(map_exit, 'OUT_1', tasklet_gen, 'a', Memlet.simple(B, '0'))
    state.add_edge(tasklet_gen, 'b', A, None, Memlet.simple(A, '0'))

    with pytest.raises(InvalidSDFGError):
        sdfg1.validate()

    # Construct SDFG 3
    sdfg2 = SDFG('shouldntwork2')
    sdfg2.add_array('A', [N], dp.int32)
    sdfg2.add_array('B', [N], dp.int32)
    sdfg2.add_transient('T', [N], dp.int32)
    state = sdfg2.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    T = state.add_access('T')

    tasklet_gen = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    map1_entry, map1_exit = state.add_map('mymap1', dict(k='0:N'))
    map2_entry, map2_exit = state.add_map('mymap2', dict(k='0:N'))

    map1_entry.add_in_connector('IN_1')
    map1_entry.add_out_connector('OUT_1')
    map1_exit.add_in_connector('IN_1')
    map1_exit.add_out_connector('OUT_1')
    map2_entry.add_in_connector('IN_1')
    map2_entry.add_out_connector('OUT_1')
    map2_exit.add_in_connector('IN_1')
    map2_exit.add_out_connector('OUT_1')

    state.add_edge(A, None, map1_entry, 'IN_1', Memlet.simple(A, '0:N'))
    state.add_edge(map1_entry, 'OUT_1', tasklet_gen, 'a', Memlet.simple(A, 'i'))
    state.add_edge(tasklet_gen, 'b', map1_exit, 'IN_1', Memlet.simple(T, 'i'))
    state.add_edge(map1_exit, 'OUT_1', map2_entry, 'IN_1', Memlet.simple(T, '0:N'))
    state.add_edge(map2_entry, 'OUT_1', T, None, Memlet.simple(T, 'i'))
    state.add_edge(T, None, map2_exit, 'IN_1', Memlet.simple(B, 'i'))
    state.add_edge(map2_exit, 'OUT_1', B, None, Memlet.simple(B, '0:N'))

    with pytest.raises(InvalidSDFGError):
        sdfg2.validate()


if __name__ == '__main__':
    test()
