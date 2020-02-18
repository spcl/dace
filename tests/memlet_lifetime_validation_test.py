#!/usr/bin/env python
import numpy as np

import dace as dp
from dace.sdfg import SDFG, InvalidSDFGError
from dace.memlet import Memlet
from dace.data import Scalar

if __name__ == '__main__':
    print('SDFG memlet lifetime validation test')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)
    input = dp.ndarray([N], dp.int32)
    output = dp.ndarray([N], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG 1
    # sdfg1 = SDFG('shouldntwork1')
    # state = sdfg1.add_state()
    # B = state.add_array('B', [N], dp.int32)
    # T = state.add_transient('T', [1], dp.int32)
    #
    # tasklet_gen = state.add_tasklet('mytasklet', {}, {'b'}, 'b = 5')
    # map_entry, map_exit = state.add_map('mymap', dict(k='0:N'))
    #
    # map_entry._in_connectors.add('IN_1')
    # map_entry._out_connectors.add('OUT_1')
    # map_exit._in_connectors.add('IN_1')
    # map_exit._out_connectors.add('OUT_1')
    #
    # state.add_edge(tasklet_gen, 'b', map_entry, 'IN_1', Memlet.simple(T, '0'))
    # state.add_edge(map_entry, 'OUT_1', T, None, Memlet.simple(T, '0'))
    # state.add_edge(T, None, map_exit, 'IN_1', Memlet.simple(B, '0'))
    # state.add_edge(map_exit, 'OUT_1', B, None, Memlet.simple(B, '0'))
    #
    # # Left for debugging purposes
    # sdfg1.draw_to_file()
    #
    # try:
    #     sdfg1.validate()
    #     print("SDFG 1 passed validation, test FAILED")
    #     exit(1)
    # except InvalidSDFGError:
    #     print("Test 1 passed, exception successfully caught")

    # Construct SDFG 2
    sdfg2 = SDFG('shouldntwork2')
    state = sdfg2.add_state()
    A = state.add_array('A', [N], dp.int32)
    B = state.add_array('B', [N], dp.int32)
    T = state.add_transient('T', [1], dp.int32)

    tasklet_gen = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    map_entry, map_exit = state.add_map('mymap', dict(k='0:N'))

    map_entry._in_connectors.add('IN_1')
    map_entry._out_connectors.add('OUT_1')
    map_exit._in_connectors.add('IN_1')
    map_exit._out_connectors.add('OUT_1')

    state.add_edge(B, None, map_entry, 'IN_1', Memlet.simple(B, '0'))
    state.add_edge(map_entry, 'OUT_1', T, None, Memlet.simple(T, '0'))
    state.add_edge(T, None, map_exit, 'IN_1', Memlet.simple(B, '0'))
    state.add_edge(map_exit, 'OUT_1', tasklet_gen, 'a', Memlet.simple(B, '0'))
    state.add_edge(tasklet_gen, 'b', A, None, Memlet.simple(A, '0'))

    # Left for debugging purposes
    sdfg2.draw_to_file()

    try:
        sdfg2.validate()
        print("SDFG 2 passed validation, test FAILED")
        exit(1)
    except InvalidSDFGError:
        print("Test 2 passed, exception successfully caught")

    # Construct SDFG 3
    sdfg3 = SDFG('shouldntwork2')
    state = sdfg3.add_state()
    A = state.add_array('A', [N], dp.int32)
    B = state.add_array('B', [N], dp.int32)
    T = state.add_transient('T', [N], dp.int32)

    tasklet_gen = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    map1_entry, map1_exit = state.add_map('mymap1', dict(k='0:N'))
    map2_entry, map2_exit = state.add_map('mymap2', dict(k='0:N'))

    map1_entry._in_connectors.add('IN_1')
    map1_entry._out_connectors.add('OUT_1')
    map1_exit._in_connectors.add('IN_1')
    map1_exit._out_connectors.add('OUT_1')
    map2_entry._in_connectors.add('IN_1')
    map2_entry._out_connectors.add('OUT_1')
    map2_exit._in_connectors.add('IN_1')
    map2_exit._out_connectors.add('OUT_1')

    state.add_edge(A, None, map1_entry, 'IN_1', Memlet.simple(A, '0:N'))
    state.add_edge(map1_entry, 'OUT_1', tasklet_gen, 'a', Memlet.simple(
        A, 'i'))
    state.add_edge(tasklet_gen, 'b', map1_exit, 'IN_1', Memlet.simple(T, 'i'))
    state.add_edge(map1_exit, 'OUT_1', map2_entry, 'IN_1',
                   Memlet.simple(T, '0:N'))
    state.add_edge(map2_entry, 'OUT_1', T, None, Memlet.simple(T, 'i'))
    state.add_edge(T, None, map2_exit, 'IN_1', Memlet.simple(B, 'i'))
    state.add_edge(map2_exit, 'OUT_1', B, None, Memlet.simple(B, '0:N'))

    # Left for debugging purposes
    sdfg3.draw_to_file()

    try:
        sdfg3.validate()
        print("SDFG 3 passed validation, test FAILED")
        exit(1)
    except InvalidSDFGError:
        print("Test 3 passed, exception successfully caught")

    exit(0)
