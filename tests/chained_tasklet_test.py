#!/usr/bin/env python
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet
from dace.data import Scalar

# Constructs an SDFG with two consecutive tasklets
if __name__ == '__main__':
    print('SDFG consecutive tasklet test')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)
    input = dp.ndarray([N], dp.int32)
    output = dp.ndarray([N], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG
    mysdfg = SDFG('ctasklet')
    state = mysdfg.add_state()
    A_ = state.add_array('A', [N], dp.int32)
    B_ = state.add_array('B', [N], dp.int32)
    mysdfg.add_scalar('something', dp.int32)

    map_entry, map_exit = state.add_map('mymap', dict(i='0:N'))
    tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'i'))
    tasklet2 = state.add_tasklet('mytasklet2', {'c'}, {'d'}, 'd = 2*c')
    state.add_edge(tasklet, 'b', tasklet2, 'c',
                   Memlet.simple('something', '0'))
    state.add_edge(tasklet2, 'd', map_exit, None, Memlet.simple(B_, 'i'))

    # Add outer edges
    state.add_edge(A_, None, map_entry, None, Memlet.simple(A_, '0:N'))
    state.add_edge(map_exit, None, B_, None, Memlet.simple(B_, '0:N'))

    # Left for debugging purposes
    mysdfg.draw_to_file()

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(10 * input - output) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
