#!/usr/bin/env python
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

# Constructs an SDFG with two consecutive tasklets
if __name__ == '__main__':
    print('Multidimensional offset and stride test')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)
    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([4, 3], dp.float32)
    input[:] = (np.random.rand(N.get(), N.get()) * 5).astype(dp.float32.type)
    output[:] = dp.float32(0)

    # Construct SDFG
    mysdfg = SDFG('offset_stride')
    state = mysdfg.add_state()
    A_ = state.add_array('A', [6, 6],
                         dp.float32,
                         offset=[2, 3],
                         strides=[N, 1],
                         total_size=N * N)
    B_ = state.add_array('B', [3, 2],
                         dp.float32,
                         offset=[-1, -1],
                         strides=[3, 1],
                         total_size=12)

    map_entry, map_exit = state.add_map('mymap', [('i', '1:4'), ('j', '1:3')])
    tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = a')
    state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'i,j'))
    state.add_edge(tasklet, 'b', map_exit, None, Memlet.simple(B_, 'i,j'))

    # Add outer edges
    state.add_edge(A_, None, map_entry, None, Memlet.simple(A_, '1:4,1:3'))
    state.add_edge(map_exit, None, B_, None, Memlet.simple(B_, '1:4,1:3'))

    # Left for debugging purposes
    mysdfg.draw_to_file()

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output[0:3, 0:2] - input[3:6, 4:6]) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
