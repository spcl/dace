#!/usr/bin/env python3
import math
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

# Constructs an SDFG manually and runs it
if __name__ == '__main__':
    print('Dynamic SDFG test with vectorization and min')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)

    input = np.random.rand(N.get()).astype(np.float32)
    input2 = np.random.rand(N.get()).astype(np.float32)
    output = dp.ndarray([N], dp.float32)
    output[:] = dp.float32(0)

    # Construct SDFG
    mysdfg = SDFG('myvmin')
    state = mysdfg.add_state()
    A = state.add_array('A', [N], dp.float32)
    B = state.add_array('B', [N], dp.float32)
    C = state.add_array('C', [N], dp.float32)

    # Easy way to add a tasklet
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
        'mytasklet', dict(i='0:N:2'),
        dict(a=Memlet.simple(A, 'i', veclen=2),
             b=Memlet.simple(B, 'i', veclen=2)), 'c = min(a, b)',
        dict(c=Memlet.simple(C, 'i', veclen=2)))

    # Add outer edges
    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))
    state.add_edge(B, None, map_entry, None, Memlet.simple(B, '0:N'))
    state.add_edge(map_exit, None, C, None, Memlet.simple(C, '0:N'))

    # Left for debugging purposes
    mysdfg.draw_to_file()

    mysdfg(A=input, B=input2, C=output, N=N)

    diff = np.linalg.norm(np.minimum(input, input2) - output) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
