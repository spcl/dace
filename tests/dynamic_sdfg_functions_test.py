#!/usr/bin/env python3
import math
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

# For comparison
#@dp.program
#def mymodexp_prog(A,B):
#    @dp.map
#    def compute(i: _[0:N]):
#        a << A[i%N]
#        b >> B[i]
#
#        b = math.exp(a)

# Constructs an SDFG manually and runs it
if __name__ == '__main__':
    print('Dynamic SDFG test with math functions')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)

    input = np.random.rand(N.get()).astype(np.float32)
    output = dp.ndarray([N], dp.float32)
    output[:] = dp.float32(0)

    # Construct SDFG
    mysdfg = SDFG('mymodexp')
    state = mysdfg.add_state()
    A = state.add_array('A', [N], dp.float32)
    B = state.add_array('B', [N], dp.float32)

    # Easy way to add a tasklet
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
        'mytasklet', dict(i='0:N'), dict(a=Memlet.simple(A, 'i % N')),
        'b = math.exp(a)', dict(b=Memlet.simple(B, 'i')))

    # Add outer edges
    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))
    state.add_edge(map_exit, None, B, None, Memlet.simple(B, '0:N'))

    # Left for debugging purposes
    mysdfg.draw_to_file()

    mysdfg(A=input, B=output, N=N)
    #mymodexp_prog(input, output)

    diff = np.linalg.norm(np.exp(input) - output) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
