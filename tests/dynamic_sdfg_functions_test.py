# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
def test_dynamic_sdfg_with_math_functions():
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20

    input = np.random.rand(n).astype(np.float32)
    output = dp.ndarray([n], dp.float32)
    output[:] = dp.float32(0)

    # Construct SDFG
    mysdfg = SDFG('mymodexp')
    mysdfg.add_array('A', [N], dp.float32)
    mysdfg.add_array('B', [N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    # Easy way to add a tasklet
    tasklet, map_entry, map_exit = state.add_mapped_tasklet('mytasklet', dict(i='0:N'),
                                                            dict(a=Memlet.simple(A, 'i % N')), 'b = math.exp(a)',
                                                            dict(b=Memlet.simple(B, 'i')))

    # Add outer edges
    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))
    state.add_edge(map_exit, None, B, None, Memlet.simple(B, '0:N'))

    mysdfg(A=input, B=output, N=n)
    #mymodexp_prog(input, output)

    diff = np.linalg.norm(np.exp(input) - output) / n
    assert diff <= 1e-5


if __name__ == "__main__":
    test_dynamic_sdfg_with_math_functions()
