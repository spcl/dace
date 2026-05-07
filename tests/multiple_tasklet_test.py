# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


# Constructs an SDFG with multiple tasklets manually and runs it
def test():
    print('SDFG multiple tasklet test')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20
    input = dp.ndarray([n], dp.int32)
    output = dp.ndarray([n], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG
    mysdfg = SDFG('multiple_tasklets')
    mysdfg.add_array('A', [N], dp.int32)
    mysdfg.add_array('B', [N], dp.int32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    map_entry, map_exit = state.add_map('mymap', dict(i='0:N:2'))

    # Tasklet 1
    t1 = state.add_tasklet('task1', {'a'}, {'b'}, 'b = 5*a')
    state.add_edge(map_entry, None, t1, 'a', Memlet.simple(A, 'i'))
    state.add_edge(t1, 'b', map_exit, None, Memlet.simple(B, 'i'))

    # Tasklet 2
    t2 = state.add_tasklet('task2', {'a'}, {'b'}, 'b = a + a + a + a + a')
    state.add_edge(map_entry, None, t2, 'a', Memlet.simple(A, 'i+1'))
    state.add_edge(t2, 'b', map_exit, None, Memlet.simple(B, 'i+1'))

    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))
    state.add_edge(map_exit, None, B, None, Memlet.simple(B, '0:N'))

    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(5 * input - output) / n
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
