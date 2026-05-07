# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


# Constructs an SDFG with two consecutive tasklets
def test():
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20
    input = dp.ndarray([n], dp.int32)
    output = dp.ndarray([n], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG
    mysdfg = SDFG('ctasklet')
    mysdfg.add_array('A', [N], dp.int32)
    mysdfg.add_array('B', [N], dp.int32)
    state = mysdfg.add_state()
    A_ = state.add_access('A')
    B_ = state.add_access('B')

    map_entry, map_exit = state.add_map('mymap', dict(i='0:N'))
    tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'i'))
    tasklet2 = state.add_tasklet('mytasklet2', {'c'}, {'d'}, 'd = 2*c')
    state.add_edge(tasklet, 'b', tasklet2, 'c', Memlet())
    state.add_edge(tasklet2, 'd', map_exit, None, Memlet.simple(B_, 'i'))

    # Add outer edges
    state.add_edge(A_, None, map_entry, None, Memlet.simple(A_, '0:N'))
    state.add_edge(map_exit, None, B_, None, Memlet.simple(B_, '0:N'))

    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(10 * input - output) / n
    assert diff <= 1e-5


if __name__ == '__main__':
    test()
