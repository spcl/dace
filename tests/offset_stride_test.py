# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


def test():
    """Multidimensional offset and stride test"""
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20
    input = dp.ndarray([n, n], dp.float32)
    output = dp.ndarray([4, 3], dp.float32)
    input[:] = (np.random.rand(n, n) * 5).astype(dp.float32.type)
    output[:] = dp.float32(0)

    # Construct SDFG
    mysdfg = SDFG('offset_stride')
    mysdfg.add_array('A', [6, 6], dp.float32, offset=[2, 3], strides=[N, 1], total_size=N * N)
    mysdfg.add_array('B', [3, 2], dp.float32, offset=[-1, -1], strides=[3, 1], total_size=12)

    state = mysdfg.add_state()
    A_ = state.add_access('A')
    B_ = state.add_access('B')

    map_entry, map_exit = state.add_map('mymap', [('i', '1:4'), ('j', '1:3')])
    tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = a')
    state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'i,j'))
    state.add_edge(tasklet, 'b', map_exit, None, Memlet.simple(B_, 'i,j'))

    # Add outer edges
    state.add_edge(A_, None, map_entry, None, Memlet.simple(A_, '1:4,1:3'))
    state.add_edge(map_exit, None, B_, None, Memlet.simple(B_, '1:4,1:3'))

    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(output[0:3, 0:2] - input[3:6, 4:6]) / n
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
