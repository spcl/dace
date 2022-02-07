# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


def test():
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
    mysdfg.add_array('A', [N], dp.float32)
    mysdfg.add_array('B', [N], dp.float32)
    mysdfg.add_array('C', [N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    C = state.add_access('C')

    tasklet, map_entry, map_exit = state.add_mapped_tasklet('mytasklet', dict(i='0:N:2'),
                                                            dict(a=Memlet.simple(A, 'i'), b=Memlet.simple(B, 'i')),
                                                            'c = min(a, b)', dict(c=Memlet.simple(C, 'i')))

    # Manually vectorize tasklet
    tasklet.in_connectors['a'] = dp.vector(dp.float32, 2)
    tasklet.in_connectors['b'] = dp.vector(dp.float32, 2)
    tasklet.out_connectors['c'] = dp.vector(dp.float32, 2)

    # Add outer edges
    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))
    state.add_edge(B, None, map_entry, None, Memlet.simple(B, '0:N'))
    state.add_edge(map_exit, None, C, None, Memlet.simple(C, '0:N'))

    mysdfg(A=input, B=input2, C=output, N=N)

    diff = np.linalg.norm(np.minimum(input, input2) - output) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
