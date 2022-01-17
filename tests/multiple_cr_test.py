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
    N.set(20)
    input = dp.ndarray([N], dp.int64)
    sum = dp.ndarray([1], dp.int64)
    product = dp.ndarray([1], dp.int64)
    input[:] = dp.int64(5)
    sum[:] = dp.int64(0)
    product[:] = dp.int64(1)

    # Construct SDFG
    mysdfg = SDFG('multiple_cr')
    state = mysdfg.add_state()
    A = state.add_array('A', [N], dp.int64)
    s = state.add_array('s', [1], dp.int64)
    p = state.add_array('p', [1], dp.int64)

    map_entry, map_exit = state.add_map('mymap', dict(i='0:N'))
    state.add_edge(A, None, map_entry, None, Memlet.simple(A, '0:N'))

    # Tasklet 1
    t1 = state.add_tasklet('task1', {'a'}, {'b'}, 'b = a')
    state.add_edge(map_entry, None, t1, 'a', Memlet.simple(A, 'i'))
    state.add_edge(t1, 'b', map_exit, None, Memlet.simple(s, '0', wcr_str='lambda a,b: a+b'))
    state.add_edge(map_exit, None, s, None, Memlet.simple(s, '0'))

    # Tasklet 2
    t2 = state.add_tasklet('task2', {'a'}, {'b'}, 'b = a')
    state.add_edge(map_entry, None, t2, 'a', Memlet.simple(A, 'i'))
    state.add_edge(t2, 'b', map_exit, None, Memlet.simple(p, '0', wcr_str='lambda a,b: a*b'))
    state.add_edge(map_exit, None, p, None, Memlet.simple(p, '0'))

    mysdfg(A=input, s=sum, p=product, N=N)

    diff_sum = 5 * 20 - sum[0]
    diff_prod = 5**20 - product[0]
    print("Difference:", diff_sum, '(sum)', diff_prod, '(product)')
    assert diff_sum <= 1e-5 and diff_prod <= 1e-5


if __name__ == '__main__':
    test()
