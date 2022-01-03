# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


# Constructs an SDFG manually and runs it
def test():
    print('SDFG direct compilation test')
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    N.set(20)
    input = dp.ndarray([N], dp.int32)
    output = dp.ndarray([N], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG
    mysdfg = SDFG('mysdfg')
    state = mysdfg.add_state()
    A_ = state.add_array('A', [N], dp.int32)  # NOTE: The names A and B are not
    B_ = state.add_array('B', [N], dp.int32)  # reserved, this is just to
    # clarify that
    # variable name != array name

    # Easy way to add a tasklet
    tasklet, map_entry, map_exit = state.add_mapped_tasklet('mytasklet', dict(i='0:N'), dict(a=Memlet.simple(A_, 'i')),
                                                            'b = 5*a', dict(b=Memlet.simple(B_, 'i')))
    # Alternatively (the explicit way):
    #map_entry, map_exit = state.add_map('mymap', dict(i='0:N'))
    #tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    #state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'i'))
    #state.add_edge(tasklet, 'b', map_exit, None, Memlet.simple(B_, 'i'))

    # Add outer edges
    state.add_edge(A_, None, map_entry, None, Memlet.simple(A_, '0:N'))
    state.add_edge(map_exit, None, B_, None, Memlet.simple(B_, '0:N'))

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(5 * input - output) / N.get()
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
