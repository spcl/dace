# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.memlet import Memlet

# Create nested SDFG
nsdfg = dace.SDFG('nested')
nsdfg.add_stream('nout', dace.int32)
state = nsdfg.add_state()
t = state.add_tasklet('task', set(), {'o'}, 'o = 2')
w = state.add_write('nout')
state.add_edge(t, 'o', w, None, Memlet.simple('nout', '0'))

# Create SDFG
sdfg = dace.SDFG('nested_stream_test')
state = sdfg.add_state('a')

# Nodes
sdfg.add_stream('SB', dace.int32, transient=True)
sdfg.add_array('B', (2, ), dace.int32)
SB = state.add_access('SB')
B = state.add_write('B')
n = state.add_nested_sdfg(nsdfg, None, set(), {'nout'})
state.add_edge(n, 'nout', SB, None, Memlet.simple('SB', '0'))
state.add_nedge(SB, B, Memlet.simple('B', '0'))


def test():
    print('Nested stream test')

    Bdata = np.zeros([2], np.int32)
    sdfg(B=Bdata)

    B_regression = np.array([2, 0], dtype=np.int32)

    diff = np.linalg.norm(B_regression - Bdata)
    print("Difference:", diff)
    assert diff == 0


if __name__ == "__main__":
    test()
