# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.memlet import Memlet

# Create nested SDFG
nsdfg = dace.SDFG('nested')
nsdfg.add_array('nout', [1], dace.int32)
state = nsdfg.add_state()
t = state.add_tasklet('task', set(), {'o'}, 'o = 2')
w = state.add_write('nout')
state.add_edge(t, 'o', w, None, Memlet.simple('nout', '0', wcr_str='lambda a, b: a*b'))

# Create SDFG
sdfg = dace.SDFG('nested_cr_test')
state = sdfg.add_state('a')

# Nodes
sdfg.add_array('B', (1, ), dace.int32)
B = state.add_write('B')
n = state.add_nested_sdfg(nsdfg, None, set(), {'nout'})
state.add_edge(n, 'nout', B, None, Memlet.simple('B', '0', wcr_str='lambda a, b: a*b'))


def test():
    print('Nested conflict resolution test')

    Bdata = np.ones([1], np.int32)
    sdfg(B=Bdata)

    B_regression = np.array([2], dtype=np.int32)

    diff = B_regression[0] - Bdata[0]
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff == 0


if __name__ == "__main__":
    test()
