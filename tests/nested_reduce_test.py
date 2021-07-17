# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.memlet import Memlet

# Create SDFG
sdfg = dace.SDFG('nested_reduction')
state = sdfg.add_state('a')

# Nodes
A = state.add_array('A', (40, ), dace.float32)
B = state.add_array('B', (20, ), dace.float32)
me, mx = state.add_map('mymap', dict(i='0:20'))
red = state.add_reduce('lambda a,b: a+b', None, 0)

# Edges
state.add_edge(A, None, me, None, Memlet.simple(A, '0:40'))
state.add_edge(me, None, red, None, Memlet.simple(A, '(2*i):(2*i+2)'))
state.add_edge(red, None, mx, None, Memlet.simple(B, 'i'))
state.add_edge(mx, None, B, None, Memlet.simple(B, '0:20'))
sdfg.fill_scope_connectors()


def test():
    print('Nested reduction test')

    Adata = np.random.rand(40).astype(np.float32)
    Bdata = np.random.rand(20).astype(np.float32)
    sdfg(A=Adata, B=Bdata)

    B_regression = np.zeros(20, dtype=np.float32)
    B_regression[:] = Adata[::2]
    B_regression[:] += Adata[1::2]

    diff = np.linalg.norm(B_regression - Bdata) / 20.0
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
