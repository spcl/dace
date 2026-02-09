# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.memlet import Memlet


@pytest.mark.gpu
def test_blockreduce():
    # Create SDFG
    sdfg = dace.SDFG('block_reduction')
    sdfg.add_array('A', (128, ), dace.float32)
    sdfg.add_array('B', (2, ), dace.float32)
    sdfg.add_transient('tA', (2, ), dace.float32)
    sdfg.add_transient('tB', (1, ), dace.float32)
    state = sdfg.add_state('a')

    # Nodes
    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('mymap', dict(bi='0:2'))
    mei, mxi = state.add_map('mymap2', dict(i='0:32'))
    red = state.add_reduce('lambda a,b: a+b', None, 0)
    red.implementation = 'CUDA (block)'
    tA = state.add_access('tA')
    tB = state.add_access('tB')
    write_tasklet = state.add_tasklet('writeout', {'inp'}, {'out'}, 'if i == 0: out = inp')

    # Edges
    state.add_edge(A, None, me, None, Memlet.simple(A, '0:128'))
    state.add_edge(me, None, mei, None, Memlet.simple(A, '(64*bi):(64*bi+64)'))
    state.add_edge(mei, None, tA, None, Memlet.simple('A', '(64*bi+2*i):(64*bi+2*i+2)'))
    state.add_edge(tA, None, red, None, Memlet.simple(tA, '0:2'))
    state.add_edge(red, None, tB, None, Memlet.simple(tB, '0'))
    state.add_edge(tB, None, write_tasklet, 'inp', Memlet.simple(tB, '0'))
    state.add_edge(write_tasklet, 'out', mxi, None, Memlet.simple('B', 'bi', num_accesses=-1))
    state.add_edge(mxi, None, mx, None, Memlet.simple(B, 'bi'))
    state.add_edge(mx, None, B, None, Memlet.simple(B, '0:2'))
    sdfg.fill_scope_connectors()

    Adata = np.random.rand(128).astype(np.float32)
    Bdata = np.random.rand(2).astype(np.float32)
    sdfg.apply_transformations(GPUTransformSDFG, options={'sequential_innermaps': False})
    sdfg(A=Adata, B=Bdata)

    B_regression = np.zeros(2, dtype=np.float32)
    B_regression[0] = np.sum(Adata[:64])
    B_regression[1] = np.sum(Adata[64:])

    diff = np.linalg.norm(B_regression - Bdata) / 128.0
    assert diff <= 1e-5


if __name__ == '__main__':
    test_blockreduce()
